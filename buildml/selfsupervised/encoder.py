"""Masked tabular autoencoder (sklearn MLP backbone; representation export)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPRegressor

from buildml.core.errors import ValidationError


class MaskedTabularEncoder(BaseEstimator, TransformerMixin):
    """Masked reconstruction encoder with bottleneck representation export.

    Fits a multi-output :class:`~sklearn.neural_network.MLPRegressor` to
    reconstruct randomly masked train features. ``transform`` returns the
    latent-layer activations (not the reconstruction).

    Honesty: this is a compact tabular pretext hook — not BERT-from-scratch,
    not contrastive SimCLR/MoCo product surface, and not a Torch FM zoo.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 16,
        hidden: tuple[int, ...] = (64,),
        mask_ratio: float = 0.15,
        n_mask_views: int = 3,
        max_iter: int = 200,
        random_state: int | None = 0,
    ) -> None:
        """Configure a masked tabular autoencoder for SSL pretext training.

        Sets MLP architecture and masking hyperparameters used during
        unsupervised reconstruction pretext on numeric tabular features.

        Parameters
        ----------
        latent_dim:
            Bottleneck width exported by :meth:`transform`.
        hidden:
            Hidden layer widths before the latent bottleneck.
        mask_ratio:
            Fraction of features masked per augmented view during fit.
        n_mask_views:
            Number of random mask views stacked per row during fit.
        max_iter:
            Maximum MLPRegressor iterations.
        random_state:
            Seed for mask sampling and MLP fitting.
        """
        self.latent_dim = int(latent_dim)
        self.hidden = tuple(int(h) for h in hidden)
        self.mask_ratio = float(mask_ratio)
        self.n_mask_views = int(n_mask_views)
        self.max_iter = int(max_iter)
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> MaskedTabularEncoder:
        """Fit the masked reconstruction MLP on augmented train features.

        Builds multiple masked views per row and learns to reconstruct the
        original features. Labels are ignored because pretext is unsupervised.

        Parameters
        ----------
        X:
            2D float feature matrix with at least two rows.
        y:
            Ignored; present for sklearn API compatibility.

        Returns
        -------
        MaskedTabularEncoder
            Fitted encoder with ``mlp_`` and ``reconstruction_mae_`` set.

        Raises
        ------
        ValidationError
            When ``X`` shape is invalid or hyperparameters are out of range.
        """
        del y  # pretext is unsupervised w.r.t. labels
        x = np.asarray(X, dtype=float)
        if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
            raise ValidationError(
                "MaskedTabularEncoder.fit requires a 2D feature matrix with "
                f"at least 2 rows (got shape={getattr(x, 'shape', None)})."
            )
        if not (0.0 < self.mask_ratio < 1.0):
            raise ValidationError("mask_ratio must be in (0, 1)")
        if self.latent_dim < 1:
            raise ValidationError("latent_dim must be >= 1")
        if self.n_mask_views < 1:
            raise ValidationError("n_mask_views must be >= 1")

        rng = np.random.default_rng(self.random_state)
        n, d = x.shape
        fill = np.nanmean(x, axis=0)
        fill = np.where(np.isfinite(fill), fill, 0.0)

        views_x: list[np.ndarray] = []
        views_y: list[np.ndarray] = []
        for _ in range(self.n_mask_views):
            mask = rng.random((n, d)) < self.mask_ratio
            # Ensure every row masks at least one feature when d > 1
            if d > 1:
                empty_rows = ~mask.any(axis=1)
                if empty_rows.any():
                    cols = rng.integers(0, d, size=int(empty_rows.sum()))
                    mask[np.where(empty_rows)[0], cols] = True
            masked = x.copy()
            masked[mask] = fill[np.where(mask)[1]]
            views_x.append(masked)
            views_y.append(x)

        x_aug = np.vstack(views_x)
        y_aug = np.vstack(views_y)
        layer_sizes = tuple(self.hidden) + (self.latent_dim,)
        self.n_features_in_ = int(d)
        self.fill_values_ = fill
        self.mlp_ = MLPRegressor(
            hidden_layer_sizes=layer_sizes,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=False,
        )
        self.mlp_.fit(x_aug, y_aug)
        self.reconstruction_mae_ = float(np.mean(np.abs(self.mlp_.predict(x) - x)))
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Export latent-layer activations for downstream heads.

        Returns bottleneck representations, not full feature reconstructions.

        Parameters
        ----------
        X:
            2D float matrix with ``n_features_in_`` columns.

        Returns
        -------
        numpy.ndarray
            Latent activations with shape ``(n_samples, latent_dim)``.

        Raises
        ------
        ValidationError
            When the encoder is not fitted or column count mismatches.
        """
        self._check_fitted()
        x = np.asarray(X, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValidationError(
                f"MaskedTabularEncoder.transform expected shape (n, {self.n_features_in_}), "
                f"got {x.shape}."
            )
        return self._latent_activations(x)

    def reconstruct(self, X: Any) -> np.ndarray:
        """Return reconstructed features for diagnostics.

        Session export uses :meth:`transform` representations by default; this
        path exposes full MLP reconstructions for quality checks.

        Parameters
        ----------
        X:
            2D float feature matrix matching ``n_features_in_``.

        Returns
        -------
        numpy.ndarray
            Reconstructed features with the same shape as ``X``.

        Raises
        ------
        ValidationError
            When the encoder is not fitted.
        """
        self._check_fitted()
        return np.asarray(self.mlp_.predict(np.asarray(X, dtype=float)), dtype=float)

    def _latent_activations(self, x: np.ndarray) -> np.ndarray:
        """Forward through hidden layers up to and including the latent layer."""
        mlp = self.mlp_
        # coefs_: list of (out_features, in_features); intercepts_ parallel
        # hidden_layer_sizes = (*hidden, latent_dim); output layer is reconstruction
        n_latent_layer = len(self.hidden)  # 0-based index of latent among hidden
        # sklearn MLP stores coefs_[i] with shape (n_fan_in, n_fan_out) and
        # forwards as activation @ coef + intercept (no transpose).
        activation = x
        for i, (coef, intercept) in enumerate(zip(mlp.coefs_, mlp.intercepts_, strict=False)):
            activation = activation @ coef + intercept
            # Apply hidden activation for all but the final output layer
            if i < len(mlp.coefs_) - 1:
                activation = _hidden_activation(activation, mlp.activation)
            if i == n_latent_layer:
                return np.asarray(activation, dtype=float)
        raise ValidationError("Failed to extract latent activations from fitted MLP.")

    def _check_fitted(self) -> None:
        if not hasattr(self, "mlp_"):
            raise ValidationError("MaskedTabularEncoder is not fitted.")


def _hidden_activation(values: np.ndarray, name: str) -> np.ndarray:
    if name == "relu":
        return np.maximum(values, 0.0)
    if name == "tanh":
        return np.tanh(values)
    if name == "logistic":
        return 1.0 / (1.0 + np.exp(-np.clip(values, -50, 50)))
    if name == "identity":
        return values
    # sklearn default for MLPRegressor is relu
    return np.maximum(values, 0.0)
