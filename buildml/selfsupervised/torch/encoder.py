"""Sklearn-compatible Torch SSL encoder wrapper (fit/transform/reconstruct)."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.dl.extras import require_torch
from buildml.selfsupervised.torch.trainer import SSLTrainConfig, SSLTrainResult, train_tabular_ssl


class TorchTabularSSLEncoder:
    """Tabular Torch SSL encoder with sklearn-style fit/transform API."""

    def __init__(
        self,
        *,
        method: str = "simclr_tabular",
        latent_dim: int = 16,
        hidden: tuple[int, ...] = (64,),
        projector_hidden: tuple[int, ...] = (64,),
        projector_dim: int = 32,
        predictor_hidden: tuple[int, ...] = (64,),
        epochs: int = 40,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        temperature: float = 0.5,
        mask_ratio: float = 0.3,
        random_state: int | None = 0,
        device: str = "cpu",
    ) -> None:
        self.method = method
        self.latent_dim = int(latent_dim)
        self.hidden = tuple(int(h) for h in hidden)
        self.projector_hidden = tuple(int(h) for h in projector_hidden)
        self.projector_dim = int(projector_dim)
        self.predictor_hidden = tuple(int(h) for h in predictor_hidden)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.temperature = float(temperature)
        self.mask_ratio = float(mask_ratio)
        self.random_state = random_state
        self.device = device
        self._train_result: SSLTrainResult | None = None

    @property
    def pretext_loss_(self) -> float | None:
        if self._train_result is None:
            return None
        return float(self._train_result.pretext_loss)

    @property
    def reconstruction_mae_(self) -> float | None:
        if self._train_result is None:
            return None
        return self._train_result.reconstruction_mae

    def fit(self, X: Any, y: Any = None) -> TorchTabularSSLEncoder:
        del y
        x = np.asarray(X, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] < 4:
            raise ValidationError(
                "TorchTabularSSLEncoder.fit requires >=4 rows and >=1 feature."
            )
        cfg = SSLTrainConfig(
            method=self.method,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            temperature=self.temperature,
            mask_ratio=self.mask_ratio,
            device=self.device,
            random_state=self.random_state,
        )
        self._train_result = train_tabular_ssl(
            x,
            method=self.method,
            latent_dim=self.latent_dim,
            hidden=self.hidden,
            projector_hidden=self.projector_hidden,
            projector_dim=self.projector_dim,
            predictor_hidden=self.predictor_hidden,
            config=cfg,
        )
        self.n_features_in_ = int(x.shape[1])
        return self

    def transform(self, X: Any) -> np.ndarray:
        self._check_fitted()
        torch = require_torch(feature="SSL transform")
        x = np.asarray(X, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValidationError(
                f"Expected shape (n, {self.n_features_in_}), got {x.shape}."
            )
        module = self._train_result.module  # type: ignore[union-attr]
        module.eval()
        device = next(module.parameters()).device
        with torch.no_grad():
            tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
            z = self._encode_batch(module, tensor)
            return z.cpu().numpy()

    def reconstruct(self, X: Any) -> np.ndarray:
        """Reconstruction diagnostic for generative methods."""
        self._check_fitted()
        if self.method not in {"mae_tabular", "vae_tabular"}:
            raise ValidationError(
                f"reconstruct() is only defined for mae_tabular/vae_tabular (got {self.method})."
            )
        torch = require_torch(feature="SSL reconstruct")
        x = np.asarray(X, dtype=np.float32)
        module = self._train_result.module  # type: ignore[union-attr]
        module.eval()
        device = next(module.parameters()).device
        with torch.no_grad():
            tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
            if self.method == "mae_tabular":
                _, recon = module(tensor)
            else:
                recon, _mu, _logvar, _z = module(tensor)
            return recon.cpu().numpy()

    def state_dict(self) -> dict[str, Any]:
        """Serializable Torch state for bundle v2."""
        self._check_fitted()
        torch = require_torch(feature="SSL state_dict")
        payload: dict[str, Any] = {
            "method": self.method,
            "latent_dim": self.latent_dim,
            "hidden": list(self.hidden),
            "projector_hidden": list(self.projector_hidden),
            "projector_dim": self.projector_dim,
            "predictor_hidden": list(self.predictor_hidden),
            "n_features_in_": self.n_features_in_,
            "module": self._train_result.module.state_dict(),  # type: ignore[union-attr]
        }
        if self._train_result.target_module is not None:  # type: ignore[union-attr]
            payload["target_module"] = self._train_result.target_module.state_dict()  # type: ignore[union-attr]
        return payload

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> TorchTabularSSLEncoder:
        """Restore encoder from bundle v2 payload."""
        from buildml.selfsupervised.torch import models

        method = str(payload["method"])
        enc = cls(
            method=method,
            latent_dim=int(payload["latent_dim"]),
            hidden=tuple(payload["hidden"]),
            projector_hidden=tuple(payload.get("projector_hidden") or (64,)),
            projector_dim=int(payload.get("projector_dim") or 32),
            predictor_hidden=tuple(payload.get("predictor_hidden") or (64,)),
        )
        enc.n_features_in_ = int(payload["n_features_in_"])
        n_features = enc.n_features_in_
        latent_dim = enc.latent_dim
        hidden = enc.hidden
        if method == "simclr_tabular":
            module = models.build_simclr_module(
                n_features,
                hidden=hidden,
                latent_dim=latent_dim,
                projector_hidden=enc.projector_hidden,
                projector_dim=enc.projector_dim,
            )
            target = None
        elif method == "byol_tabular":
            module, target = models.build_byol_module(
                n_features,
                hidden=hidden,
                latent_dim=latent_dim,
                projector_hidden=enc.projector_hidden,
                projector_dim=enc.projector_dim,
                predictor_hidden=enc.predictor_hidden,
            )
        elif method == "vicreg_tabular":
            module = models.build_vicreg_module(
                n_features,
                hidden=hidden,
                latent_dim=latent_dim,
                projector_hidden=enc.projector_hidden,
                projector_dim=enc.projector_dim,
            )
            target = None
        elif method == "mae_tabular":
            module = models.build_mae_module(n_features, hidden=hidden, latent_dim=latent_dim)
            target = None
        elif method == "vae_tabular":
            module = models.build_vae(n_features, hidden=hidden, latent_dim=latent_dim)
            target = None
        else:
            raise ValidationError(f"Cannot restore unknown Torch SSL method {method!r}")
        module.load_state_dict(payload["module"])
        if target is not None and "target_module" in payload:
            target.load_state_dict(payload["target_module"])
        enc._train_result = SSLTrainResult(
            module=module,
            target_module=target,
            pretext_loss=float("nan"),
            reconstruction_mae=None,
            method=method,
            latent_dim=latent_dim,
            n_features=n_features,
            config=SSLTrainConfig(method=method),
        )
        return enc

    def _encode_batch(self, module: Any, tensor: Any) -> Any:
        if self.method in {"simclr_tabular", "byol_tabular", "vicreg_tabular"}:
            z, _p = module(tensor)
            return z
        if self.method == "mae_tabular":
            z, _recon = module(tensor)
            return z
        if self.method == "vae_tabular":
            _recon, _mu, _logvar, z = module(tensor)
            return z
        raise ValidationError(f"Unknown method {self.method!r}")

    def _check_fitted(self) -> None:
        if self._train_result is None:
            raise ValidationError("TorchTabularSSLEncoder is not fitted.")
