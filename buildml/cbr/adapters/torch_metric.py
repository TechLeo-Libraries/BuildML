"""Learn a space where distance reflects the target, not just the features.

Every other backend measures similarity in the feature space as given, which
assumes that geometric closeness in those coordinates means similarity for the
problem at hand. Often it does not: two features may be equally scaled and
wildly unequal in relevance, and standardisation cannot tell the difference.

Metric learning replaces the assumption with training. A small network is
trained to predict the target, and the layer *before* its output head is used as
the search space. Because that layer had to be informative enough to predict
from, points close together in it are close in a way that matters — the network
has effectively learned which features to weight.

The costs are real. Training requires torch and takes time. The learned space is
uninterpretable, so "why is this case similar?" stops having an answer in terms
of your columns, which sacrifices part of what makes case-based reasoning
attractive. And the encoder is fitted on train, so it can overfit like any other
model.

This is a light supervised encoder, not a contrastive or triplet-loss metric
learner. It is a useful improvement over raw feature distance when features are
unequally relevant, and it is not the state of the art in metric learning.

See Also
--------
buildml.cbr.extras.require_torch_cbr : The dependency gate.
buildml.cbr.adapters.sklearn_retrieval : Distance in the original space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from buildml.cbr.extras import require_torch_cbr
from buildml.core.errors import ValidationError


@dataclass
class TorchMetricEncoder:
    """A trained encoder plus the settings it was trained with.

    Holds the network and the hyperparameters, so a fitted encoder can be stored
    on a case base, saved in a bundle, and used to encode queries later.

    Attributes
    ----------
    hidden_dim:
        Width of the hidden layer.
    embed_dim:
        Dimension of the search space. Smaller spaces make distance more
        meaningful — high dimensions push everything toward equidistance — and
        risk discarding structure.
    epochs:
        Training passes over the data.
    learning_rate:
        Adam step size.
    device:
        Where training and encoding run.
    random_state:
        Seed, for reproducible training.
    n_features_, n_classes_, task_:
        The shape the encoder was fitted for. A query with a different feature
        count cannot be encoded.
    module_:
        The trained network, or ``None`` before fitting.

    Notes
    -----
    **The encoder is fitted on train and can overfit it.** An encoder that has
    memorised the training rows produces a space where those rows are neatly
    separated and new rows land arbitrarily. Watch holdout metrics, not the
    training loss.

    See Also
    --------
    build_torch_encoder : Constructing the network.
    fit_torch_encoder : Training it.
    """

    hidden_dim: int = 64
    embed_dim: int = 32
    epochs: int = 40
    learning_rate: float = 1e-3
    device: str = "cpu"
    random_state: int | None = 0
    n_features_: int = 0
    n_classes_: int = 0
    task_: str = "classification"
    module_: Any = field(default=None, repr=False)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Map feature rows into the learned space.

        Runs the trunk without the prediction head, which is the whole idea:
        the head's output is a prediction, while the layer beneath it is a
        representation that had to be informative enough to predict from.

        Parameters
        ----------
        x:
            Feature rows, matching the width the encoder was fitted on.

        Returns
        -------
        numpy.ndarray
            Embeddings, shape ``(n_rows, embed_dim)``.

        Raises
        ------
        ValidationError
            If the encoder has not been fitted.
        MissingExtraError
            If torch is not installed.

        Notes
        -----
        **Runs under ``no_grad`` in eval mode**, so no gradients are tracked and
        nothing about the network changes.

        **The output dimensions have no individual meaning.** They are learned
        coordinates; only distances between points in them are interpretable,
        and only as "similar for predicting this target".
        """
        torch = require_torch_cbr(feature="CBR learned-metric encoding")
        if self.module_ is None:
            raise ValidationError("TorchMetricEncoder is not fitted.")
        self.module_.eval()
        device = torch.device(self.device)
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32, device=device)
            emb = self.module_.encode(xt)
        return emb.cpu().numpy()


def build_torch_encoder(
    n_features: int,
    *,
    n_classes: int,
    task: str,
    hidden_dim: int = 64,
    embed_dim: int = 32,
    device: str = "cpu",
) -> Any:
    """Construct the untrained network: a trunk to embed, a head to supervise.

    The trunk maps features to the embedding that will be searched; the head
    maps that embedding to a prediction. Only the head's loss trains anything,
    and only the trunk's output is ever used for retrieval — the head exists
    purely to give the trunk something to learn from, and is discarded at query
    time.

    Parameters
    ----------
    n_features:
        Input width.
    n_classes:
        Output classes for classification. Ignored for regression.
    task:
        ``'classification'`` or ``'regression'``, deciding the head's shape.
    hidden_dim:
        Hidden layer width.
    embed_dim:
        Embedding width, and therefore the dimension of the search space.
    device:
        Where the network lives.

    Returns
    -------
    torch.nn.Module
        An untrained network exposing ``encode`` and ``forward``.

    Raises
    ------
    MissingExtraError
        If torch is not installed.

    Notes
    -----
    **Deliberately small.** Two hidden layers with the given widths is enough to
    reweight and recombine features, which is what metric learning needs here. A
    larger network would memorise the training rows and produce a space that
    separates them beautifully and generalises poorly.

    **The trunk ends in a ReLU**, so every embedding coordinate is
    non-negative. Points therefore occupy one orthant, which is harmless for
    Euclidean distance and means cosine similarity between any two embeddings is
    never negative.

    See Also
    --------
    fit_torch_encoder : Training it.
    """
    torch = require_torch_cbr(feature="CBR learned-metric encoder")
    task_key = str(task).lower()

    class _Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(n_features, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, embed_dim),
                torch.nn.ReLU(),
            )
            out_dim = 1 if task_key == "regression" else max(n_classes, 2)
            self.head = torch.nn.Linear(embed_dim, out_dim)

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            return self.trunk(x)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.encode(x))

    return _Encoder().to(torch.device(device))


def fit_torch_encoder(
    encoder: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    task: str,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    random_state: int | None = 0,
) -> Any:
    """Train the network so its embedding layer becomes a useful search space.

    Full-batch gradient descent against the task loss — mean squared error for
    regression, cross-entropy for classification. The predictions produced along
    the way are never used. What matters is the representation the trunk is
    forced to learn in order to make them.

    Parameters
    ----------
    encoder:
        The network from :func:`build_torch_encoder`.
    x_train:
        Training features. Standardise them first; the network trains far more
        readily on comparable scales.
    y_train:
        Training targets. Encoded class indices, or numeric values.
    task:
        ``'classification'`` or ``'regression'``, selecting the loss.
    epochs:
        Passes over the data.
    learning_rate:
        Adam step size.
    device:
        Where training runs.
    random_state:
        Seed for weight initialisation.

    Returns
    -------
    torch.nn.Module
        The trained network, left in eval mode.

    Raises
    ------
    MissingExtraError
        If torch is not installed.

    Notes
    -----
    **Full-batch, with the whole training set resident on the device.** Fine for
    the case-base sizes CBR works with, and a memory problem for very large
    ones.

    **No validation split and no early stopping.** The epoch count is the only
    control over fitting, so an over-trained encoder shows up as good training
    scores with poor holdout scores rather than as a warning here.

    **Train-only, always.** Fitting the encoder on data that includes the
    holdout leaks the target into the geometry of the search space, and every
    downstream metric becomes optimistic.

    See Also
    --------
    encode_with_torch : Using the trained encoder.
    """
    torch = require_torch_cbr(feature="CBR learned-metric training")
    if int(random_state) is not None:
        torch.manual_seed(int(random_state))
    x = np.asarray(x_train, dtype=float)
    y = np.asarray(y_train)
    dev = torch.device(device)
    encoder = encoder.to(dev)
    encoder.train()
    opt = torch.optim.Adam(encoder.parameters(), lr=float(learning_rate))
    xt = torch.as_tensor(x, dtype=torch.float32, device=dev)
    task_key = str(task).lower()
    if task_key == "regression":
        yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
        for _ in range(int(epochs)):
            opt.zero_grad()
            pred = encoder(xt).squeeze(-1)
            loss = torch.nn.functional.mse_loss(pred, yt)
            loss.backward()
            opt.step()
    else:
        yt = torch.as_tensor(y, dtype=torch.long, device=dev)
        for _ in range(int(epochs)):
            opt.zero_grad()
            logits = encoder(xt)
            loss = torch.nn.functional.cross_entropy(logits, yt)
            loss.backward()
            opt.step()
    encoder.eval()
    return encoder


def encode_with_torch(encoder: Any, x: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    """Embed rows with a trained encoder module.

    The function form of :meth:`TorchMetricEncoder.encode`, for call sites that
    hold a bare module rather than the dataclass wrapper. Cases and queries both
    go through here, which is what keeps them in the same space.

    Parameters
    ----------
    encoder:
        A trained network exposing ``encode``.
    x:
        Feature rows, matching the width the encoder was trained on.
    device:
        Where to run. Should match the device the encoder is on.

    Returns
    -------
    numpy.ndarray
        Embeddings, shape ``(n_rows, embed_dim)``.

    Raises
    ------
    MissingExtraError
        If torch is not installed.

    Notes
    -----
    **Preprocess queries exactly as the training features were preprocessed.**
    The encoder learned a mapping from standardised inputs; feeding it raw ones
    produces embeddings that sit somewhere unrelated to the case vectors, with
    no error to signal it.

    **Eval mode and no gradients**, so encoding never perturbs the network.
    """
    torch = require_torch_cbr(feature="CBR learned-metric encoding")
    encoder.eval()
    dev = torch.device(device)
    with torch.no_grad():
        xt = torch.as_tensor(np.asarray(x, dtype=float), dtype=torch.float32, device=dev)
        emb = encoder.encode(xt)
    return emb.cpu().numpy()
