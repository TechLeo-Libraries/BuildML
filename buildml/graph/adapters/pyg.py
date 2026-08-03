"""PyTorch Geometric adapter: GCN, GraphSAGE, GAT node classification."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.graph.data import edge_index_from_pairs
from buildml.graph.extras import require_pyg
from buildml.graph.types import PyGModel

_PYG_MODEL_MAP: dict[str, str] = {
    "gcn": "GCNConv",
    "graphsage": "SAGEConv",
    "gat": "GATConv",
}


class PyGNodeClassifier:
    """Industry PyG node classifier with train-mask cross-entropy only."""

    def __init__(
        self,
        *,
        pyg_model: PyGModel = "gcn",
        in_dim: int,
        n_classes: int,
        hidden_dim: int = 32,
        n_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 0.01,
        weight_decay: float = 5e-4,
        epochs: int = 80,
        random_state: int | None = 0,
    ) -> None:
        """Configure a PyG node classifier for train-mask cross-entropy.

        Validates architecture knobs and defers module construction until
        :meth:`fit` when PyTorch Geometric is available.

        Parameters
        ----------
        pyg_model:
            Convolution type: ``gcn``, ``graphsage``, or ``gat``.
        in_dim:
            Number of tabular node features.
        n_classes:
            Number of target classes (must be >= 2).
        hidden_dim, n_layers, heads, dropout:
            Architecture and regularisation knobs.
        learning_rate, weight_decay, epochs:
            Optimiser settings for Adam training.
        random_state:
            Optional seed for reproducible weight init.

        Raises
        ------
        ValidationError
            When dimensions, layer count, or model name are invalid.
        """
        if in_dim < 1:
            raise ValidationError("PyG in_dim must be >= 1.")
        if n_classes < 2:
            raise ValidationError("PyG requires at least 2 classes.")
        if n_layers not in {1, 2}:
            raise ValidationError("This surface supports n_layers in {1, 2} only.")
        model_key = str(pyg_model).lower().replace("-", "_")
        if model_key not in _PYG_MODEL_MAP:
            raise ValidationError(
                f"Unknown pyg_model={pyg_model!r}. Supported: {sorted(_PYG_MODEL_MAP)}."
            )
        self.pyg_model = model_key  # type: ignore[assignment]
        self.in_dim = int(in_dim)
        self.n_classes = int(n_classes)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.random_state = random_state
        self._module: Any = None
        self._torch = None
        self.train_losses_: list[float] = []

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        src: np.ndarray,
        dst: np.ndarray,
        *,
        directed: bool,
        train_mask: np.ndarray,
        class_to_index: dict[Any, int],
    ) -> PyGNodeClassifier:
        """Train the PyG module on train-masked node labels.

        Builds a sparse ``edge_index`` from endpoint arrays and minimises
        cross-entropy only on nodes marked True in ``train_mask``.

        Parameters
        ----------
        x:
            Node feature matrix of shape ``(n_nodes, in_dim)``.
        y:
            Raw target labels aligned to rows of ``x``.
        src, dst:
            Edge endpoint row indices.
        directed:
            When False, edges are symmetrised for message passing.
        train_mask:
            Boolean mask selecting supervised nodes.
        class_to_index:
            Mapping from raw label values to class indices.

        Returns
        -------
        PyGNodeClassifier
            Fitted classifier (``self``).

        Raises
        ------
        ValidationError
            When fewer than two train nodes are labeled.
        """
        torch = require_pyg(feature="Graph PyG node classification")
        self._torch = torch
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
            np.random.seed(int(self.random_state))

        device = torch.device("cpu")
        edge_index = torch.as_tensor(
            edge_index_from_pairs(src, dst, directed=directed),
            dtype=torch.long,
            device=device,
        )
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
        y_idx = np.asarray([class_to_index[v] for v in y.tolist()], dtype=np.int64)
        y_t = torch.as_tensor(y_idx, dtype=torch.long, device=device)
        mask = torch.as_tensor(train_mask, dtype=torch.bool, device=device)
        if int(mask.sum().item()) < 2:
            raise ValidationError("PyG fit needs at least 2 labeled train nodes.")

        module = _build_pyg_module(
            self.pyg_model,
            in_dim=self.in_dim,
            hidden_dim=self.hidden_dim,
            n_classes=self.n_classes,
            n_layers=self.n_layers,
            heads=self.heads,
            dropout=self.dropout,
        ).to(device)
        opt = torch.optim.Adam(
            module.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        self.train_losses_ = []
        module.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            logits = module(x_t, edge_index)
            loss = loss_fn(logits[mask], y_t[mask])
            loss.backward()
            opt.step()
            self.train_losses_.append(float(loss.detach().cpu().item()))
        module.eval()
        self._module = module
        return self

    def predict_proba(
        self,
        x: np.ndarray,
        src: np.ndarray,
        dst: np.ndarray,
        *,
        directed: bool,
    ) -> np.ndarray:
        """Return per-node class probabilities from a fitted PyG module.

        Runs a forward pass with the stored ``edge_index`` and applies softmax
        over logits for each node row.

        Parameters
        ----------
        x:
            Node feature matrix of shape ``(n_nodes, in_dim)``.
        src, dst:
            Edge endpoint row indices for message passing.
        directed:
            When False, edges are symmetrised before building ``edge_index``.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_nodes, n_classes)`` with softmax probabilities.

        Raises
        ------
        ValidationError
            When the classifier has not been fitted.
        """
        torch = self._torch or require_pyg(feature="Graph PyG node classification")
        if self._module is None:
            raise ValidationError("PyGNodeClassifier is not fitted.")
        device = next(self._module.parameters()).device
        edge_index = torch.as_tensor(
            edge_index_from_pairs(src, dst, directed=directed),
            dtype=torch.long,
            device=device,
        )
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
        self._module.eval()
        with torch.no_grad():
            logits = self._module(x_t, edge_index)
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
        return np.asarray(proba, dtype=np.float64)

    def predict(
        self,
        x: np.ndarray,
        src: np.ndarray,
        dst: np.ndarray,
        *,
        directed: bool,
    ) -> np.ndarray:
        """Return the argmax class index for each node.

        Delegates to :meth:`predict_proba` and selects the highest-probability
        class per row.

        Parameters
        ----------
        x:
            Node feature matrix of shape ``(n_nodes, in_dim)``.
        src, dst:
            Edge endpoint row indices for message passing.
        directed:
            When False, edges are symmetrised before building ``edge_index``.

        Returns
        -------
        numpy.ndarray
            Integer class indices of shape ``(n_nodes,)``.
        """
        proba = self.predict_proba(x, src, dst, directed=directed)
        return proba.argmax(axis=1)

    def to_state(self) -> dict[str, Any]:
        """Serialise hyperparameters and PyG module weights for bundle reload.

        Captures enough state to rebuild the same architecture and restore
        trained convolution weights via :meth:`from_state`.

        Returns
        -------
        dict[str, Any]
            Architecture knobs, optimiser settings, ``state_dict``, and
            per-epoch train losses.

        Raises
        ------
        ValidationError
            When the classifier has not been fitted.
        """
        torch = self._torch or require_pyg(feature="Graph PyG node classification")
        if self._module is None:
            raise ValidationError("PyGNodeClassifier is not fitted.")
        return {
            "pyg_model": self.pyg_model,
            "in_dim": self.in_dim,
            "n_classes": self.n_classes,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "heads": self.heads,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "random_state": self.random_state,
            "state_dict": {
                k: v.detach().cpu() for k, v in self._module.state_dict().items()
            },
            "train_losses": list(self.train_losses_),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> PyGNodeClassifier:
        """Reconstruct a fitted classifier from :meth:`to_state` output.

        Instantiates the same architecture, loads stored weights, and sets the
        module to evaluation mode for scoring.

        Parameters
        ----------
        state:
            Payload produced by :meth:`to_state` or loaded from a graph bundle.

        Returns
        -------
        PyGNodeClassifier
            Classifier with weights restored and set to eval mode.
        """
        require_pyg(feature="Graph PyG node classification")
        obj = cls(
            pyg_model=state["pyg_model"],  # type: ignore[arg-type]
            in_dim=int(state["in_dim"]),
            n_classes=int(state["n_classes"]),
            hidden_dim=int(state["hidden_dim"]),
            n_layers=int(state["n_layers"]),
            heads=int(state.get("heads", 4)),
            dropout=float(state["dropout"]),
            learning_rate=float(state["learning_rate"]),
            weight_decay=float(state["weight_decay"]),
            epochs=int(state["epochs"]),
            random_state=state.get("random_state"),
        )
        module = _build_pyg_module(
            obj.pyg_model,
            in_dim=obj.in_dim,
            hidden_dim=obj.hidden_dim,
            n_classes=obj.n_classes,
            n_layers=obj.n_layers,
            heads=obj.heads,
            dropout=obj.dropout,
        )
        module.load_state_dict(state["state_dict"])
        module.eval()
        obj._module = module
        obj.train_losses_ = list(state.get("train_losses") or [])
        return obj


def _build_pyg_module(
    pyg_model: str,
    *,
    in_dim: int,
    hidden_dim: int,
    n_classes: int,
    n_layers: int,
    heads: int,
    dropout: float,
) -> Any:
    """Construct a 1–2 layer PyG node classifier."""
    require_pyg(feature="Graph PyG node classification")
    torch = __import__("torch")
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv

    class _PyGNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pyg_model = pyg_model
            self.n_layers = n_layers
            self.dropout = torch.nn.Dropout(dropout)
            if pyg_model == "gcn":
                if n_layers == 1:
                    self.conv1 = GCNConv(in_dim, n_classes)
                    self.conv2 = None
                else:
                    self.conv1 = GCNConv(in_dim, hidden_dim)
                    self.conv2 = GCNConv(hidden_dim, n_classes)
            elif pyg_model == "graphsage":
                if n_layers == 1:
                    self.conv1 = SAGEConv(in_dim, n_classes)
                    self.conv2 = None
                else:
                    self.conv1 = SAGEConv(in_dim, hidden_dim)
                    self.conv2 = SAGEConv(hidden_dim, n_classes)
            elif pyg_model == "gat":
                if n_layers == 1:
                    self.conv1 = GATConv(
                        in_dim, n_classes, heads=1, concat=False, dropout=dropout
                    )
                    self.conv2 = None
                else:
                    self.conv1 = GATConv(
                        in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout
                    )
                    self.conv2 = GATConv(
                        hidden_dim * heads,
                        n_classes,
                        heads=1,
                        concat=False,
                        dropout=dropout,
                    )
            else:
                raise ValidationError(f"Unsupported pyg_model={pyg_model!r}.")

        def forward(self, x: Any, edge_index: Any) -> Any:
            h = self.conv1(x, edge_index)
            if self.conv2 is None:
                return h
            h = torch.relu(h)
            h = self.dropout(h)
            return self.conv2(h, edge_index)

    return _PyGNet()


def fit_pyg(
    *,
    x: np.ndarray,
    y_all: np.ndarray,
    src_fit: np.ndarray,
    dst_fit: np.ndarray,
    directed: bool,
    train_mask: np.ndarray,
    class_to_index: dict[Any, int],
    pyg_model: PyGModel,
    hidden_dim: int,
    n_layers: int,
    heads: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    random_state: int | None,
) -> PyGNodeClassifier:
    """Fit a PyG node classifier on mode-filtered edges.

    Convenience wrapper around :class:`PyGNodeClassifier` that wires
    hyperparameters and calls :meth:`PyGNodeClassifier.fit` on the
    train-induced subgraph.

    Parameters
    ----------
    x:
        Node feature matrix of shape ``(n_nodes, in_dim)``.
    y_all:
        Raw target labels for all nodes.
    src_fit, dst_fit:
        Edge endpoints retained after inductive/transductive fit filtering.
    directed:
        When False, edges are symmetrised for message passing.
    train_mask:
        Boolean mask selecting supervised nodes.
    class_to_index:
        Mapping from raw label values to class indices.
    pyg_model:
        Convolution type: ``gcn``, ``graphsage``, or ``gat``.
    hidden_dim, n_layers, heads:
        Architecture knobs passed to the classifier.
    epochs, learning_rate, weight_decay, dropout:
        Training hyperparameters.
    random_state:
        Optional seed for reproducible weight init.

    Returns
    -------
    PyGNodeClassifier
        Fitted classifier ready for :func:`predict_pyg_logits`.
    """
    clf = PyGNodeClassifier(
        pyg_model=pyg_model,
        in_dim=x.shape[1],
        n_classes=len(class_to_index),
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        heads=heads,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        random_state=random_state,
    )
    clf.fit(
        x,
        y_all,
        src_fit,
        dst_fit,
        directed=directed,
        train_mask=train_mask,
        class_to_index=class_to_index,
    )
    return clf


def predict_pyg_logits(
    clf: PyGNodeClassifier,
    *,
    x: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    directed: bool,
) -> np.ndarray:
    """Return class probabilities from a fitted PyG classifier.

    Thin wrapper around :meth:`PyGNodeClassifier.predict_proba` for scoring
    pipelines that already hold a fitted classifier.

    Parameters
    ----------
    clf:
        Fitted :class:`PyGNodeClassifier`.
    x:
        Node feature matrix of shape ``(n_nodes, in_dim)``.
    src, dst:
        Edge endpoint row indices for message passing.
    directed:
        When False, edges are symmetrised before building ``edge_index``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_nodes, n_classes)`` with softmax probabilities.
    """
    return clf.predict_proba(x, src, dst, directed=directed)
