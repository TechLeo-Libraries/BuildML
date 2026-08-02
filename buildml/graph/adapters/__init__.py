"""Graph ML industry backends (PyTorch Geometric)."""

from __future__ import annotations

__all__ = ["PyGNodeClassifier", "fit_pyg", "predict_pyg_logits"]

from buildml.graph.adapters.pyg import PyGNodeClassifier, fit_pyg, predict_pyg_logits
