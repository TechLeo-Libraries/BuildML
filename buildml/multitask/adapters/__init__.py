"""Multi-task backend adapters (sklearn / industry GBDT / torch multi-head)."""

from buildml.multitask.adapters.gbdt_multioutput import build_gbdt_estimator
from buildml.multitask.adapters.sklearn import build_sklearn_estimator
from buildml.multitask.adapters.torch_multihead import build_torch_estimator

__all__ = [
    "build_gbdt_estimator",
    "build_sklearn_estimator",
    "build_torch_estimator",
]
