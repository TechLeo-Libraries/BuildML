"""Meta-learning backend adapters (sklearn / torch ProtoNet / industry MAML)."""

from __future__ import annotations

__all__ = [
    "TabularProtoNet",
    "build_tabular_classifier",
    "meta_train_maml",
    "meta_train_prototypical_torch",
    "meta_train_reptile",
]
