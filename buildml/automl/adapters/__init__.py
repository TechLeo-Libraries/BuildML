"""Optional industry AutoML adapters (``buildml[automl-industry]``)."""

from __future__ import annotations

from buildml.automl.adapters.autogluon import run_autogluon_adapter
from buildml.automl.adapters.flaml import run_flaml_adapter

__all__ = ["run_flaml_adapter", "run_autogluon_adapter"]
