"""Compatibility shim — prefer :mod:`buildml.eda.profile`."""

from buildml.eda.profile import explore_dataset, summarize_dataset

__all__ = ["explore_dataset", "summarize_dataset"]
