"""Automated data ingest and detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from buildml.ingest.pipeline import ingest as ingest

__all__ = ["ingest"]


def __getattr__(name: str) -> Any:
    if name == "ingest":
        from buildml.ingest.pipeline import ingest as ingest_fn

        return ingest_fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
