"""Checkpoint export and reattach."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from buildml.checkpoint.bundle import load_checkpoint, save_checkpoint

__all__ = ["load_checkpoint", "save_checkpoint"]


def __getattr__(name: str) -> Any:
    if name in {"load_checkpoint", "save_checkpoint"}:
        from buildml.checkpoint import bundle

        return getattr(bundle, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
