"""Human teaching overlays for Session operations, split by domain."""

from __future__ import annotations

from buildml.explain.overlays.ai import _OPERATIONS as _AI
from buildml.explain.overlays.classical import _OPERATIONS as _CLASSICAL
from buildml.explain.overlays.dl import _OPERATIONS as _DL
from buildml.explain.overlays.rag import _OPERATIONS as _RAG
from buildml.explain.overlays.workflow import _OPERATIONS as _WORKFLOW
from buildml.explain.schemas import OperationSpec

_OPERATIONS: tuple[OperationSpec, ...] = (
    *_CLASSICAL,
    *_DL,
    *_RAG,
    *_AI,
    *_WORKFLOW,
)

__all__ = ["_OPERATIONS"]
