"""Industry federated-learning adapters (optional extras).

Lazy exports keep ``import buildml.federated.adapters`` from pulling Flower
into the host process before a runtime probe / ``require_flwr`` gate.
"""

from __future__ import annotations

from typing import Any

__all__ = ["fit_flower"]


def __getattr__(name: str) -> Any:
    if name == "fit_flower":
        from buildml.federated.adapters.flower import fit_flower

        return fit_flower
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
