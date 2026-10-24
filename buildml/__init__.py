"""
BuildML 2.x
===========

Manage machine-learning data, training, evaluation, and saved artifacts.

The public entry point is the :class:`~buildml.session.Session` API. Legacy 1.x
modules are retained under ``buildml._legacy`` and are not part of the supported
import graph.
"""

from typing import TYPE_CHECKING, Any

from buildml._version import __version__

if TYPE_CHECKING:
    from buildml.session import Session


def __getattr__(name: str) -> Any:
    """Load Session when requested, keeping dependency probes lightweight."""
    if name == "Session":
        from buildml.session import Session

        globals()[name] = Session
        return Session
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazy public entry point in interactive discovery."""
    return sorted(set(globals()) | {"Session"})

__author__ = "Leonard Onyiriuba"
__email__ = "leonard.c.onyiriuba@gmail.com"
__copyright__ = "Copyright (c) 2023-2026 Leonard Onyiriuba"
__license__ = "Apache-2.0"

__all__ = [
    "Session",
    "__version__",
]
