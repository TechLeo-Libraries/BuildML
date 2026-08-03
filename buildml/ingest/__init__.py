"""Get data in, and know what you got.

The entry point is :func:`~buildml.ingest.pipeline.ingest`, which detects the
format, sizes the source, picks a mode and engine, loads, and hands back both
the dataset and a report of the decisions it made along the way.

The import is deferred through a module ``__getattr__``, so ``import buildml``
does not pull in pandas, pyarrow, and the optional engine adapters before anyone
has asked to read a file. That keeps startup fast for the many uses of BuildML
that never touch this package.

See Also
--------
buildml.ingest.pipeline : The ingest function itself.
buildml.ingest.detect : Format, schema, and scale detection.
"""

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
