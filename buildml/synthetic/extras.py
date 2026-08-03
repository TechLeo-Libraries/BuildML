"""Optional dependency gates for synthetic industry backends."""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

from buildml.core.errors import MissingExtraError
from buildml.dl.extras import _subprocess_import_ok


def sdv_available() -> bool:
    """True when SDV imports cleanly (may pull torch: catch broken wheels).

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Returns
-------
bool
    ``True`` when the capability or dependency check succeeds.
    """
    if importlib.util.find_spec("sdv") is None:
        return False
    if sys.platform.startswith("win"):
        return _subprocess_import_ok("sdv")
    try:
        import sdv  # noqa: F401
    except Exception:
        return False
    return True


def sdmetrics_available() -> bool:
    """True when sdmetrics imports cleanly.

find_spec alone is insufficient: sdmetrics may import torch and raise
OSError on broken Windows wheels.

Returns
-------
bool
    ``True`` when the capability or dependency check succeeds.
    """
    if importlib.util.find_spec("sdmetrics") is None:
        return False
    if sys.platform.startswith("win"):
        return _subprocess_import_ok("sdmetrics")
    try:
        import sdmetrics  # noqa: F401
    except Exception:
        return False
    return True


def great_expectations_available() -> bool:
    """Return whether great expectations optional dependencies are installed and usable.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Returns
-------
bool
    ``True`` when the capability or dependency check succeeds.
    """
    if importlib.util.find_spec("great_expectations") is None:
        return False
    try:
        import great_expectations  # noqa: F401
    except Exception:
        return False
    return True


def synthetic_industry_available() -> bool:
    """True when SDV (CTGAN/TVAE/CopulaGAN) is importable.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Returns
-------
bool
    ``True`` when the capability or dependency check succeeds.
    """
    return sdv_available()


def require_sdv(*, feature: str = "SDV tabular synthesizers") -> Any:
    """Import optional dependency for sdv or raise MissingExtraError.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
feature:
    Capability name included in missing-extra error messages.

Returns
-------
Any
    Adapter-specific estimator or model object.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    try:
        import sdv
    except ImportError as exc:
        raise MissingExtraError("synthetic-industry", feature) from exc
    except OSError as exc:
        raise MissingExtraError("synthetic-industry", feature) from exc
    return sdv


def require_sdmetrics(*, feature: str = "SDMetrics synthetic quality reports") -> Any:
    """Import optional dependency for sdmetrics or raise MissingExtraError.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
feature:
    Capability name included in missing-extra error messages.

Returns
-------
Any
    Adapter-specific estimator or model object.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    try:
        import sdmetrics
    except ImportError as exc:
        raise MissingExtraError("synthetic-industry", feature) from exc
    except OSError as exc:
        raise MissingExtraError("synthetic-industry", feature) from exc
    return sdmetrics
