"""Optional dependency gates for synthetic industry backends."""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

from buildml.core.errors import MissingExtraError
from buildml.dl.extras import _subprocess_import_ok


def sdv_spec_present() -> bool:
    """Check whether Python can locate ``sdv``.

This discovery check does not import the package or establish runtime usability.

Returns
-------
bool
    Whether an import specification exists for the package.
    """
    return importlib.util.find_spec("sdv") is not None


def sdv_available() -> bool:
    """Check runtime availability for sdv.

The package must be discoverable and import successfully. SDV and SDMetrics use an isolated import probe on Windows.

Returns
-------
bool
    Whether the required runtime import checks succeed.
    """
    if not sdv_spec_present():
        return False
    if sys.platform.startswith("win"):
        return _subprocess_import_ok("sdv")
    try:
        import sdv  # noqa: F401
    except Exception:
        return False
    return True


def sdmetrics_available() -> bool:
    """Check runtime availability for sdmetrics.

The package must be discoverable and import successfully. SDV and SDMetrics use an isolated import probe on Windows.

Returns
-------
bool
    Whether the required runtime import checks succeed.
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
    """Check runtime availability for great expectations.

The package must be discoverable and import successfully. SDV and SDMetrics use an isolated import probe on Windows.

Returns
-------
bool
    Whether the required runtime import checks succeed.
    """
    if importlib.util.find_spec("great_expectations") is None:
        return False
    try:
        import great_expectations  # noqa: F401
    except Exception:
        return False
    return True


def synthetic_industry_available() -> bool:
    """Check runtime availability for synthetic industry.

This delegates to the SDV runtime check used by the tabular synthesizers.

Returns
-------
bool
    Whether the required runtime import checks succeed.
    """
    return sdv_available()


def require_sdv(*, feature: str = "SDV tabular synthesizers") -> Any:
    """Import and return ``sdv`` for the requested feature.

Dependency failures handled by this helper are reported with an installation
hint identifying the required BuildML extra.

Parameters
----------
feature : str
    Feature name included in the missing-dependency message.

Returns
-------
Any
    Imported ``sdv`` module.

Raises
------
MissingExtraError
    If the dependency cannot be imported by this helper.
    """
    try:
        import sdv
    except ImportError as exc:
        raise MissingExtraError("synthetic-industry", feature) from exc
    except OSError as exc:
        raise MissingExtraError("synthetic-industry", feature) from exc
    return sdv


def require_sdmetrics(*, feature: str = "SDMetrics synthetic quality reports") -> Any:
    """Import and return ``sdmetrics`` for the requested feature.

Dependency failures handled by this helper are reported with an installation
hint identifying the required BuildML extra.

Parameters
----------
feature : str
    Feature name included in the missing-dependency message.

Returns
-------
Any
    Imported ``sdmetrics`` module.

Raises
------
MissingExtraError
    If the dependency cannot be imported by this helper.
    """
    try:
        import sdmetrics
    except ImportError as exc:
        raise MissingExtraError("synthetic-industry", feature) from exc
    except OSError as exc:
        raise MissingExtraError("synthetic-industry", feature) from exc
    return sdmetrics
