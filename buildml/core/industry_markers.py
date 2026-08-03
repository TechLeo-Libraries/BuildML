"""Documented Python/platform markers for fragile industry extras.

Capability matrices and ``scripts/probe_industry_extras.py`` should surface the
same skip reasons. BuildML cannot ship missing upstream wheels: these helpers
only make marker-driven skips explicit at runtime.
"""

from __future__ import annotations

import sys
from typing import Any


def _py_lt_313() -> bool:
    return sys.version_info < (3, 13)


def _not_windows() -> bool:
    return sys.platform != "win32"


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


# package_import_name → marker predicate + human reason (mirrors pyproject.toml).
_MARKER_SPECS: dict[str, tuple[bool, str]] = {
    "lightfm": (
        _py_lt_313() and _not_windows(),
        "python_version < '3.13' and sys_platform != 'win32'",
    ),
    "giotto_tda": (
        _py_lt_313(),
        "python_version < '3.13'",
    ),
    "learn2learn": (
        _py_lt_313(),
        "python_version < '3.13'",
    ),
    "skope_rules": (
        _py_lt_313(),
        "python_version < '3.13'",
    ),
    "neuralforecast": (
        _py_lt_313(),
        "python_version < '3.13'",
    ),
    "autosklearn": (
        _is_linux(),
        "typically Linux-only (no reliable Win/macOS wheels)",
    ),
}


def marker_allows(package: str) -> bool:
    """Return whether the current interpreter satisfies a package's install marker.

    Unknown packages (no documented marker) are treated as allowed so callers
    can probe availability without special-casing every import name.

    Parameters
    ----------
    package:
        Import name (for example ``lightfm`` or ``giotto_tda``).

    Returns
    -------
    bool
        ``True`` when there is no marker, or when the marker predicate holds.
    """
    spec = _MARKER_SPECS.get(package)
    if spec is None:
        return True
    return bool(spec[0])


def marker_reason(package: str) -> str | None:
    """Return the documented environment-marker expression for ``package``.

    Mirrors the install marker recorded in ``pyproject.toml`` so probes and
    capability matrices can explain *why* a wheel was skipped on this platform.

    Parameters
    ----------
    package:
        Import name to look up.

    Returns
    -------
    str or None
        Marker text mirrored from ``pyproject.toml``, or ``None`` when unmarked.
    """
    spec = _MARKER_SPECS.get(package)
    if spec is None:
        return None
    return spec[1]


def platform_skip_entry(package: str, *, extra: str | None = None) -> dict[str, Any]:
    """Build a capability-matrix fragment for a possibly marker-skipped package.

    Call from domain catalogs so Session matrices can surface
    ``skipped_by_marker`` instead of silently looking unavailable.

    Parameters
    ----------
    package:
        Import name (for example ``lightfm`` or ``giotto_tda``).
    extra:
        Optional ``buildml[...]`` extra name for install hints.

    Returns
    -------
    dict
        Keys ``package``, ``marker``, ``marker_allows_install``,
        ``skipped_by_marker``, and optional ``extra``.
    """
    allows = marker_allows(package)
    reason = marker_reason(package)
    payload: dict[str, Any] = {
        "package": package,
        "marker": reason,
        "marker_allows_install": allows,
        "skipped_by_marker": bool(reason) and not allows,
    }
    if extra is not None:
        payload["extra"] = extra
    return payload


def listed_marker_packages() -> tuple[str, ...]:
    """Return import names that have documented platform or version markers.

    Useful for tests and probes that need to iterate the same table capability
    matrices advertise, without hard-coding the fragile package list twice.

    Returns
    -------
    tuple of str
        Sorted package import names from the internal marker table.
    """
    return tuple(sorted(_MARKER_SPECS))
