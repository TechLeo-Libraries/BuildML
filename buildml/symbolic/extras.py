"""Optional dependency gates for symbolic industry backends.

Sklearn rule induction is always available. Industry paths add skope-rules,
imodels, optional Z3 verification, and torch neuro-symbolic bases behind
``buildml[symbolic-industry]`` and ``buildml[torch]``.

See Also
--------
buildml.symbolic.catalog.symbolic_capability_matrix : What is installed here.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError
from buildml.dl.extras import torch_available, torch_spec_available


def skope_rules_available() -> bool:
    """Return whether skope-rules (``skrules``) imports cleanly on this machine.

    Uses a real import probe because find_spec alone misses Python 3.13
    incompatibilities in older skope-rules releases.

    Returns
    -------
    bool
        ``True`` when ``SkopeRules`` can be imported.
    """
    if importlib.util.find_spec("skrules") is None:
        return False
    try:
        from skrules import SkopeRules  # noqa: F401
    except Exception:
        return False
    return True


def imodels_available() -> bool:
    """Return whether ``imodels`` is importable for RuleFit/BoostedRules export.

    Called from the capability matrix and fit routing so industry rule export
    is offered only when imodels actually imports on this machine.

    Returns
    -------
    bool
        ``True`` when imodels imports successfully.
    """
    if importlib.util.find_spec("imodels") is None:
        return False
    try:
        import imodels  # noqa: F401
    except Exception:
        return False
    return True


def z3_available() -> bool:
    """Return whether ``z3`` is importable for optional constraint verification.

    Gates the optional SAT check in :mod:`buildml.symbolic.adapters.z3_verify`
    without importing Z3 at module load time.

    Returns
    -------
    bool
        ``True`` when Z3 imports successfully.
    """
    if importlib.util.find_spec("z3") is None:
        return False
    try:
        import z3  # noqa: F401
    except Exception:
        return False
    return True


def symbolic_industry_available() -> bool:
    """Return whether any industry symbolic backend (skope-rules or imodels) is usable.

    Used when choosing the default industry backend in
    :func:`buildml.symbolic.catalog.symbolic_capability_matrix`.

    Returns
    -------
    bool
        ``True`` when at least one industry rule-induction extra imports cleanly.
    """
    return skope_rules_available() or imodels_available()


def require_skope_rules(*, feature: str = "SkopeRules rule induction") -> Any:
    """Import and return ``SkopeRules``, or raise :class:`MissingExtraError`.

    Adapter entry points call this at fit time so missing extras surface as
    actionable install guidance instead of opaque import errors.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    type
        The ``SkopeRules`` class.

    Raises
    ------
    MissingExtraError
        When skope-rules is not installed. Install with
        ``pip install 'buildml[symbolic-industry]'``.
    """
    try:
        from skrules import SkopeRules
    except ImportError as exc:
        raise MissingExtraError("symbolic-industry", feature) from exc
    return SkopeRules


def require_imodels(*, feature: str = "imodels interpretable rule export") -> Any:
    """Import and return ``imodels``, or raise :class:`MissingExtraError`.

    Called by the imodels adapter when RuleFit or BoostedRules export is
    requested on the industry backend path.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported imodels module.

    Raises
    ------
    MissingExtraError
        When imodels is not installed.
    """
    try:
        import imodels
    except ImportError as exc:
        raise MissingExtraError("symbolic-industry", feature) from exc
    return imodels


def require_z3(*, feature: str = "Z3 rule-set constraint verification") -> Any:
    """Import and return ``z3``, or raise :class:`MissingExtraError`.

    Reserved for callers that need direct Z3 access beyond the lite verifier in
    :func:`buildml.symbolic.adapters.z3_verify.verify_rule_constraints`.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported z3 module.

    Raises
    ------
    MissingExtraError
        When Z3 is not installed.
    """
    try:
        import z3
    except ImportError as exc:
        raise MissingExtraError("symbolic-industry", feature) from exc
    return z3


def require_torch_symbolic(*, feature: str = "Torch neuro-symbolic tabular models") -> Any:
    """Import torch for neuro-symbolic bases, or raise :class:`MissingExtraError`.

    Delegates to :func:`buildml.dl.extras.require_torch`.

    Parameters
    ----------
    feature:
        Capability name for the error message.

    Returns
    -------
    module
        The imported torch module.

    Raises
    ------
    MissingExtraError
        When torch is not installed.
    """
    from buildml.dl.extras import require_torch

    return require_torch(feature=feature)


def torch_neuro_available() -> bool:
    """Return whether torch imports cleanly for neuro-symbolic tabular models.

    Mirrors :func:`buildml.dl.extras.torch_available` for neuro-symbolic catalog
    entries and torch backend routing in fit.

    Returns
    -------
    bool
        ``True`` when :func:`buildml.dl.extras.torch_available` succeeds.
    """
    return torch_available()
