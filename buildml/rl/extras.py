"""Check for the optional packages the environment paths need.

Gymnasium, Stable-Baselines3, and imitation are heavy and are not installed with
BuildML. The bandit and behavioural cloning paths never need them; everything
that touches an environment does. These functions are how the rest of the
package finds out which world it is in.

Two shapes, used for two purposes. **``*_available``** returns a boolean and
never raises: for deciding what to offer, as :mod:`buildml.rl.catalog` does.
**``require_*``** imports and returns the module, or raises
:class:`~buildml.core.errors.MissingExtraError` naming the extra to install :
for the moment a feature genuinely needs the package.

The availability checks are also deliberately layered. ``find_spec`` asks
whether a package is installed without executing it, which is fast and safe.
A full import additionally proves it *works*: a half-installed package with a
broken shared library is installed but unusable. Gymnasium gets the full probe
because a broken install there would surface deep inside a training loop;
Stable-Baselines3 and imitation get the cheap one, because importing them drags
in PyTorch and costs seconds.

See Also
--------
buildml.rl.catalog : Turns these checks into a capability report.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_gymnasium(*, feature: str = "Gymnasium RL loop") -> Any:
    """Import Gymnasium, or explain how to install it.

    Called at the point an environment is genuinely needed, so the error names
    the feature that wanted it rather than appearing as a bare import failure.

    Parameters
    ----------
    feature:
        What needed Gymnasium, quoted back in the error message. Callers pass
        something recognisable such as ``"fit_rl(mode='gym_reinforce')"``.

    Returns
    -------
    module
        The ``gymnasium`` module.

    Raises
    ------
    MissingExtraError
        If Gymnasium is absent or unusable, pointing at ``buildml[rl]``.
        ``OSError`` is caught alongside ``ImportError`` because a broken native
        dependency fails that way, and the remedy: reinstall the extra: is the
        same.

    See Also
    --------
    gymnasium_available : The non-raising check.
    """
    try:
        import gymnasium
    except ImportError as exc:
        raise MissingExtraError("rl", feature) from exc
    except OSError as exc:
        raise MissingExtraError("rl", feature) from exc
    return gymnasium


def gymnasium_available() -> bool:
    """Say whether Gymnasium is installed and actually works.

    Gets the full probe: spec lookup *and* import: unlike the other checks
    here. A Gymnasium that is present but broken would otherwise fail deep
    inside a training loop, long after the point where the diagnosis is easy.

    Returns
    -------
    bool
        ``True`` when Gymnasium can be imported without error.

    See Also
    --------
    require_gymnasium : The raising form, for when it is actually needed.
    """
    if importlib.util.find_spec("gymnasium") is None:
        return False
    try:
        import gymnasium  # noqa: F401
    except Exception:
        return False
    return True


def stable_baselines3_spec_present() -> bool:
    """Say whether Stable-Baselines3 is installed, without importing it.

    Importing Stable-Baselines3 pulls in PyTorch and costs seconds, which is too
    much for a capability check that runs on every catalog call.

    Returns
    -------
    bool
        ``True`` when the package is importable in principle. It does not prove
        the import would succeed: a broken install still reports ``True``.

    See Also
    --------
    require_stable_baselines3 : Where a real failure would surface.
    """
    return importlib.util.find_spec("stable_baselines3") is not None


def stable_baselines3_runtime_available() -> bool:
    """Return whether Stable-Baselines3 imports cleanly in a child process.

    Uses subprocess isolation so a broken torch/SB3 stack cannot hard-crash
    the host when capability matrices are built.
    """
    if not stable_baselines3_spec_present():
        return False
    from buildml.dl.extras import _subprocess_import_ok

    return _subprocess_import_ok("stable_baselines3")


def stable_baselines3_available() -> bool:
    """Say whether Stable-Baselines3 is usable for the ``'gym_sb3'`` mode.

    Runtime import probe (subprocess). Use
    :func:`stable_baselines3_spec_present` for cheap discovery-only flags.

    Returns
    -------
    bool
        ``True`` when the package imports successfully.

    See Also
    --------
    rl_industry_available : Whether the whole industry path is usable.
    """
    return stable_baselines3_runtime_available()


def imitation_spec_present() -> bool:
    """Say whether the ``imitation`` library is installed, without importing it.

    Kept to a spec lookup for the same reason as Stable-Baselines3: importing it
    is expensive, and a capability check should be cheap.

    Returns
    -------
    bool
        ``True`` when the package is importable in principle.

    See Also
    --------
    require_imitation : Where a real failure would surface.
    """
    return importlib.util.find_spec("imitation") is not None


def imitation_runtime_available() -> bool:
    """Return whether ``imitation`` imports cleanly in a child process."""
    if not imitation_spec_present():
        return False
    from buildml.dl.extras import _subprocess_import_ok

    return _subprocess_import_ok("imitation")


def imitation_available() -> bool:
    """Say whether the ``imitation`` library is usable for neural cloning.

    Runtime import probe (subprocess). Use :func:`imitation_spec_present` for
    cheap discovery-only flags. Backs ``'bc_mlp'`` / ``'gail_lite'`` offering.

    Returns
    -------
    bool
        ``True`` when the package imports successfully.

    See Also
    --------
    rl_industry_available : Whether the whole industry path is usable.
    """
    return imitation_runtime_available()


def rl_industry_available() -> bool:
    """Say whether the whole industry path is usable at runtime.

    ``buildml[rl-industry]`` is three packages, and every one of them is needed:
    Gymnasium supplies the environment, Stable-Baselines3 the deep RL
    algorithms, and ``imitation`` the neural cloning methods. A partial or
    broken install is not a partial capability, so this reports all-or-nothing
    on successful import probes (not find_spec alone).

    Returns
    -------
    bool
        ``True`` only when all three import successfully.

    See Also
    --------
    buildml.rl.catalog.rl_capability_matrix : Which package is missing.
    """
    return (
        gymnasium_available()
        and stable_baselines3_runtime_available()
        and imitation_runtime_available()
    )


def require_stable_baselines3(*, feature: str = "Stable-Baselines3 RL") -> Any:
    """Import Stable-Baselines3, or explain how to install it.

    This is where a broken install actually surfaces, since the availability
    checks only look for a spec.

    Parameters
    ----------
    feature:
        What needed it, quoted back in the error message.

    Returns
    -------
    module
        The ``stable_baselines3`` module.

    Raises
    ------
    MissingExtraError
        If it is absent or unusable, pointing at ``buildml[rl-industry]``.

    See Also
    --------
    stable_baselines3_available : The non-raising check.
    """
    try:
        import stable_baselines3 as sb3
    except ImportError as exc:
        raise MissingExtraError("rl-industry", feature) from exc
    except OSError as exc:
        raise MissingExtraError("rl-industry", feature) from exc
    return sb3


def require_imitation(*, feature: str = "imitation BC/GAIL") -> Any:
    """Import the ``imitation`` library, or explain how to install it.

    Called by the neural cloning methods at the point they need it.

    Parameters
    ----------
    feature:
        What needed it, quoted back in the error message.

    Returns
    -------
    module
        The ``imitation`` module.

    Raises
    ------
    MissingExtraError
        If it is absent or unusable, pointing at ``buildml[rl-industry]``.

    See Also
    --------
    imitation_available : The non-raising check.
    """
    try:
        import imitation
    except ImportError as exc:
        raise MissingExtraError("rl-industry", feature) from exc
    except OSError as exc:
        raise MissingExtraError("rl-industry", feature) from exc
    return imitation


__all__ = [
    "gymnasium_available",
    "imitation_available",
    "imitation_runtime_available",
    "imitation_spec_present",
    "require_gymnasium",
    "require_imitation",
    "require_stable_baselines3",
    "rl_industry_available",
    "stable_baselines3_available",
    "stable_baselines3_runtime_available",
    "stable_baselines3_spec_present",
]
