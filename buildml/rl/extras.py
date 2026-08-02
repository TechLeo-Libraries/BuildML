"""Optional dependency gates for ``buildml[rl]`` and ``buildml[rl-industry]``."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_gymnasium(*, feature: str = "Gymnasium RL loop") -> Any:
    """Import and return ``gymnasium``, or raise :class:`MissingExtraError`."""
    try:
        import gymnasium
    except ImportError as exc:
        raise MissingExtraError("rl", feature) from exc
    except OSError as exc:
        raise MissingExtraError("rl", feature) from exc
    return gymnasium


def gymnasium_available() -> bool:
    """Return True when ``gymnasium`` can be imported."""
    if importlib.util.find_spec("gymnasium") is None:
        return False
    try:
        import gymnasium  # noqa: F401
    except Exception:
        return False
    return True


def stable_baselines3_spec_present() -> bool:
    return importlib.util.find_spec("stable_baselines3") is not None


def stable_baselines3_available() -> bool:
    """True when stable-baselines3 is installed (find_spec only — no import probe)."""
    return stable_baselines3_spec_present()


def imitation_spec_present() -> bool:
    return importlib.util.find_spec("imitation") is not None


def imitation_available() -> bool:
    """True when imitation is installed (find_spec only — no import probe)."""
    return imitation_spec_present()


def rl_industry_available() -> bool:
    """Industry SB3 + imitation depth (buildml[rl-industry])."""
    return (
        gymnasium_available()
        and stable_baselines3_spec_present()
        and imitation_spec_present()
    )


def require_stable_baselines3(*, feature: str = "Stable-Baselines3 RL") -> Any:
    try:
        import stable_baselines3 as sb3
    except ImportError as exc:
        raise MissingExtraError("rl-industry", feature) from exc
    except OSError as exc:
        raise MissingExtraError("rl-industry", feature) from exc
    return sb3


def require_imitation(*, feature: str = "imitation BC/GAIL") -> Any:
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
    "imitation_spec_present",
    "require_gymnasium",
    "require_imitation",
    "require_stable_baselines3",
    "rl_industry_available",
    "stable_baselines3_available",
    "stable_baselines3_spec_present",
]
