"""Optional AI dependency gate."""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_ai_stack(*, feature: str = "AI operator") -> None:
    """Gate Session AI entrypoints.

    The core AI types (EgressLevel, ToolRegistry, MockProvider) do not require
    external LLM clients. This check verifies the optional OpenAI client is
    importable when a real provider is needed.

    Raises
    ------
    MissingExtraError
        When ``openai`` is not installed.
    """
    if importlib.util.find_spec("openai") is None:
        raise MissingExtraError("ai", feature)


def require_openai(*, feature: str = "OpenAI provider") -> Any:
    """Import and return ``openai``, or raise :class:`MissingExtraError`."""
    try:
        import openai
    except ImportError as exc:
        raise MissingExtraError("ai", feature) from exc
    except OSError as exc:
        raise MissingExtraError("ai", feature) from exc
    return openai


def ai_available() -> bool:
    """Return True when optional AI deps can be imported.

    MockProvider path does not require openai; this checks the full stack.
    """
    if importlib.util.find_spec("openai") is None:
        return False
    try:
        import openai  # noqa: F401
    except (ImportError, OSError):
        return False
    return True
