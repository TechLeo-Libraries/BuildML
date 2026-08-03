"""Decide whether a real LLM provider can be reached.

Most of the AI domain runs without any network dependency. The tool registry,
the egress rules, the transcript, and :class:`~buildml.ai.provider.MockProvider`
are all pure Python, which is what makes the domain testable offline and safe to
import unconditionally.

The ``openai`` client is needed only when a request actually goes to a hosted
model. These helpers draw that line: :func:`ai_available` answers the question,
:func:`require_ai_stack` and :func:`require_openai` enforce it with a message
naming the extra to install.

See Also
--------
buildml.ai.provider : Where the client is used.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from buildml.core.errors import MissingExtraError


def require_ai_stack(*, feature: str = "AI operator") -> None:
    """Refuse a Session AI entrypoint when no provider client is installed.

    Guards the Session methods that reach a hosted model. Checks package
    metadata rather than importing, so the cost is negligible on every call.

    Parameters
    ----------
    feature:
        What the caller was doing. Appears in the error message.

    Returns
    -------
    None
        Returns nothing on success; the value is the absence of an exception.

    Raises
    ------
    MissingExtraError
        If ``openai`` is not installed. Install with ``pip install buildml[ai]``.

    Notes
    -----
    **The mock provider path deliberately skips this**, so tests and offline
    walkthroughs exercise the same tool and egress machinery without needing a
    key or a network.

    See Also
    --------
    require_openai : Imports the client rather than just checking for it.
    ai_available : The boolean form.
    """
    if importlib.util.find_spec("openai") is None:
        raise MissingExtraError("ai", feature)


def require_openai(*, feature: str = "OpenAI provider") -> Any:
    """Import the OpenAI client, or explain how to install it.

    Imported lazily, so the AI domain stays importable: and the mock provider
    stays usable: on installations that have no LLM client at all.

    Parameters
    ----------
    feature:
        What the caller was doing. Appears in the error message.

    Returns
    -------
    module
        The ``openai`` module.

    Raises
    ------
    MissingExtraError
        If the client is missing or fails to load. Install with
        ``pip install buildml[ai]``.

    Notes
    -----
    ``OSError`` is caught alongside ``ImportError`` because a partially
    installed or platform-mismatched client can fail at the operating-system
    level rather than raising a clean import error. Either way the outcome for
    the caller is the same, and a named error beats a stack trace.

    See Also
    --------
    require_ai_stack : Checks without importing.
    """
    try:
        import openai
    except ImportError as exc:
        raise MissingExtraError("ai", feature) from exc
    except OSError as exc:
        raise MissingExtraError("ai", feature) from exc
    return openai


def ai_available() -> bool:
    """Report whether a real LLM provider can be used.

    Unlike :func:`require_ai_stack`, this actually imports the client, so a
    broken install reports ``False`` here rather than passing the check and
    failing later.

    Returns
    -------
    bool
        True when ``openai`` is installed and imports cleanly.

    Notes
    -----
    Use this to branch: offering AI features when available and hiding them
    otherwise: rather than to gate. Gating belongs in
    :func:`require_ai_stack`, where the failure carries an actionable message.

    **A ``False`` result does not mean the AI domain is unusable.** The mock
    provider needs nothing installed.

    See Also
    --------
    require_ai_stack : The gate.
    """
    if importlib.util.find_spec("openai") is None:
        return False
    try:
        import openai  # noqa: F401
    except (ImportError, OSError):
        return False
    return True
