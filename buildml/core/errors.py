"""BuildML error hierarchy."""

from __future__ import annotations


class BuildMLError(Exception):
    """Base error for all BuildML failures."""


class ValidationError(BuildMLError):
    """Raised when user input, schema, roles, or state is invalid."""


class IngestError(BuildMLError):
    """Raised when data cannot be ingested or inspected safely."""


class MissingExtraError(BuildMLError):
    """Raised when an optional dependency/extra is required but not installed.

    Parameters
    ----------
    extra:
        The optional-extra name as declared in packaging metadata
        (for example ``"polars"`` or ``"engines"``).
    feature:
        Human-readable feature that needs the extra.
    """

    def __init__(self, extra: str, feature: str) -> None:
        self.extra = extra
        self.feature = feature
        message = (
            f"{feature} requires the optional extra '{extra}'. "
            f"Install it with: pip install 'buildml[{extra}]'"
        )
        super().__init__(message)


class LeakageError(BuildMLError):
    """Raised when an operation would violate train/test fit-scope rules."""
