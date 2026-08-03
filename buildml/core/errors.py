"""One base error, and four kinds of failure worth telling apart.

Everything BuildML raises inherits from :class:`BuildMLError`, so a caller who
only wants to know that *something* went wrong can catch one type. The
subclasses exist because the four failures call for genuinely different
responses: fix your input, fix your data source, install something, or rethink
your workflow.

:class:`LeakageError` is the one that distinguishes this library. It is raised
not because an operation would fail, but because it would succeed and produce a
score that flatters the model — fitting an imputer on the full frame, or
retaining a holdout row into a case base. Those operations run fine in plain
scikit-learn and quietly ruin the evaluation.

See Also
--------
buildml.core.validation : Where most :class:`ValidationError` raises originate.
"""

from __future__ import annotations


class BuildMLError(Exception):
    """The base for everything BuildML raises deliberately.

    Catch this to handle any BuildML failure without enumerating the subclasses.
    Anything escaping the library that is *not* one of these is a bug or an
    error from an underlying dependency.
    """


class ValidationError(BuildMLError):
    """Raised when the request itself cannot be honoured as written.

    Covers a misspelled role, a column that does not exist, a parameter outside
    its valid range, and operations attempted out of order — predicting before
    fitting, or splitting before roles are assigned. The fix is always in the
    calling code, and the message says what it is.
    """


class IngestError(BuildMLError):
    """Raised when the data cannot be read or inspected safely.

    Distinct from :class:`ValidationError` because the problem is the source
    rather than the request: an unreadable file, an unrecognised format, a
    schema that cannot be inferred. The call was reasonable; the data was not
    what it claimed to be.
    """


class MissingExtraError(BuildMLError):
    """Raised when a feature needs an optional dependency that is not installed.

    BuildML keeps its base install small, so deep learning, plotting, alternate
    engines, and the various adapters live behind extras. Rather than an
    ``ImportError`` naming some internal module, this names the feature you
    asked for and gives the exact command that enables it.

    Attributes
    ----------
    extra:
        The extra name from packaging metadata, also exposed as an attribute so
        callers can offer to install it programmatically.
    feature:
        What the user was trying to do.

    Notes
    -----
    **Always raise this with ``from exc``** at an ``ImportError`` boundary, so
    the underlying failure stays in the traceback. An extra can be installed and
    still fail to import for reasons of its own — a missing native library, a
    version conflict — and the original error is what distinguishes the two.

    Examples
    --------
    >>> exc = MissingExtraError("viz", "Plot boards")
    >>> print(exc)
    Plot boards requires the optional extra 'viz'. Install it with: pip install 'buildml[viz]'
    >>> exc.extra
    'viz'
    """

    def __init__(self, extra: str, feature: str) -> None:
        """Build the message from the extra name and the feature that needs it.

        Both arguments are kept as attributes as well as being interpolated into
        the message, so a caller catching the error can act on the extra name
        without parsing the text.

        Parameters
        ----------
        extra:
            The optional-extra name as declared in packaging metadata, for
            example ``'polars'`` or ``'viz'``. Must match the name in
            ``pyproject.toml``, since it is interpolated straight into the
            install command shown to the user.
        feature:
            A human-readable description of what needs it, used as the subject
            of the message — so phrase it as a noun, like ``'Evaluation plot
            boards'`` rather than ``'plotting'``.
        """
        self.extra = extra
        self.feature = feature
        message = (
            f"{feature} requires the optional extra '{extra}'. "
            f"Install it with: pip install 'buildml[{extra}]'"
        )
        super().__init__(message)


class LeakageError(BuildMLError):
    """Raised when an operation would let holdout information reach training.

    The refusal is the point. Fitting a scaler on the full frame, imputing with
    a mean computed over test rows, or cross-validating a frame that already has
    a split all run perfectly well and produce a score that will not survive
    production. This error stops them.

    Notes
    -----
    **Do not work around this by widening the fit scope.** If it fires, the
    workflow needs reordering — fit the transform on train, then apply it to the
    other partitions — not a way past the check.
    """
