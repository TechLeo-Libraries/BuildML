"""Decide which columns a preprocessing step should touch, and reshape the frame.

Every preprocessing step faces the same question: given a dataset and possibly
an explicit list from the user, which columns do I actually operate on? Getting
that wrong is quietly destructive — scaling a customer ID produces a
meaningless float, imputing a target fabricates labels, and encoding a column
you meant to ignore inflates the frame with useless width.

The rule this module implements is that column *roles* answer the question by
default, and an explicit list overrides them. Columns marked ``target``,
``id``, ``group``, ``time``, ``weight``, or ``ignore`` are left alone unless you
name them, because there is almost always a reason they carry that role. Naming
a column explicitly is treated as informed consent: the role filter is skipped,
though the dtype check still applies so you cannot scale a text column by
accident.

Also here are the two structural operations — dropping columns and keeping only
some — which preserve row order and therefore leave any existing split valid.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.core.validation import validate_column_names
from buildml.data.dataset import Dataset
from buildml.ingest.detect import schema_from_dataframe

# Roles never mutated by default Session preprocess (scale/encode/impute/…).
# Explicit ``columns=…`` is the opt-in to force-include any of these.
DEFAULT_SKIP_ROLES: frozenset[ColumnRole] = frozenset(
    {
        ColumnRole.TARGET,
        ColumnRole.ID,
        ColumnRole.GROUP,
        ColumnRole.TIME,
        ColumnRole.WEIGHT,
        ColumnRole.IGNORE,
    }
)

ColumnKind = Literal["numeric", "categorical", "text", "any"]


def drop_columns(dataset: Dataset, columns: list[str] | tuple[str, ...]) -> Dataset:
    """Remove columns you have decided not to model.

    The usual reasons are leakage (a field that would not exist at prediction
    time), redundancy (two columns carrying the same information), or width (a
    high-cardinality identifier that would explode under encoding).

    Parameters
    ----------
    dataset:
        The source dataset, which is not modified.
    columns:
        Names to remove. Every name must exist — a typo raises rather than
        silently dropping nothing, since a silent no-op here means training on
        a column you believed was gone.

    Returns
    -------
    ~buildml.data.dataset.Dataset
        A new dataset without those columns, with their roles discarded too.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        A named column does not exist, or the request would leave the dataset
        with no columns at all.

    Notes
    -----
    Row identity and order are untouched, so an existing split stays valid and
    you can drop columns after splitting without re-splitting.

    Marking a column with the ``ignore`` role is the softer alternative: it is
    excluded from preprocessing and modelling but stays available for
    inspection and error analysis.

    See Also
    --------
    select_columns : The inverse — name what to keep instead.
    """
    cols = validate_column_names(columns, dataset.columns)
    remaining = [c for c in dataset.columns if c not in set(cols)]
    if not remaining:
        raise ValidationError("Cannot drop all columns from the dataset")

    frame = dataset._ensure_pandas().drop(columns=list(cols)).copy()
    roles = {k: v for k, v in dataset.roles.items() if k in remaining}
    return Dataset.from_transformed(
        dataset,
        frame,
        schema=schema_from_dataframe(frame),
        roles=roles,
    )


def select_columns(dataset: Dataset, columns: list[str] | tuple[str, ...]) -> Dataset:
    """Keep only the named columns and discard the rest.

    The inverse of :func:`drop_columns`, and the better choice when you know
    the short list you want rather than the long list you do not. It is also
    the safer of the two against schema drift: if tomorrow's data arrives with
    an extra column, selecting keeps your set fixed whereas dropping lets the
    newcomer through.

    Parameters
    ----------
    dataset:
        The source dataset, which is not modified.
    columns:
        Names to keep, and also the output order. Remember to include the
        target, the identifier, and anything a later split needs — they are not
        retained automatically.

    Returns
    -------
    ~buildml.data.dataset.Dataset
        A new dataset with only those columns, carrying over just their roles.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        A named column does not exist in the dataset.

    Notes
    -----
    Row identity and order are preserved, so an existing split remains valid.

    See Also
    --------
    drop_columns : Name what to remove instead.
    """
    cols = validate_column_names(columns, dataset.columns)
    frame: pd.DataFrame = dataset._ensure_pandas().loc[:, list(cols)].copy()
    roles = {k: v for k, v in dataset.roles.items() if k in cols}
    return Dataset.from_transformed(
        dataset,
        frame,
        schema=schema_from_dataframe(frame),
        roles=roles,
    )


def protected_role_columns(
    dataset: Dataset,
    *,
    skip_roles: frozenset[ColumnRole] | set[ColumnRole] | None = None,
) -> list[str]:
    """List the columns preprocessing will leave alone unless told otherwise.

    Useful for checking your own assumptions before a pipeline runs. If a
    column you expected to be scaled shows up in this list, its role is the
    reason, and the fix is either to correct the role or to name the column
    explicitly in the step.

    Parameters
    ----------
    dataset:
        The dataset whose roles are inspected.
    skip_roles:
        Which roles count as protected. Defaults to
        :data:`DEFAULT_SKIP_ROLES` — ``target``, ``id``, ``group``, ``time``,
        ``weight``, and ``ignore``. Pass a narrower set to ask a more specific
        question, such as "which columns are protected only because they are
        identifiers".

    Returns
    -------
    list of str
        Names of columns carrying a protected role, in role-assignment order.
        Columns with no role assigned are not protected and so are absent.

    See Also
    --------
    resolve_transform_columns : Applies this filter to pick a step's columns.
    """
    blocked = DEFAULT_SKIP_ROLES if skip_roles is None else frozenset(skip_roles)
    return [name for name, role in dataset.roles.items() if role in blocked]


def resolve_transform_columns(
    dataset: Dataset,
    train: pd.DataFrame,
    columns: list[str] | None,
    *,
    kind: ColumnKind = "numeric",
    require_dtype: bool = True,
    empty_message: str | None = None,
) -> list[str]:
    """Work out which columns a preprocessing step should operate on.

    Every fit function in this package routes through here, which is what keeps
    their behaviour consistent: scaling, encoding, imputation, and binning all
    make the same decision the same way.

    When ``columns`` is ``None`` the roles decide. Columns explicitly marked
    ``feature`` win if any exist; otherwise everything that is not protected by
    :data:`DEFAULT_SKIP_ROLES` is considered. The survivors are then filtered by
    dtype, so asking for numeric columns will not hand back a text column.

    When ``columns`` is given, the role filter is skipped entirely. Naming a
    column is treated as a deliberate override — that is how you scale
    something marked ``ignore`` when you genuinely mean to. The dtype check
    still runs unless you turn it off, because a named column of the wrong type
    is far more likely to be a mistake than an intention.

    Parameters
    ----------
    dataset:
        Supplies the role assignments and the full column list used for
        validating names.
    train:
        The training rows. Dtypes are read from here rather than from the full
        dataset, so the decision reflects the data the step will actually fit
        on.
    columns:
        An explicit list, or ``None`` to let roles and dtype decide.
    kind:
        Which dtypes qualify. ``'numeric'`` accepts pandas numeric dtypes;
        ``'categorical'`` accepts object, category, and string;
        ``'text'`` accepts string and object but excludes anything numeric;
        ``'any'`` applies no dtype filter, for steps that work on everything.
    require_dtype:
        Whether an explicitly named column of the wrong type is an error.
        Leave this on. Turning it off is for steps that coerce types
        themselves, such as imputation, which can fill a numeric column that
        pandas currently reads as object because it is entirely missing.
    empty_message:
        A step-specific message to raise when nothing resolves. Worth supplying
        — "no numeric columns available for scaling" tells the reader far more
        than a generic failure.

    Returns
    -------
    list of str
        Column names in a stable order: the order given when explicit,
        otherwise dataset column order.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        A named column does not exist, a named column fails the dtype check
        while ``require_dtype`` is on, nothing survived the filters, or ``kind``
        is not recognised.

    Notes
    -----
    The fallback when no ``feature`` roles are set is intentionally
    permissive — it lets a dataset work without any role assignment at all —
    but assigning roles is what makes preprocessing predictable. If a step
    keeps touching columns you did not expect, that is the signal to set roles.
    """
    if columns is not None:
        names = validate_column_names(columns, dataset.columns)
        if require_dtype and kind != "any":
            bad = [n for n in names if not _matches_kind(train[n], kind)]
            if bad:
                raise ValidationError(_dtype_error(kind, bad))
        return names

    feature_roles = [str(c) for c in dataset.role_columns(ColumnRole.FEATURE) if c in train.columns]
    if feature_roles:
        candidates = feature_roles
    else:
        blocked = set(protected_role_columns(dataset))
        candidates = [str(c) for c in train.columns if c not in blocked]

    names = [c for c in candidates if c in train.columns and _matches_kind(train[c], kind)]
    if not names:
        raise ValidationError(
            empty_message
            or _default_empty_message(kind)
        )
    return names


def _matches_kind(series: pd.Series, kind: ColumnKind) -> bool:
    if kind == "any":
        return True
    if kind == "numeric":
        return bool(pd.api.types.is_numeric_dtype(series))
    if kind == "categorical":
        return bool(
            pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(series)
        )
    if kind == "text":
        return bool(
            (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series))
            and not pd.api.types.is_numeric_dtype(series)
        )
    raise ValidationError(f"Unknown column kind '{kind}'")


def _dtype_error(kind: ColumnKind, bad: list[str]) -> str:
    shown = bad[:12]
    if kind == "numeric":
        return f"Requires numeric columns; non-numeric: {shown}"
    if kind == "categorical":
        return f"Requires categorical columns; non-categorical: {shown}"
    if kind == "text":
        return f"Requires text/object columns; invalid: {shown}"
    return f"Invalid columns for kind '{kind}': {shown}"


def _default_empty_message(kind: ColumnKind) -> str:
    if kind == "numeric":
        return (
            "No numeric feature columns available. "
            "Pass columns=... explicitly to include ignore/id roles."
        )
    if kind == "categorical":
        return (
            "No categorical feature columns available. "
            "Pass columns=... explicitly to include ignore/id roles."
        )
    if kind == "text":
        return (
            "No text/object feature columns available. "
            "Pass columns=... explicitly."
        )
    return (
        "No feature columns available. "
        "Pass columns=... explicitly to include ignore/id roles."
    )
