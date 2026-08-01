"""Column-level preparation helpers."""

from __future__ import annotations

import pandas as pd

from buildml.core.errors import ValidationError
from buildml.core.validation import validate_column_names
from buildml.data.dataset import Dataset
from buildml.ingest.detect import schema_from_dataframe


def drop_columns(dataset: Dataset, columns: list[str] | tuple[str, ...]) -> Dataset:
    """Return a new dataset with the given columns removed.

    Parameters
    ----------
    dataset:
        Source dataset.
    columns:
        Column names to drop.

    Returns
    -------
    Dataset
        New dataset (original is not mutated).

    Notes
    -----
    Roles for removed columns are discarded. Split membership remains valid
    because row identity/order is unchanged.
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
    """Return a new dataset keeping only the requested columns."""
    cols = validate_column_names(columns, dataset.columns)
    frame: pd.DataFrame = dataset._ensure_pandas().loc[:, list(cols)].copy()
    roles = {k: v for k, v in dataset.roles.items() if k in cols}
    return Dataset.from_transformed(
        dataset,
        frame,
        schema=schema_from_dataframe(frame),
        roles=roles,
    )
