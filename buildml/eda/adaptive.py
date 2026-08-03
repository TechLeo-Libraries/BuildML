"""Decide which charts are worth drawing before drawing any of them.

Plotting every column of a fifty-column frame produces a report nobody reads.
Worse, it produces charts that actively mislead: a bar chart of a column with
2,000 categories, a scatter plot of an identifier, a histogram of a constant.

So the plan is computed first, as data, and rendering happens afterwards.
Separating the two means the decisions can be inspected and tested without
importing Matplotlib or generating a single image: and a caller who wants
different charts can edit the plan rather than the renderer.

The choices follow from what each column is. A numeric column with a long tail
wants a log-scaled histogram. A categorical with 30 levels wants a bar chart; one
with 3,000 wants nothing. A pair of numerics against a target wants a scatter.
Roles matter too: identifiers, ignored columns, and the target itself are
excluded from feature plots, because a chart of a row ID is noise on the page.

See Also
--------
buildml.eda.visualize : Rendering a plan.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset


def build_adaptive_plan(
    dataset: Dataset,
    frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    max_plots: int = 24,
) -> list[dict[str, Any]]:
    """Work out which charts this particular dataset warrants.

    Filters to columns worth plotting, then matches each to a chart type that
    suits it. Columns are excluded when their role makes them meaningless as
    features: target, identifier, ignored, group, time, weight: or when they
    hold a single distinct value, since a chart of a constant is a rectangle.

    The cap is what keeps a report readable. Beyond a couple of dozen charts,
    each additional one reduces the chance any of them is looked at, so the plan
    is truncated at the most informative ones rather than extended.

    Parameters
    ----------
    dataset:
        The data, with roles assigned. Roles drive most of the exclusions, so a
        dataset without them will plan charts for identifiers.
    frame:
        The frame to plan against: usually a sample. Dtypes and cardinality are
        read from here.
    feature_columns:
        Restrict to these columns. Defaults to everything, filtered by role.
    max_plots:
        Ceiling on the plan length. Twenty-four fills a scrollable report
        without exhausting the reader.

    Returns
    -------
    list of dict
        Plot specifications, each naming a chart type and the columns it uses.
        Consumed by :func:`~buildml.eda.visualize.render_adaptive_plots`.

    Notes
    -----
    **This renders nothing.** No Matplotlib import, no images. Inspect or edit
    the plan before rendering when you want different charts.

    **Roles are load-bearing here.** Without them, the planner cannot tell an
    identifier from a feature, and the resulting report is mostly charts of
    unique values.

    See Also
    --------
    buildml.eda.visualize.render_adaptive_plots : Executing the plan.
    """
    plan: list[dict[str, Any]] = []
    invalid_feature_roles = {
        ColumnRole.TARGET,
        ColumnRole.ID,
        ColumnRole.IGNORE,
        ColumnRole.GROUP,
        ColumnRole.TIME,
        ColumnRole.WEIGHT,
    }
    candidates = feature_columns if feature_columns is not None else list(frame.columns.astype(str))
    eligible = [
        str(column)
        for column in candidates
        if column in frame.columns
        and dataset.roles.get(str(column)) not in invalid_feature_roles
        and frame[column].nunique(dropna=False) > 1
    ]
    numeric = [
        column for column in eligible if pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical = [
        str(c)
        for c in eligible
        if c not in numeric
        and (
            pd.api.types.is_object_dtype(frame[c])
            or isinstance(frame[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(frame[c])
            or pd.api.types.is_bool_dtype(frame[c])
        )
    ]
    datetime_cols = list(
        frame.select_dtypes(include=["datetime", "datetimetz"]).columns.astype(str)
    )
    target_cols = dataset.role_columns(ColumnRole.TARGET)
    target = target_cols[0] if target_cols else None

    plan.append({"kind": "missingness_matrix", "title": "Missingness map", "priority": 100})
    plan.append({"kind": "dtype_overview", "title": "Schema / dtype mosaic", "priority": 95})

    if numeric:
        plan.append(
            {
                "kind": "correlation_heatmap",
                "title": "Numeric association heatmap",
                "columns": numeric[:40],
                "priority": 90,
            }
        )
        for col in numeric[:8]:
            plan.append(
                {
                    "kind": "numeric_distribution",
                    "title": f"Distribution · {col}",
                    "column": col,
                    "priority": 80,
                }
            )
        if len(numeric) >= 2:
            plan.append(
                {
                    "kind": "pair_sample",
                    "title": "Top numeric relationships",
                    "columns": numeric[:5],
                    "priority": 75,
                }
            )

    for col in categorical[:8]:
        nunique = int(frame[col].nunique(dropna=True))
        kind = "categorical_bars" if nunique <= 30 else "categorical_topk"
        plan.append(
            {
                "kind": kind,
                "title": f"Category profile · {col}",
                "column": col,
                "priority": 70,
            }
        )

    if target is not None:
        plan.append(
            {
                "kind": "target_balance",
                "title": f"Target balance · {target}",
                "column": target,
                "priority": 92,
            }
        )
        for col in numeric[:6]:
            if col == target:
                continue
            plan.append(
                {
                    "kind": "target_vs_numeric",
                    "title": f"{col} by {target}",
                    "feature": col,
                    "target": target,
                    "priority": 85,
                }
            )
        for col in categorical[:5]:
            if col == target:
                continue
            plan.append(
                {
                    "kind": "target_vs_categorical",
                    "title": f"{target} vs {col}",
                    "feature": col,
                    "target": target,
                    "priority": 84,
                }
            )

    for col in datetime_cols[:3]:
        plan.append(
            {
                "kind": "temporal_density",
                "title": f"Temporal density · {col}",
                "column": col,
                "priority": 65,
            }
        )

    plan.append({"kind": "outlier_board", "title": "Outlier board (IQR)", "priority": 60})

    plan.sort(key=lambda item: int(item.get("priority", 0)), reverse=True)
    return plan[:max_plots]
