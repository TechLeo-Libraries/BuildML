"""Look at the thing you are predicting, and at what seems to predict it.

Everything else in an EDA pass treats columns evenly. This one does not: the
target is what the model has to reproduce, and its shape determines which
algorithms are appropriate, which metrics mean anything, and whether the problem
is tractable at all.

The analysis branches on task, because the questions differ. For a regression
target: how skewed, what spread, which features correlate. For a classification
target: how many classes, how imbalanced, which features separate them.

The tests used are non-parametric throughout — Kruskal-Wallis rather than ANOVA,
Mann-Whitney rather than a t-test. Real feature distributions are rarely normal,
and a test that assumes normality on data that is not gives confident and wrong
p-values.

See Also
--------
buildml.eda.analyzers.bivariate : Association measured without regard to roles.
buildml.model.supervised : Where the target's shape determines the task.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats

from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset


def analyze_target(
    dataset: Dataset,
    frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Profile the target and rank the features that appear to move with it.

    Detects the task the same way the modelling code does — numeric with more
    than 15 distinct values is regression, anything else is classification — so
    the EDA and the model agree about what kind of problem this is. A target
    with 12 integer levels is treated as classification, which is usually right
    and occasionally not; check the returned ``summary.type`` if the problem is
    borderline.

    For a regression target: mean, standard deviation, and skew, plus Pearson
    correlations for numeric features and Kruskal-Wallis tests for categorical
    ones. Skew is the number to look at first, since a heavily skewed target is
    the most common reason a regression model underperforms and a log transform
    fixes it.

    For a classification target: class counts, rates, and the imbalance ratio
    between the most and least common class, plus Mann-Whitney tests for
    numeric features when the target is binary. The imbalance ratio decides
    whether accuracy is a usable metric — at 99 to 1, predicting the majority
    class always scores 99%.

    Parameters
    ----------
    dataset:
        The data, with a target role assigned. Without one, this returns an
        empty dict rather than raising, since unsupervised EDA is legitimate.
    frame:
        The frame to analyse — usually a sample, since these tests are only
        screening.
    feature_columns:
        Which columns count as features. Defaults to everything but the target,
        which will put identifiers in your association ranking.

    Returns
    -------
    dict
        Empty when there is no target. For regression: ``column``, ``summary``
        with ``type='regression_target'``, ``top_numeric_associations`` (up to
        20 by absolute correlation), ``categorical_effect_tests`` (up to 15 by
        p-value), ``n_rows``, ``non_missing_target_rows``. For classification:
        ``column``, ``summary`` with class counts, rates, ``n_classes``, and
        ``imbalance_ratio``, plus ``numeric_separation_tests`` for binary
        targets.

    Notes
    -----
    **The p-values are unadjusted.** Twenty features tested at the 0.05 level
    will produce about one significant result by chance. Treat the ranking as a
    reading order, not as evidence.

    **Association is not causation, and it is not importance either.** A feature
    correlated with the target may be a proxy for it, may be unavailable at
    score time, or may be leaking. The rank ordering here tells you what to look
    at, not what to keep.

    **Separation tests need a binary target.** Multi-class problems get class
    counts and rates only; per-feature separation for many classes needs a
    different framing.

    **Only the first target is analysed** when several columns hold the target
    role.

    **Some columns are silently skipped.** Categorical effect tests cover at
    most the first 12 categoricals, and only those with between 2 and 30 levels
    and at least 3 rows per group.

    See Also
    --------
    buildml.model.selection : Feature selection with validation behind it.
    """
    targets = dataset.role_columns(ColumnRole.TARGET)
    if not targets or targets[0] not in frame.columns:
        return {}
    target = targets[0]
    y = frame[target]
    features = {
        str(column)
        for column in (feature_columns if feature_columns is not None else frame.columns)
        if column in frame.columns and column != target
    }

    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 15:
        associations = []
        for col in frame.select_dtypes(include="number").columns.astype(str):
            if col not in features:
                continue
            corr = frame[[col, target]].corr().iloc[0, 1]
            if pd.notna(corr):
                associations.append({"feature": col, "pearson": float(corr)})
        associations.sort(key=lambda item: abs(item["pearson"]), reverse=True)

        # Categorical vs regression target: Kruskal-Wallis where feasible.
        cat_effects = []
        for col in frame.select_dtypes(exclude="number").columns.astype(str)[:12]:
            if col not in features:
                continue
            if frame[col].nunique(dropna=True) < 2 or frame[col].nunique(dropna=True) > 30:
                continue
            groups = [
                g[target].dropna().to_numpy()
                for _, g in frame[[col, target]].dropna().groupby(col)
                if len(g) >= 3
            ]
            if len(groups) >= 2:
                try:
                    stat, p = stats.kruskal(*groups)
                    cat_effects.append(
                        {"feature": col, "kruskal_h": float(stat), "pvalue": float(p)}
                    )
                except Exception:  # noqa: BLE001
                    pass
        cat_effects.sort(key=lambda item: item.get("pvalue", 1.0))
        return {
            "column": target,
            "summary": {
                "type": "regression_target",
                "mean": float(y.mean()),
                "std": float(y.std()),
                "skew": float(y.skew()),
            },
            "top_numeric_associations": associations[:20],
            "categorical_effect_tests": cat_effects[:15],
            "n_rows": int(len(frame)),
            "non_missing_target_rows": int(y.notna().sum()),
        }

    counts = y.astype(str).value_counts(dropna=False)
    rates = {str(k): float(v / len(y)) for k, v in counts.items()} if len(y) else {}
    imbalance_ratio = (
        max(rates.values()) / max(min(rates.values()), 1e-12) if rates else None
    )

    # Numeric feature separation via AUC-like rank proxy (Mann-Whitney for binary).
    separation = []
    if y.nunique(dropna=True) == 2:
        classes = list(pd.unique(y.dropna()))
        for col in frame.select_dtypes(include="number").columns.astype(str):
            if col not in features:
                continue
            a = frame.loc[y == classes[0], col].dropna()
            b = frame.loc[y == classes[1], col].dropna()
            if len(a) >= 5 and len(b) >= 5:
                try:
                    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                    separation.append(
                        {"feature": col, "mannwhitney_u": float(stat), "pvalue": float(p)}
                    )
                except Exception:  # noqa: BLE001
                    pass
        separation.sort(key=lambda item: item.get("pvalue", 1.0))

    return {
        "column": target,
        "summary": {
            "type": "classification_target",
            "class_counts": {str(k): int(v) for k, v in counts.items()},
            "class_rates": rates,
            "n_classes": int(y.nunique(dropna=True)),
            "imbalance_ratio": imbalance_ratio,
        },
        "numeric_separation_tests": separation[:20],
        "n_rows": int(len(frame)),
        "non_missing_target_rows": int(y.notna().sum()),
    }
