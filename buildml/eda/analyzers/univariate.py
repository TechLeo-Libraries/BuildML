"""Describe each column on its own, before any relationships are considered.

Univariate analysis is the first pass: one column at a time, no pairs, no
target. It answers the questions you would ask about a single variable — where
is it centred, how spread out, how skewed, how many zeros, how many distinct
values.

That is deliberately modest, and it catches most data problems. A column that is
99% zeros, a "price" with negatives, a categorical with ten thousand levels: all
visible here, all fatal downstream, none of them requiring a model to find.

See Also
--------
buildml.eda.analyzers.bivariate : Relationships between columns.
buildml.eda.analyzers.quality : Completeness, duplicates, and type problems.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def analyze_univariate(frame: pd.DataFrame) -> dict[str, Any]:
    """Profile every column on its own terms, numeric or categorical.

    Splits by dtype and asks the questions appropriate to each. Numeric columns
    get location, spread, shape, and a normality screen. Categorical columns get
    cardinality, the common values, and entropy.

    Some of the less obvious numbers and why they are here. Skew above about 1
    in absolute value means a long tail, which is the usual signal that a log
    transform will help a linear model. Excess kurtosis says the tails are
    heavier than a normal distribution's, so outlier handling matters more.
    Entropy in bits measures how concentrated a distribution is: near zero means
    almost every row is the same value, and a column like that carries no signal
    regardless of what it is called. ``rare_level_rate`` is the share of
    categories appearing in under 1% of rows, which is what predicts trouble
    with one-hot encoding and unseen levels at score time.

    Parameters
    ----------
    frame:
        The data. Column names are stringified in the output, so an integer
        label becomes ``'0'``.

    Returns
    -------
    dict
        ``per_column`` — one entry per column, tagged ``kind`` as ``'numeric'``
        or ``'categorical'``. ``numeric_describe`` — the pandas summary with
        5th and 95th percentiles. ``categorical_uniques`` — distinct counts.

    Notes
    -----
    **The normality screen is a screen, not a verdict.** Shapiro-Wilk is used
    under 500 values and D'Agostino-Pearson above, both on a sample capped at
    5,000 for speed. With a large sample, any real data will be flagged
    non-normal — the test detects deviations far too small to matter. Read
    ``appears_non_normal`` alongside the skew and the histogram, not on its own.
    Each entry carries ``normality_assumptions`` spelling this out.

    **Statistics are computed after dropping missing values**, so ``mean``
    describes the rows that had a value. ``count`` says how many that was, and
    the quality analyzer says how many did not.

    **``zeros`` and ``negatives`` count the full column**, including rows
    excluded from the other statistics. They are separated because a zero and a
    missing value often mean the same thing in practice and are stored
    differently — a "zeros" count near the row count usually means a column
    where absence was encoded as 0.

    **The coefficient of variation is ``None`` when the mean is zero**, since
    the ratio is undefined there. For a column centred on zero, it would be
    meaningless anyway.

    See Also
    --------
    buildml.eda.findings.build_findings : Turning these numbers into advice.
    """
    numeric = frame.select_dtypes(include="number")
    categorical = frame.select_dtypes(exclude="number")
    per_column: dict[str, Any] = {}

    for col in numeric.columns.astype(str):
        s = numeric[col].dropna()
        entry: dict[str, Any] = {
            "kind": "numeric",
            "count": int(s.count()),
            "mean": _f(s.mean()),
            "std": _f(s.std()),
            "min": _f(s.min()),
            "q05": _f(s.quantile(0.05)),
            "q25": _f(s.quantile(0.25)),
            "median": _f(s.median()),
            "q75": _f(s.quantile(0.75)),
            "q95": _f(s.quantile(0.95)),
            "max": _f(s.max()),
            "iqr": _f(s.quantile(0.75) - s.quantile(0.25)),
            "skew": _f(s.skew()),
            "kurtosis": _f(s.kurtosis()),
            "zeros": int((numeric[col] == 0).sum()),
            "negatives": int((numeric[col] < 0).sum()),
            "cv": (
                _f(s.std() / s.mean())
                if s.mean() not in (0, None) and pd.notna(s.mean())
                else None
            ),
            "entropy_hist": _hist_entropy(s),
            "normality_method": None,
            "normality_sample_size": 0,
            "normality_stat": None,
            "normality_pvalue": None,
            "appears_non_normal": None,
            "normality_assumptions": (
                "Observations should be independent and measured on a continuous scale.",
                "The p-value is an unadjusted screening result and is sensitive to sample size.",
                "Non-significance does not prove that the data are normally distributed.",
            ),
        }
        if len(s) >= 8:
            # Normality screens (sample-capped for speed).
            sample = s if len(s) <= 5000 else s.sample(5000, random_state=0)
            entry["normality_sample_size"] = int(len(sample))
            try:
                if sample.nunique(dropna=True) > 1:
                    if len(sample) > 500:
                        entry["normality_method"] = "D'Agostino-Pearson normaltest"
                        stat, p = stats.normaltest(sample)
                    else:
                        entry["normality_method"] = "Shapiro-Wilk"
                        stat, p = stats.shapiro(sample)
                    entry["normality_stat"] = _f(stat)
                    entry["normality_pvalue"] = _f(p)
                    entry["appears_non_normal"] = bool(p is not None and p < 0.05)
                else:
                    entry["normality_reason"] = "Constant values do not support a normality test."
            except Exception as exc:  # noqa: BLE001
                entry["normality_reason"] = f"Normality screen failed: {exc}"
        else:
            entry["normality_sample_size"] = int(len(s))
            entry["normality_reason"] = "At least 8 non-missing observations are required."
        per_column[col] = entry

    for col in categorical.columns.astype(str):
        s = categorical[col]
        top = s.astype(str).value_counts(dropna=False).head(25)
        probs = s.value_counts(normalize=True, dropna=False)
        entropy = float(-(probs * np.log2(probs.replace(0, np.nan))).sum(skipna=True))
        per_column[col] = {
            "kind": "categorical",
            "nunique": int(s.nunique(dropna=True)),
            "top_values": {str(k): int(v) for k, v in top.items()},
            "mode": None if top.empty else str(top.index[0]),
            "entropy_bits": entropy,
            "rare_level_rate": float((probs < 0.01).sum() / len(probs)) if len(probs) else 0.0,
        }

    return {
        "per_column": per_column,
        "numeric_describe": numeric.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
        if not numeric.empty
        else {},
        "categorical_uniques": {
            str(c): int(categorical[c].nunique(dropna=True)) for c in categorical.columns
        },
    }


def _hist_entropy(s: pd.Series, bins: int = 20) -> float | None:
    if s.empty:
        return None
    hist, _ = np.histogram(s.to_numpy(), bins=bins)
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if len(p) else None


def _f(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
