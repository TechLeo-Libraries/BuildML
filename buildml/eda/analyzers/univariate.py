"""Univariate analyzer with distributional diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def analyze_univariate(frame: pd.DataFrame) -> dict[str, Any]:
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
