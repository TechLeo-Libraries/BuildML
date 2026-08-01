"""Bivariate analyzer: correlations, MI, categorical associations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder


def analyze_bivariate(
    frame: pd.DataFrame,
    target: str | None = None,
    *,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Measure pairwise associations over role-valid features.

    ``feature_columns`` prevents targets, identifiers, ignored fields, and
    constants from silently entering feature rankings.  The target is added
    only for target-specific mutual information.
    """
    features = [
        str(column)
        for column in (feature_columns if feature_columns is not None else frame.columns)
        if column in frame.columns and column != target
    ]
    analysis = frame[features]
    numeric = analysis.select_dtypes(include="number")
    result: dict[str, Any] = {
        "pearson": {},
        "spearman": {},
        "kendall_top_pairs": [],
        "top_abs_pearson_pairs": [],
        "categorical_pairs": [],
        "mutual_information_vs_target": {},
    }

    if numeric.shape[1] >= 2:
        pearson = numeric.corr(method="pearson")
        spearman = numeric.corr(method="spearman")
        result["pearson"] = pearson.replace({np.nan: None}).to_dict()
        result["spearman"] = spearman.replace({np.nan: None}).to_dict()
        pairs: list[dict[str, Any]] = []
        cols = list(pearson.columns.astype(str))
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                val = pearson.loc[a, b]
                if pd.notna(val):
                    pairs.append({"a": a, "b": b, "corr": float(val)})
        pairs.sort(key=lambda item: abs(item["corr"]), reverse=True)
        result["top_abs_pearson_pairs"] = pairs[:30]

        # Kendall on top candidates only (expensive).
        top_cols = list({p["a"] for p in pairs[:12]} | {p["b"] for p in pairs[:12]})
        if len(top_cols) >= 2:
            kendall = numeric[top_cols].corr(method="kendall")
            k_pairs = []
            for i, a in enumerate(top_cols):
                for b in top_cols[i + 1 :]:
                    val = kendall.loc[a, b]
                    if pd.notna(val):
                        k_pairs.append({"a": a, "b": b, "corr": float(val)})
            k_pairs.sort(key=lambda item: abs(item["corr"]), reverse=True)
            result["kendall_top_pairs"] = k_pairs[:20]

    cats = [
        str(c)
        for c in analysis.columns
        if not pd.api.types.is_numeric_dtype(analysis[c])
        and analysis[c].nunique(dropna=True) <= 40
    ]
    for i, a in enumerate(cats[:8]):
        for b in cats[i + 1 : 8]:
            ct = pd.crosstab(frame[a].astype(str), frame[b].astype(str))
            result["categorical_pairs"].append({"a": a, "b": b, "cramers_v": _cramers_v(ct)})

    if target and target in frame.columns:
        result["mutual_information_vs_target"] = _mi_vs_target(
            frame[[*features, target]], target
        )

    result["feature_columns_analyzed"] = features
    result["n_rows"] = int(len(frame))

    return result


def _mi_vs_target(frame: pd.DataFrame, target: str) -> dict[str, float]:
    y_raw = frame[target]
    feature_cols = [c for c in frame.columns if c != target]
    if not feature_cols or y_raw.isna().all():
        return {}

    x = frame[feature_cols].copy()
    for col in x.columns:
        if not pd.api.types.is_numeric_dtype(x[col]):
            x[col] = LabelEncoder().fit_transform(x[col].astype(str).fillna("__NA__"))
        else:
            x[col] = x[col].fillna(x[col].median())

    mask = y_raw.notna()
    x = x.loc[mask]
    y = y_raw.loc[mask]
    if len(x) < 10:
        return {}

    try:
        if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 15:
            scores = mutual_info_regression(x, y, random_state=0)
        else:
            y_enc = LabelEncoder().fit_transform(y.astype(str))
            scores = mutual_info_classif(x, y_enc, random_state=0)
        ranked = sorted(
            ((str(c), float(s)) for c, s in zip(feature_cols, scores, strict=True)),
            key=lambda item: item[1],
            reverse=True,
        )
        return {c: s for c, s in ranked[:40]}
    except Exception:  # noqa: BLE001
        return {}


def _cramers_v(confusion: pd.DataFrame) -> float | None:
    if confusion.size == 0:
        return None
    table = confusion.to_numpy(dtype=float)
    total = table.sum()
    if total == 0:
        return None
    row_sum = table.sum(axis=1, keepdims=True)
    col_sum = table.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / total
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.nan_to_num((table - expected) ** 2 / expected)
    chi2 = float(terms.sum())
    r, k = confusion.shape
    denom = min(k - 1, r - 1)
    if denom <= 0:
        return None
    return float(np.sqrt(chi2 / (total * denom)))
