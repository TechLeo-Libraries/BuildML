"""Target-aware analyzer."""

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
