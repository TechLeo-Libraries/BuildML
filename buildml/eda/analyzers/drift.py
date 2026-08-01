"""Train/test drift analyzer for split-aware EDA."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, frame_for_partition


def analyze_drift(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    if split_plan is None:
        return {"available": False, "reason": "No split defined"}

    train = frame_for_partition(dataset, split_plan, "train")
    test = frame_for_partition(dataset, split_plan, "test")
    selected = [
        str(column)
        for column in (feature_columns if feature_columns is not None else train.columns)
        if column in train.columns and column in test.columns
    ]
    train = train[selected]
    test = test[selected]
    numeric_drift = []
    for col in train.select_dtypes(include="number").columns.astype(str):
        a = train[col].dropna()
        b = test[col].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        stat, p = ks_2samp(a, b)
        mean_shift = (
            float(b.mean() - a.mean())
            if pd.notna(a.mean()) and pd.notna(b.mean())
            else None
        )
        numeric_drift.append(
            {
                "column": col,
                "ks_stat": float(stat),
                "pvalue": float(p),
                "mean_shift": mean_shift,
                "flag": bool(p < 0.01 and abs(stat) > 0.1),
                "train_n": int(len(a)),
                "test_n": int(len(b)),
            }
        )
    numeric_drift.sort(key=lambda item: item["ks_stat"], reverse=True)

    categorical_drift = []
    for col in train.select_dtypes(exclude="number").columns.astype(str)[:30]:
        a = train[col].astype(str).value_counts(normalize=True)
        b = test[col].astype(str).value_counts(normalize=True)
        keys = sorted(set(a.index) | set(b.index))
        if not keys:
            continue
        pa = np.array([a.get(k, 0.0) for k in keys], dtype=float)
        pb = np.array([b.get(k, 0.0) for k in keys], dtype=float)
        # Jensen-Shannon divergence
        m = 0.5 * (pa + pb)
        js = float(
            0.5
            * (
                np.sum(pa * np.log2((pa + 1e-12) / (m + 1e-12)))
                + np.sum(pb * np.log2((pb + 1e-12) / (m + 1e-12)))
            )
        )
        categorical_drift.append(
            {
                "column": col,
                "js_divergence": js,
                "flag": bool(js > 0.1),
                "train_n": int(train[col].notna().sum()),
                "test_n": int(test[col].notna().sum()),
            }
        )
    categorical_drift.sort(key=lambda item: item["js_divergence"], reverse=True)

    flags = [r for r in numeric_drift if r["flag"]] + [r for r in categorical_drift if r["flag"]]
    return {
        "available": True,
        "numeric_drift": numeric_drift[:40],
        "categorical_drift": categorical_drift[:40],
        "flagged_columns": flags,
        "flagged_count": len(flags),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_columns_analyzed": selected,
        "settings": {
            "numeric_test": "two-sample Kolmogorov-Smirnov",
            "numeric_flag": "pvalue < 0.01 and KS statistic > 0.1",
            "categorical_metric": "Jensen-Shannon divergence (base 2)",
            "categorical_flag": "JS divergence > 0.1",
        },
        "summary": (
            f"{len(flags)} columns flagged for train/test distribution shift"
            if flags
            else "No strong train/test drift flags under current thresholds"
        ),
    }
