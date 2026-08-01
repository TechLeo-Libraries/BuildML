"""Outlier analyzer: IQR, z-score, isolation-forest screen."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import IsolationForest


def analyze_outliers(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Screen numeric feature values without scoring targets or identifiers."""
    selected = [
        str(column)
        for column in (feature_columns if feature_columns is not None else frame.columns)
        if column in frame.columns
    ]
    numeric = frame[selected].select_dtypes(include="number")
    per_column: dict[str, Any] = {}
    for col in numeric.columns.astype(str):
        s = numeric[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (numeric[col] < lower) | (numeric[col] > upper)
        z = (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) else s * 0
        per_column[col] = {
            "iqr_outlier_count": int(mask.sum()),
            "iqr_outlier_rate": float(mask.mean()),
            "iqr_bounds": [float(lower), float(upper)],
            "zscore_abs_gt_3": int((z.abs() > 3).sum()),
            "zscore_abs_gt_3_rate": float((z.abs() > 3).mean()) if len(z) else 0.0,
        }

    multivariate = {}
    clean = numeric.dropna()
    if clean.shape[1] >= 2 and len(clean) >= 30:
        sample = clean if len(clean) <= 20000 else clean.sample(20000, random_state=0)
        model = IsolationForest(random_state=0, contamination="auto")
        pred = model.fit_predict(sample)
        multivariate = {
            "method": "isolation_forest",
            "n_rows_scored": int(len(sample)),
            "anomaly_count": int((pred == -1).sum()),
            "anomaly_rate": float((pred == -1).mean()),
        }

    return {
        "per_column": per_column,
        "multivariate": multivariate,
        "feature_columns_analyzed": selected,
    }
