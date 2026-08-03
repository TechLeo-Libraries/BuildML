"""Check whether train and test actually came from the same distribution.

The premise behind every holdout evaluation is that the test partition looks
like the training partition, drawn from the same process. When that fails, the
test score stops estimating future performance and starts measuring something
else entirely.

It fails more often than people expect. A time-based split where behaviour
changed mid-period. A random split that happened, by chance, to put most of a
rare category on one side. A concatenation of two data sources where the second
was collected differently. Each produces a test score that is wrong in a
direction you cannot predict.

This is the one analyzer that reads the split, and it is worth running before
you trust any evaluation number.

See Also
--------
buildml.data.splits : Where partitions are defined.
buildml.model.evidence : Uncertainty on the metrics this affects.
"""

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
    """Compare each column's train and test distributions, and flag the gaps.

    Two tests, one per kind of column. Numeric columns get a two-sample
    Kolmogorov-Smirnov test, which compares the whole shape of the distribution
    rather than just the mean: a column with the same average and twice the
    variance is caught. Categorical columns get Jensen-Shannon divergence
    between the category frequencies, a bounded symmetric measure where 0 means
    identical and 1 means no overlap.

    The numeric flag requires both statistical significance (p below 0.01) and
    practical size (KS statistic above 0.1), and requiring both is the point.
    With 100,000 rows, a p-value alone flags differences far too small to
    matter; with 200 rows, a large difference may not reach significance. Either
    criterion alone produces a report nobody can act on.

    Parameters
    ----------
    dataset:
        The data.
    split_plan:
        Partition membership. ``None`` returns an unavailable result rather than
        raising, since drift analysis is optional in an EDA pass.
    feature_columns:
        Which columns to check. Defaults to everything present in both
        partitions. Restrict this to the role-valid features, or you will get
        drift reports on identifier columns, which always drift and never
        matter.

    Returns
    -------
    dict
        ``available`` is ``False`` with a ``reason`` when there is no split.
        Otherwise: ``numeric_drift`` and ``categorical_drift``: up to 40 each,
        sorted worst first. ``flagged_columns`` and ``flagged_count``: those
        that crossed both thresholds. ``train_rows``, ``test_rows``,
        ``feature_columns_analyzed`` for provenance. ``settings``: the tests
        and thresholds in words, so a report is readable without this docstring.
        ``summary``: a sentence.

    Notes
    -----
    **Drift in a feature is a warning, not a verdict.** Some is expected from
    the randomness of splitting. What matters is drift in a feature the model
    relies on heavily, or drift in many features at once, which suggests the two
    partitions are not the same population.

    **The absence of flags is not proof of no drift.** Columns with fewer than
    five non-missing values on either side are skipped, only the first 30
    categoricals are checked, and both lists are truncated at 40.

    **Detecting drift does not tell you what to do.** A time-based split showing
    drift may be correctly simulating deployment, in which case the honest
    response is to accept a lower score, not to reshuffle until the drift
    disappears: that would just hide the problem you will meet in production.

    Examples
    --------
    ::

        report = analyze_drift(dataset, split_plan, feature_columns=features)
        if report["available"] and report["flagged_count"]:
            for entry in report["flagged_columns"]:
                print(entry["column"], entry.get("ks_stat") or entry["js_divergence"])
    """
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
