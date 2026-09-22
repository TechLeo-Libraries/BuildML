"""Find unusual values, per column and in combination, without judging them.

"Outlier" is not a property of a data point. It is a statement about a
distribution, and whether a given point is an error, a rare event, or the whole
reason the project exists depends entirely on context. A transaction ten times
the median may be a valid large purchase, a recording error, or an event worth
investigating for fraud.

So this reports and does not act. Three methods provide complementary screens. The IQR rule uses quartiles and is resistant to isolated extremes, though
contamination can still shift its boundaries. Z-scores can be computed for
non-normal data, but normal-tail interpretations require additional assumptions. Extreme values can
also influence the mean and standard deviation used by z-scores.
Isolation Forest looks across numeric features and can flag unusual
combinations, but does not encode domain consistency rules such as a 19-year-old with 30
years of driving experience.

See Also
--------
buildml.eda.analyzers.univariate : The distributions these are unusual against.
buildml.preprocess.outliers : Acting on what is found here.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import IsolationForest


def analyze_outliers(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Count unusual values per column, and unusual rows across columns.

    Per column, two counts. The IQR rule flags values more than 1.5 interquartile
    ranges beyond the quartiles: the same rule that draws the whiskers on a box
    plot. Quartiles resist isolated extremes but can shift when enough values
    change or when the sample is small.
    Z-scores flag values more than three standard deviations from the mean,
    which is less robust for exactly the opposite reason: a single extreme value
    inflates the standard deviation and can hide itself.

    Across columns, Isolation Forest scores whole rows. It can detect
    multivariate patterns that per-column screens miss, without guaranteeing
    detection of every implausible combination.

    Restricting to feature columns matters here. Numeric distances between identifier values may not
    have a meaningful interpretation; a target's extremes may be
    the cases you most want to predict, not errors to remove.

    Parameters
    ----------
    frame:
        The data.
    feature_columns:
        Which columns to screen. Defaults to all, which will happily report
        outliers in your row IDs.

    Returns
    -------
    dict
        ``per_column``: for each numeric column, the IQR count, rate, and
        bounds, plus the count and rate beyond three standard deviations.
        ``multivariate``: the Isolation Forest result, or empty when there was
        not enough data. ``feature_columns_analyzed`` for provenance.

    Notes
    -----
    **Nothing here says a point is wrong.** For a skewed distribution: income,
    duration, transaction size: the IQR rule flags a large fraction of the
    upper tail as a matter of arithmetic, not because anything is amiss. Read
    the flags together with the skew from the univariate analysis.

    **Compare the two per-column counts.** The rules use different
    centers, scales, and thresholds, so their counts can disagree for several
    reasons. Inspect the histogram, missingness, tail shape, and domain context;
    counts alone do not identify skewness or explain the source of extremes.

    **Isolation Forest needs complete rows.** Any row with a missing value in
    any screened column is excluded, so a frame with scattered gaps can leave
    very few rows scored. Check ``n_rows_scored``.

    **``contamination='auto'`` is a threshold, not a measurement.** The
    ``anomaly_rate`` reflects that setting as much as the data. Use it to find
    rows worth inspecting, not as an estimate of how much of your data is bad.

    **Rows are sampled above 20,000** for the multivariate screen.

    See Also
    --------
    buildml.preprocess.outliers : Winsorising or removing what is found.
    """
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
