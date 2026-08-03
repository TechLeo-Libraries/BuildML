"""Find unusual values, per column and in combination, without judging them.

"Outlier" is not a property of a data point. It is a statement about a
distribution, and whether a given point is an error, a rare event, or the whole
reason the project exists depends entirely on context. A transaction ten times
the median is an outlier in a spending model and the entire target in a fraud
model.

So this reports and does not act. Three methods, because each sees something the
others miss. The IQR rule is distribution-free and robust: it uses quartiles, so
extreme values cannot drag the boundaries out to include themselves. Z-scores
assume roughly normal data and are pulled around by the very points they are
meant to find, which is why they are reported alongside rather than alone.
Isolation Forest looks at rows rather than values, and catches the combination
that is strange while every individual field is ordinary: a 19-year-old with 30
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
    plot, and robust because quartiles are not moved by extreme values.
    Z-scores flag values more than three standard deviations from the mean,
    which is less robust for exactly the opposite reason: a single extreme value
    inflates the standard deviation and can hide itself.

    Across columns, Isolation Forest scores whole rows. It finds combinations
    that are individually plausible and jointly strange, which no per-column
    method can see.

    Restricting to feature columns matters here. An identifier is uniformly
    distributed and will produce nonsense bounds; a target's extremes are often
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

    **Compare the two per-column counts.** When the IQR count is much larger
    than the z-score count, the distribution is skewed. When the z-score count
    is larger, there are extremes so severe they have inflated the standard
    deviation, and both numbers understate the problem.

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
