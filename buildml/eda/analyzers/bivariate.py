"""Measure how columns move together, using the right measure for each pair.

There is no single "correlation". Pearson measures straight-line relationships
between numbers. Spearman measures whether one rises as the other rises, curve
or not. Kendall measures agreement between pair orderings. Cramér's V measures
association between categoricals. Mutual information can detect nonlinear
dependence, but finite-sample estimates can miss relationships or reflect noise.

Using the wrong one is how relationships get missed. A symmetric U-shaped relationship can have
Pearson correlation near zero despite deterministic dependence; a near-zero
linear correlation should not be interpreted as independence.

Cost is why not everything is computed for everything. Pearson and Spearman are
cheap and run over all numeric pairs. Kendall runs only
on up to 20 of the strongest numeric pairs. Cramér's V runs on the first few low-cardinality
categoricals. Mutual information runs only against the target. These caps limit analysis cost
and should be considered when interpreting missing results.

See Also
--------
buildml.eda.analyzers.univariate : Each column on its own.
buildml.eda.analyzers.multivariate : Structure across many columns at once.
"""

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
    """Measure pairwise association, using each measure where it applies.

    Runs several association measures and returns them together to provide complementary screens;
    none guarantees detection of every relationship.
    Correlations over numeric pairs, Cramér's V over categorical pairs, and
    mutual information against the target when one is named.

    ``feature_columns`` is what keeps the rankings honest. Without it, an
    identifier column correlates with the row order, an ignored field shows up
    in the top pairs, and the target appears as its own best predictor. Passing
    the role-valid features excludes all of that; the target is reintroduced
    only for the mutual-information ranking, where it belongs.

    Parameters
    ----------
    frame:
        The data.
    target:
        The target column, if there is one. Excluded from feature-feature
        analysis and used for the mutual-information ranking.
    feature_columns:
        Which columns count as features. Defaults to everything except the
        target, which is rarely what you want: pass the role-valid set.

    Returns
    -------
    dict
        ``pearson`` and ``spearman``: full correlation matrices as nested
        dicts, ``None`` where undefined. ``top_abs_pearson_pairs``: the 30
        strongest, sorted by absolute value, which is where multicollinearity
        shows up. ``kendall_top_pairs``: up to 20, computed only on the
        strongest candidates. ``categorical_pairs``: Cramér's V for a few
        low-cardinality pairs. ``mutual_information_vs_target``: the top 40
        features by mutual information. ``feature_columns_analyzed`` and
        ``n_rows`` for provenance.

    Notes
    -----
    **Pearson near zero does not mean unrelated.** It means not *linearly*
    related. Compare it with Spearman and with the mutual information: a large
    gap can motivate checking for nonlinear monotone structure, outliers,
    or other distributional effects.

    **Mutual information is not a correlation.** It is non-negative, unbounded,
    and has no sign, so 0.4 is meaningful only relative to the other features in
    the same ranking. It can screen for nonlinear dependence, but its sensitivity depends on
    sample size, encoding, and estimator settings.

    **Mutual information here has no cross-validation and no significance
    test.** It is a screening tool for deciding what to look at, not evidence
    for a feature-selection decision: see :mod:`buildml.model.selection` for
    that.

    **Categoricals are label-encoded before mutual information**, which imposes
    an arbitrary order on unordered categories. This encoding can influence the estimate, especially when codes are
    treated as continuous values; confirm important rankings with suitable
    categorical methods and held-out evaluation.

    **The caps are real and silent.** Cramér's V covers at most the first eight
    categoricals with 40 or fewer levels; Kendall covers only the strongest
    numeric pairs. A relationship outside those caps simply will not appear.

    See Also
    --------
    buildml.model.selection : Feature selection with proper validation.
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
        from buildml.eda.analyzers.target import is_regression_target

        if is_regression_target(y):
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
