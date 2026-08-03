"""Find the redundancy that only shows up when you look at columns together.

Pairwise correlation misses a whole class of problem. Three columns can each be
weakly correlated with the others and still be perfectly redundant — if
``total = subtotal + tax``, no single pair looks alarming, and the three
together are linearly dependent. Coefficients in a linear model on such data are
arbitrary, unstable, and uninterpretable.

Three views of that redundancy. Correlation clusters group columns that are
transitively linked above a threshold, which is the quick answer to "which of
these are measuring the same thing". VIF quantifies how well each column is
predicted by *all* the others, catching the multi-column case that pairs miss.
PCA describes how much genuinely independent variation exists — if five
components explain 99% of twenty columns, there are effectively five things
being measured.

See Also
--------
buildml.eda.analyzers.bivariate : Pairwise association.
buildml.preprocess.reduce : Acting on this with dimensionality reduction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def analyze_multivariate(
    frame: pd.DataFrame,
    bivariate: dict[str, Any],
    *,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Group redundant columns, score their collinearity, and size the signal.

    Three complementary readings of the same structure.

    *Correlation clusters* are built by union-find over pairs correlated above
    0.7 in absolute value, so the grouping is transitive: A with B and B with C
    puts all three together even when A and C are not directly correlated. Each
    cluster is a set of columns measuring roughly one thing, and usually a
    candidate for keeping one.

    *VIF* — variance inflation factor — regresses each column on all the others
    and reports ``1 / (1 - R²)``. Above 5 is worth a look, above 10 is severe.
    This is the measure that catches the ``total = subtotal + tax`` case, where
    no pair is alarming.

    *PCA* on standardised columns says how many independent directions there
    really are. When the first three components explain 95% of twenty columns,
    the data is three-dimensional wearing a twenty-column costume.

    Parameters
    ----------
    frame:
        The data.
    bivariate:
        Output from :func:`~buildml.eda.analyzers.bivariate.analyze_bivariate`.
        Its ``top_abs_pearson_pairs`` are reused for clustering rather than
        recomputing the correlation matrix.
    feature_columns:
        Which columns to consider. Defaults to all, which will include
        identifiers and targets in the collinearity analysis.

    Returns
    -------
    dict
        ``correlation_clusters`` — groups of two or more linked columns.
        ``vif`` — per column, worst first, with the underlying
        ``r2_other_features``. ``pca`` — explained variance ratios, the
        cumulative curve, and the top loadings per component; ``None`` when
        there was not enough data. ``numeric_column_count``,
        ``complete_case_rows``, and ``feature_columns_analyzed`` for provenance.

    Notes
    -----
    **Rows with any missing value are dropped, and this can be brutal.**
    Complete-case analysis on twenty columns each 5% missing can leave a third
    of the rows. Compare ``complete_case_rows`` against the frame length before
    reading anything else here; a large gap means these results describe an
    unrepresentative subset.

    **Collinearity hurts interpretation more than prediction.** A random forest
    barely notices; a linear model's coefficients become meaningless. If you are
    reading coefficients, this section matters. If you are only predicting, it
    matters much less.

    **Only the first 20 numeric columns get a VIF**, and it needs at least five
    more rows than columns.

    **Clusters are not a recommendation to drop.** Two correlated columns may
    still be worth keeping if they differ in the tails, or if one is available
    at score time and the other is not.

    See Also
    --------
    buildml.eda.analyzers.bivariate.analyze_bivariate : Produces the input.
    """
    selected = [
        str(column)
        for column in (feature_columns if feature_columns is not None else frame.columns)
        if column in frame.columns
    ]
    numeric_source = frame[selected].select_dtypes(include="number")
    numeric = numeric_source.dropna()
    clusters: list[list[str]] = []
    if numeric.shape[1] >= 3 and bivariate.get("top_abs_pearson_pairs"):
        threshold = 0.7
        parent = {str(c): str(c) for c in numeric.columns}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pb] = pa

        for pair in bivariate["top_abs_pearson_pairs"]:
            if abs(pair["corr"]) >= threshold and pair["a"] in parent and pair["b"] in parent:
                union(pair["a"], pair["b"])
        groups: dict[str, list[str]] = {}
        for col in parent:
            groups.setdefault(find(col), []).append(col)
        clusters = [sorted(v) for v in groups.values() if len(v) >= 2]

    vif = _vif_screen(numeric)
    pca_note = None
    if numeric.shape[1] >= 3 and len(numeric) >= 15:
        scaled = StandardScaler().fit_transform(numeric)
        n_comp = min(5, scaled.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(scaled)
        pca_note = {
            "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
            "cumulative_explained_variance": [
                float(x) for x in np.cumsum(pca.explained_variance_ratio_)
            ],
            "n_components": int(n_comp),
            "components_top_loadings": _top_loadings(pca, list(numeric.columns.astype(str))),
        }

    return {
        "correlation_clusters": clusters,
        "vif": vif,
        "pca": pca_note,
        "numeric_column_count": int(numeric_source.shape[1]),
        "complete_case_rows": int(len(numeric)),
        "feature_columns_analyzed": selected,
    }


def _vif_screen(numeric: pd.DataFrame, max_cols: int = 20) -> list[dict[str, Any]]:
    cols = list(numeric.columns.astype(str))[:max_cols]
    if len(cols) < 2 or len(numeric) < len(cols) + 5:
        return []
    x = numeric[cols].to_numpy(dtype=float)
    # Add intercept
    x = np.column_stack([np.ones(len(x)), x])
    rows = []
    for i, col in enumerate(cols, start=1):
        y = x[:, i]
        preds = x[:, [j for j in range(x.shape[1]) if j != i]]
        try:
            coef, _, _, _ = np.linalg.lstsq(preds, y, rcond=None)
            y_hat = preds @ coef
            ssr = float(np.sum((y - y_hat) ** 2))
            sst = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ssr / sst if sst > 0 else 0.0
            vif = float(1.0 / max(1.0 - r2, 1e-12))
            rows.append({"column": col, "vif": vif, "r2_other_features": r2})
        except Exception:  # noqa: BLE001
            continue
    rows.sort(key=lambda item: item["vif"], reverse=True)
    return rows


def _top_loadings(
    pca: PCA,
    columns: list[str],
    top_k: int = 5,
) -> dict[str, list[dict[str, float]]]:
    out: dict[str, list[dict[str, float]]] = {}
    for i, component in enumerate(pca.components_):
        pairs = sorted(
            (
                {"column": columns[j], "loading": float(component[j])}
                for j in range(len(columns))
            ),
            key=lambda item: abs(item["loading"]),
            reverse=True,
        )
        out[f"pc{i + 1}"] = pairs[:top_k]
    return out
