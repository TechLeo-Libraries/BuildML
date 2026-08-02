"""Train-fitted dimensionality reduction (PCA) with explained-variance reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from buildml.core.errors import MissingExtraError, ValidationError
from buildml.core.types import ColumnRole
from buildml.core.validation import validate_column_names
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.explain.schemas import (
    Action,
    ActionPriority,
    Evidence,
    EvidenceKind,
    Finding,
    FindingSeverity,
    Recommendation,
)
from buildml.ingest.detect import schema_from_dataframe
from buildml.preprocess.result import PreprocessResult

ReduceMethod = Literal["pca", "umap", "tsne"]


@dataclass(slots=True)
class ReducePlan:
    """Train-fitted dimensionality-reduction plan."""

    columns: tuple[str, ...]
    method: ReduceMethod
    n_components: int
    feature_names_: tuple[str, ...]
    explained_variance_ratio_: tuple[float, ...]
    cumulative_explained_variance_: tuple[float, ...]
    reducer_: Any = field(repr=False)
    drop_input_columns: bool = True
    prefix: str = "pc"
    disclosures: tuple[str, ...] = ()
    train_embedding_: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "method": self.method,
            "n_components": self.n_components,
            "feature_names_": list(self.feature_names_),
            "explained_variance_ratio_": list(self.explained_variance_ratio_),
            "cumulative_explained_variance_": list(self.cumulative_explained_variance_),
            "drop_input_columns": self.drop_input_columns,
            "prefix": self.prefix,
            "disclosures": list(self.disclosures),
            "total_explained_variance": (
                float(self.cumulative_explained_variance_[-1])
                if self.cumulative_explained_variance_
                else 0.0
            ),
        }


def fit_reducer(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    columns: list[str] | None = None,
    method: ReduceMethod = "pca",
    n_components: int | float | None = None,
    drop_input_columns: bool = True,
    prefix: str = "pc",
    random_state: int | None = 0,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    tsne_perplexity: float = 30.0,
    tsne_learning_rate: str | float = "auto",
) -> ReducePlan:
    """Fit a dimensionality reducer on the train partition only."""
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    if method not in {"pca", "umap", "tsne"}:
        raise ValidationError(f"Unsupported reduce method '{method}'")
    if not prefix or not str(prefix).replace("_", "").isalnum():
        raise ValidationError("prefix must be a non-empty alphanumeric token")

    train = frame_for_partition(dataset, split_plan, "train")
    cols = _resolve_numeric_columns(dataset, train, columns)
    x_arr = train[list(cols)].to_numpy(dtype=float)
    if np.isnan(x_arr).any():
        raise ValidationError(
            "Dimensionality reduction requires non-null train features. "
            "Call session.impute(...) first."
        )
    n_samples, n_features = x_arr.shape
    max_components = min(n_samples, n_features)
    if max_components < 1:
        raise ValidationError("Not enough train rows/columns for reduction")

    disclosures: list[str] = []

    if method == "pca":
        return _fit_pca(
            cols,
            x_arr,
            n_components=n_components,
            max_components=max_components,
            drop_input_columns=drop_input_columns,
            prefix=prefix,
        )

    if method == "umap":
        from buildml.unsupervised.extras import umap_available, require_umap

        if not umap_available():
            raise MissingExtraError(
                "unsupervised",
                "UMAP reduction (pip install 'buildml[unsupervised]')",
            )
        umap = require_umap()
        n_out = _resolve_n_components_int(n_components, max_components, default=2)
        reducer = umap.UMAP(
            n_components=n_out,
            n_neighbors=int(umap_n_neighbors),
            min_dist=float(umap_min_dist),
            random_state=random_state,
        )
        reducer.fit(x_arr)
        names = tuple(f"{prefix}_{i + 1}" for i in range(n_out))
        disclosures.append(
            "UMAP (umap-learn) used as industry default when buildml[unsupervised] installed."
        )
        return ReducePlan(
            columns=tuple(cols),
            method="umap",
            n_components=n_out,
            feature_names_=names,
            explained_variance_ratio_=(),
            cumulative_explained_variance_=(),
            reducer_=reducer,
            drop_input_columns=drop_input_columns,
            prefix=prefix,
            disclosures=tuple(disclosures),
        )

    # t-SNE — transductive on train; holdout via nearest-neighbor embedding transfer
    n_out = _resolve_n_components_int(n_components, max_components, default=2)
    perplexity = min(float(tsne_perplexity), max(5.0, (n_samples - 1) / 3.0))
    tsne = TSNE(
        n_components=n_out,
        perplexity=perplexity,
        learning_rate=tsne_learning_rate,
        random_state=random_state,
        init="pca",
    )
    embedding = tsne.fit_transform(x_arr)
    names = tuple(f"{prefix}_{i + 1}" for i in range(n_out))
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(x_arr)
    disclosures.extend(
        [
            "t-SNE is transductive: embedding is computed on train only.",
            "Holdout/full-frame transform uses nearest train neighbor embedding "
            "(disclosed approximation — not a native t-SNE out-of-sample map).",
        ]
    )
    return ReducePlan(
        columns=tuple(cols),
        method="tsne",
        n_components=n_out,
        feature_names_=names,
        explained_variance_ratio_=(),
        cumulative_explained_variance_=(),
        reducer_=nn,
        drop_input_columns=drop_input_columns,
        prefix=prefix,
        disclosures=tuple(disclosures),
        train_embedding_=np.asarray(embedding, dtype=float),
    )


def _resolve_n_components_int(
    n_components: int | float | None,
    max_components: int,
    *,
    default: int,
) -> int:
    if n_components is None:
        return min(default, max_components)
    if isinstance(n_components, float):
        if not (0.0 < n_components <= 1.0):
            raise ValidationError("Float n_components must be in (0, 1] for PCA variance target")
        return max(1, int(max_components * n_components))
    if int(n_components) < 1:
        raise ValidationError("Integer n_components must be >= 1")
    return min(int(n_components), max_components)


def _fit_pca(
    cols: list[str],
    x_arr: np.ndarray,
    *,
    n_components: int | float | None,
    max_components: int,
    drop_input_columns: bool,
    prefix: str,
) -> ReducePlan:
    pca_n: int | float
    if n_components is None:
        pca_n = max_components
    elif isinstance(n_components, float):
        if not (0.0 < n_components <= 1.0):
            raise ValidationError("Float n_components must be in (0, 1] (variance target)")
        pca_n = n_components
    else:
        if int(n_components) < 1:
            raise ValidationError("Integer n_components must be >= 1")
        pca_n = min(int(n_components), max_components)

    reducer = PCA(n_components=pca_n, svd_solver="full")
    reducer.fit(x_arr)
    ratios = tuple(float(v) for v in np.asarray(reducer.explained_variance_ratio_, dtype=float))
    cumulative = tuple(float(v) for v in np.cumsum(ratios))
    n_out = len(ratios)
    names = tuple(f"{prefix}_{i + 1}" for i in range(n_out))
    return ReducePlan(
        columns=tuple(cols),
        method="pca",
        n_components=n_out,
        feature_names_=names,
        explained_variance_ratio_=ratios,
        cumulative_explained_variance_=cumulative,
        reducer_=reducer,
        drop_input_columns=drop_input_columns,
        prefix=prefix,
    )


def transform_reducer(
    dataset: Dataset,
    plan: ReducePlan,
) -> tuple[Dataset, PreprocessResult]:
    """Apply a train-fitted reduction plan to the full dataset."""
    missing = [c for c in plan.columns if c not in dataset.columns]
    if missing:
        raise ValidationError(f"Reduce plan columns missing from dataset: {missing}")

    frame = dataset._ensure_pandas().copy()
    values = frame[list(plan.columns)].to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValidationError(
            "Dimensionality reduction transform found nulls. Impute before reduce_dimensions."
        )
    if plan.method == "pca":
        transformed = plan.reducer_.transform(values)
    elif plan.method == "umap":
        transformed = plan.reducer_.transform(values)
    elif plan.method == "tsne":
        if plan.train_embedding_ is None:
            raise ValidationError("t-SNE plan missing train embedding for holdout transform")
        nn_model = plan.reducer_
        if not isinstance(nn_model, NearestNeighbors):
            raise ValidationError("t-SNE plan reducer must be NearestNeighbors for transform")
        _, indices = nn_model.kneighbors(values)
        transformed = plan.train_embedding_[indices[:, 0]]
    else:
        raise ValidationError(f"Unsupported reduce method '{plan.method}'")
    component_frame = pd.DataFrame(
        transformed,
        columns=list(plan.feature_names_),
        index=frame.index,
    )
    roles = dict(dataset.roles)
    for column in plan.columns:
        roles.pop(column, None)
    for name in plan.feature_names_:
        roles[name] = ColumnRole.FEATURE

    if plan.drop_input_columns:
        frame = frame.drop(columns=list(plan.columns))
    out = pd.concat([frame, component_frame], axis=1)
    new_dataset = Dataset.from_transformed(
        dataset,
        out,
        schema=schema_from_dataframe(out),
        roles=roles,
    )
    return new_dataset, _build_result(plan)


def _resolve_numeric_columns(
    dataset: Dataset,
    train: pd.DataFrame,
    columns: list[str] | None,
) -> list[str]:
    protected = {
        ColumnRole.TARGET,
        ColumnRole.ID,
        ColumnRole.GROUP,
        ColumnRole.TIME,
        ColumnRole.WEIGHT,
    }
    if columns is not None:
        names = validate_column_names(columns, dataset.columns)
        names = [name for name in names if dataset.roles.get(name) not in protected]
    else:
        feature_roles = dataset.role_columns(ColumnRole.FEATURE)
        candidates = feature_roles or [
            str(c) for c in train.columns if dataset.roles.get(str(c)) not in protected
        ]
        names = [
            str(c)
            for c in candidates
            if c in train.columns and pd.api.types.is_numeric_dtype(train[c])
        ]
    if not names:
        raise ValidationError("No numeric columns available for dimensionality reduction")
    non_numeric = [c for c in names if not pd.api.types.is_numeric_dtype(train[c])]
    if non_numeric:
        raise ValidationError(
            "Dimensionality reduction requires numeric columns; "
            f"encode/scale first. Non-numeric: {non_numeric[:12]}"
        )
    return names


def _build_result(plan: ReducePlan) -> PreprocessResult:
    total = (
        float(plan.cumulative_explained_variance_[-1])
        if plan.cumulative_explained_variance_
        else 0.0
    )
    method_label = plan.method.upper()
    evidence = [
        Evidence(
            key="reduce_dimensions.explained_variance",
            kind=EvidenceKind.METRIC,
            summary=f"Train-fitted {method_label} reduction.",
            value={
                "method": plan.method,
                "n_components": plan.n_components,
                "explained_variance_ratio": list(plan.explained_variance_ratio_),
                "cumulative_explained_variance": list(plan.cumulative_explained_variance_),
                "total_explained_variance": total,
                "source_columns": list(plan.columns),
                "disclosures": list(plan.disclosures),
            },
            source=f"train.{plan.method}",
            limitations=(
                "Reduction quality is unsupervised; it is not predictive utility.",
                *plan.disclosures,
            ),
        )
    ]
    detail_suffix = (
        f" capturing {total:.1%} of train variance among those columns."
        if plan.method == "pca"
        else f" via {method_label} embedding."
    )
    findings = [
        Finding(
            key="reduce_dimensions.applied",
            title=f"{method_label} components fitted on train",
            detail=(
                f"Replaced {len(plan.columns)} numeric column(s) with "
                f"{plan.n_components} component(s){detail_suffix}"
            ),
            severity=FindingSeverity.INFO,
            evidence=tuple(evidence),
            affected_columns=plan.columns,
        )
    ]
    limitations = [
        f"{method_label} is fit on train only; holdout rows use the frozen transform.",
        "Embedding quality does not guarantee better predictive metrics.",
    ]
    if plan.method == "pca":
        limitations.append("Components are linear mixes; interpret loadings before domain claims.")
    if plan.method == "tsne":
        limitations.extend(list(plan.disclosures))
    methods = [
        f"{method_label} fitted on train numeric columns.",
        f"Output columns: {', '.join(plan.feature_names_[:8])}"
        + ("…" if len(plan.feature_names_) > 8 else ""),
    ]
    interpretation = [
        f"{method_label} kept {plan.n_components} component(s) from {len(plan.columns)} column(s).",
    ]
    if plan.method == "pca":
        interpretation.append(f"Cumulative train variance explained: {total:.1%}.")
    recommendations = [
        Recommendation(
            key="reduce_dimensions.scale-first",
            title="Confirm scaling before interpreting PCA variance shares",
            rationale=(
                "Unscaled columns with large magnitudes dominate components. "
                "Compare holdout metrics with and without reduction before keeping it."
            ),
            priority=ActionPriority.NEXT,
            action=Action(
                key="reduce_dimensions.eval-action",
                label="Session.evaluate(partition='validation')",
                operation="evaluate",
                parameters={"partition": "validation"},
            ),
            based_on=("reduce_dimensions.applied",),
            caveats=("Variance explained is not a substitute for supervised selection.",),
        )
    ]
    return PreprocessResult(
        operation="reduce_dimensions",
        plan=plan.to_dict(),
        evidence=evidence,
        findings=findings,
        interpretation=interpretation,
        limitations=limitations,
        recommendations=recommendations,
        methods=methods,
        warnings=list(plan.disclosures),
    )
