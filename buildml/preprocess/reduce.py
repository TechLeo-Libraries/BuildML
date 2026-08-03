"""Compress many correlated columns into a few, learning the compression from train.

When features are highly correlated they are, in a real sense, measuring the
same thing several times. Thirty sensor readings from the same machine, or
hundreds of TF-IDF columns from the same corpus, occupy far fewer genuinely
independent directions than their count suggests. Reduction finds those
directions and rewrites each row in terms of them.

The payoff is fewer columns for the model to overfit against, no
multicollinearity to destabilise a linear model's coefficients, and: at two or
three components: a plot you can actually look at. The price is
interpretability: a component is a weighted blend of your original columns, so
"component 1 increased" is not a sentence anyone outside the analysis can act
on.

Three methods, and they are not interchangeable.

**PCA** finds the directions of greatest variance using a linear transform. It
is the only one of the three that produces a genuine reusable mapping: new data
goes through the same matrix multiplication, which is what makes it the only
sound choice for a production pipeline. It also reports how much variance each
component retains, so you can see what you gave up.

**UMAP** captures curved structure that PCA misses and is excellent for
visualisation and clustering. It needs ``buildml[unsupervised]``.

**t-SNE** produces the clearest visual separation of clusters but is
*transductive*: it embeds the points it was fitted on and has no formula for a
new point. Holdout rows here are placed at their nearest training neighbour's
position, which is an approximation, not a transform. Use t-SNE to look at your
data, not to feed a model.

For all three, scale first: reduction operates on variance, and an unscaled
column with large units will dominate the first component purely through its
units.
"""

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
from buildml.preprocess.columns import resolve_transform_columns
from buildml.preprocess.result import PreprocessResult

ReduceMethod = Literal["pca", "umap", "tsne"]


@dataclass(slots=True)
class ReducePlan:
    """The compression learned from training rows, and what it cost.

    Attributes
    ----------
    columns:
        The source numeric columns that were compressed.
    method:
        ``'pca'``, ``'umap'``, or ``'tsne'``.
    n_components:
        How many output columns the compression produces.
    feature_names_:
        The output column names, ``<prefix>_1`` upward.
    explained_variance_ratio_:
        For PCA, the share of total variance each component retains. The first
        component always captures the most; watch for where the values flatten
        out, since components past that point are mostly noise. Empty for UMAP
        and t-SNE, which have no equivalent measure.
    cumulative_explained_variance_:
        The running total of the above. The last value is the headline number:
        how much of the original variation survived. Below about 0.8 you have
        thrown away a lot.
    reducer_:
        The fitted object. A PCA or UMAP model for those methods; for t-SNE it
        is a nearest-neighbour index, since t-SNE itself cannot transform new
        points.
    drop_input_columns:
        Whether the source columns were removed after reduction.
    prefix:
        The naming stem for the output columns.
    disclosures:
        Method-specific caveats recorded at fit time and surfaced in the
        result: most importantly the warning that t-SNE holdout positions are
        approximated by nearest neighbour rather than computed.
    train_embedding_:
        For t-SNE only, the training-row coordinates. This is what holdout
        rows are matched against, so it must be kept.
    """

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
        """Return the plan's settings and variance accounting as JSON-safe values.

        The fitted reducer and the stored t-SNE embedding are omitted, since
        neither serialises to JSON: save a pipeline to round-trip those.

        Returns
        -------
        dict
            The settings, both variance sequences, the recorded disclosures,
            and a convenience ``total_explained_variance`` holding the final
            cumulative value (0.0 when the method reports none).
        """
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
    """Learn a compression of the numeric columns from the training rows.

    Nothing is transformed here: pass the plan to :func:`transform_reducer` to
    apply it.

    Parameters
    ----------
    dataset:
        The full dataset. Only rows the split assigns to ``train`` are read.
    split_plan:
        The split defining the training rows. Required, because components
        derived from all rows are oriented partly by the test data, and every
        subsequent score is compromised.
    columns:
        Which numeric columns to compress. Defaults to the numeric ``feature``
        columns. Naming a subset is common: you might reduce a block of
        correlated sensor readings while leaving your interpretable business
        features intact.
    method:
        ``'pca'`` (the default) for anything feeding a model; ``'umap'`` or
        ``'tsne'`` for visualisation. See the module docstring before choosing
        either of the latter for a pipeline.
    n_components:
        How many dimensions to keep. An integer is a literal count. A float
        between 0 and 1 asks PCA to keep however many components are needed to
        retain that share of variance, which is usually the better way to
        express the intent: ``0.95`` says "lose no more than 5% of the
        variation" and lets the data decide the count. ``None`` picks a
        sensible default: full rank for PCA, two dimensions for UMAP and
        t-SNE.
    drop_input_columns:
        Remove the source columns after reduction. Usually correct, since
        keeping both the originals and their compression reintroduces the
        collinearity you were removing.
    prefix:
        Naming stem for the output columns, giving ``pc_1``, ``pc_2``, and so
        on. Change it when reducing several column blocks separately so the
        names stay distinguishable.
    random_state:
        Seed for the stochastic parts of PCA's solver and for UMAP and t-SNE,
        both of which are genuinely random and will produce visibly different
        embeddings run to run without it.
    umap_n_neighbors:
        For UMAP, how many neighbours define the local neighbourhood. Small
        values preserve fine local structure; large values preserve the broad
        shape. The default of 15 is a middle setting.
    umap_min_dist:
        For UMAP, how tightly points may pack together in the output. Low
        values produce dense, clearly separated clumps; higher values spread
        points out and show relative density better.
    tsne_perplexity:
        For t-SNE, roughly how many neighbours each point balances against.
        Automatically reduced when the training set is too small to support the
        value you asked for. Results change substantially with this setting, so
        try a few before drawing conclusions from a t-SNE plot.
    tsne_learning_rate:
        For t-SNE, the optimisation step size. ``'auto'`` scales it to the
        dataset and is almost always the right choice.

    Returns
    -------
    ReducePlan
        The fitted compression, its variance accounting, and any method
        disclosures.

    Raises
    ------
    ~buildml.core.errors.LeakageError
        No split plan was supplied.
    ~buildml.core.errors.ValidationError
        ``method`` is unrecognised, ``prefix`` is not alphanumeric, training
        features contain missing values, or there are too few training rows or
        columns to reduce.
    ~buildml.core.errors.MissingExtraError
        UMAP was requested without ``buildml[unsupervised]`` installed.

    Notes
    -----
    **Scale first.** Every method here works on variance, so an unscaled column
    measured in thousands will dominate the leading component through its units
    alone. Run :func:`~buildml.preprocess.scale.fit_scaler` before this.

    **Impute first.** Reduction cannot proceed with gaps and will raise rather
    than guess.

    **Read the variance you kept.** ``cumulative_explained_variance_[-1]`` is
    the honest summary of what survived. Compressing forty columns into three
    that retain 60% of the variance has discarded a great deal, and the model
    will feel it.

    Examples
    --------
    >>> plan = fit_reducer(  # doctest: +SKIP
    ...     dataset, split_plan, method="pca", n_components=0.95
    ... )
    >>> plan.n_components  # doctest: +SKIP
    12

    See Also
    --------
    transform_reducer : Applies the plan produced here.
    """
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
        from buildml.unsupervised.extras import require_umap, umap_available

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

    # t-SNE: transductive on train; holdout via nearest-neighbor embedding transfer
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
            "(disclosed approximation: not a native t-SNE out-of-sample map).",
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
    """Project every row into the reduced space using an already-fitted plan.

    The source columns are replaced by the component columns. Because the
    mapping was learned from training rows, every partition lands in the same
    coordinate system without the test rows having helped define it.

    Parameters
    ----------
    dataset:
        The dataset to compress. Every column the plan names must be present.
    plan:
        A plan from :func:`fit_reducer`.

    Returns
    -------
    tuple of (~buildml.data.dataset.Dataset, ~buildml.preprocess.result.PreprocessResult)
        The dataset with the source columns replaced by components, and a
        narrated record covering how much variance was retained, what was lost,
        and any method caveats that apply.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        A column the plan expects is missing, or the frame still contains
        missing values in those columns.

    Notes
    -----
    **PCA and UMAP genuinely transform; t-SNE approximates.** For a t-SNE plan,
    rows outside the training set are given the coordinates of their nearest
    training neighbour, because t-SNE has no out-of-sample formula. Those
    positions are a stand-in, not a projection, and the plan's ``disclosures``
    say so in the returned result. Do not build a production feature on them.

    **Components are unnamed blends.** After this step your model's feature
    importances refer to ``pc_1``, not to anything a stakeholder recognises.
    For PCA you can recover the mixture from the reducer's ``components_``
    attribute; keep that mapping if you will need to explain the model.

    See Also
    --------
    fit_reducer : Produces the plan this consumes.
    """
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
    from buildml.preprocess.columns import DEFAULT_SKIP_ROLES

    names = resolve_transform_columns(
        dataset,
        train,
        columns,
        kind="numeric",
        empty_message=(
            "No numeric feature columns available for dimensionality reduction. "
            "Encode/scale first, or pass feature column names via columns=..."
        ),
    )
    # Dimensionality reduction never consumes protected roles, even when named.
    filtered = [n for n in names if dataset.roles.get(n) not in DEFAULT_SKIP_ROLES]
    if not filtered:
        raise ValidationError(
            "No numeric columns available for dimensionality reduction "
            "(protected roles target/id/ignore/group/time/weight are excluded)"
        )
    return filtered


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
