"""Fit TDA features (and optional sklearn head) on Session train only."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.neighbors import NearestNeighbors

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.tda.catalog import resolve_backend_vectorization
from buildml.tda.extras import require_tda_stack
from buildml.tda.features import (
    encode_classification_targets,
    infer_tda_task,
    matrix_from_frame,
    regression_targets,
    resolve_tda_columns,
    standardize_fit,
    train_partition_frame,
)
from buildml.tda.homology import compute_rips_diagrams, local_point_cloud
from buildml.tda.results import TdaFitResult, TdaPlan
from buildml.tda.subsample import SubsampleStrategy, apply_train_subsample
from buildml.tda.types import TdaBackend, TdaHead, TdaTask, Vectorization
from buildml.tda.vectorize import (
    feature_names_from_state,
    fit_vectorizer_state,
    vectorize_diagrams,
)


def fit_tda(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    backend: TdaBackend | None = None,
    vectorization: Vectorization = "persistence_image",
    homology_dims: Sequence[int] = (0, 1),
    knn: int = 16,
    maxdim: int = 1,
    thresh: float | None = None,
    n_bins: int = 20,
    n_layers: int = 3,
    pixel_size: float | None = None,
    standardize: bool = True,
    head: TdaHead = "logistic_regression",
    task: TdaTask | None = None,
    columns: list[str] | None = None,
    random_state: int | None = 0,
    prefer_reduce_components: bool = True,
    reduce_plan: Any | None = None,
    max_points_guard: int = 4000,
    subsample_strategy: SubsampleStrategy = "error",
    mapper: bool = False,
) -> tuple[TdaPlan, TdaFitResult]:
    """Fit a leakage-safe TDA pipeline on the Session **train** partition.

    Builds local Vietoris–Rips clouds per row, vectorizes persistence diagrams,
    and optionally fits a sklearn head on train topological features only.
    Dispatches to giotto-tda when ``backend='giotto'`` and
    ``buildml[tda-industry]`` is installed.

    Parameters
    ----------
    dataset:
        Tabular Session dataset with target and numeric features.
    split_plan:
        Train/validation/test split. Fit uses train only.
    backend:
        ``native`` (ripser/persim) or ``giotto`` (giotto-tda). Defaults from
        :func:`buildml.tda.catalog.tda_capability_matrix`.
    vectorization:
        Persistence diagram vectorizer (see capability matrix for per-backend
        names).
    homology_dims:
        Homology dimensions to include (e.g. ``(0, 1)`` for H0 and H1).
    knn:
        Neighbors per local point cloud (must be >= 2).
    maxdim:
        Maximum homology dimension passed to the PH engine.
    thresh:
        Optional ripser filtration cutoff (native backend).
    n_bins, n_layers:
        Vectorizer grid resolution (landscapes, silhouettes, Betti curves).
    pixel_size:
        Optional persistence-image pixel size (native PI path).
    standardize:
        When True, z-score numeric features using train statistics before PH.
    head:
        Sklearn supervised head or ``none`` for transformer-only fit.
    task:
        ``classification`` or ``regression``. Inferred from target when ``None``.
    columns:
        Feature columns for clouds. Resolved from roles when ``None``.
    random_state:
        Seed for subsampling and sklearn heads.
    prefer_reduce_components:
        Prefer PCA components from an active reduce plan when available.
    reduce_plan:
        Optional dimensionality-reduction plan for column resolution.
    max_points_guard:
        Refuse or subsample when train rows exceed this count.
    subsample_strategy:
        ``error``, ``random``, or ``stratified`` when above ``max_points_guard``.
    mapper:
        When True on giotto backend, attach a diagnostic KeplerMapper summary.

    Returns
    -------
    tuple[TdaPlan, TdaFitResult]
        Frozen plan for transform/predict/evaluate and a fit report with
        disclosures and train score when a head is fitted.

    Notes
    -----
    **Leakage:** PH, vectorizer ranges, NN index, and head are train-fitted only.
    Honesty: Session-shaped PH + vectorization → sklearn: not a Mapper research
    suite.
    """
    resolved_backend, resolved_vec = resolve_backend_vectorization(
        backend=backend,
        vectorization=str(vectorization),
    )
    if resolved_backend == "giotto":
        from buildml.tda.adapters.giotto import fit_giotto

        return fit_giotto(
            dataset,
            split_plan,
            vectorization=resolved_vec,  # type: ignore[arg-type]
            homology_dims=homology_dims,
            knn=knn,
            maxdim=maxdim,
            thresh=thresh,
            n_bins=n_bins,
            n_layers=n_layers,
            pixel_size=pixel_size,
            standardize=standardize,
            head=head,
            task=task,
            columns=columns,
            random_state=random_state,
            prefer_reduce_components=prefer_reduce_components,
            reduce_plan=reduce_plan,
            max_points_guard=max_points_guard,
            subsample_strategy=subsample_strategy,
            mapper=mapper,
            **_prepare_subsample_context(
                dataset,
                split_plan,
                max_points_guard=max_points_guard,
                subsample_strategy=subsample_strategy,
                random_state=random_state,
            ),
        )
    return _fit_native(
        dataset,
        split_plan,
        vectorization=resolved_vec,  # type: ignore[arg-type]
        homology_dims=homology_dims,
        knn=knn,
        maxdim=maxdim,
        thresh=thresh,
        n_bins=n_bins,
        n_layers=n_layers,
        pixel_size=pixel_size,
        standardize=standardize,
        head=head,
        task=task,
        columns=columns,
        random_state=random_state,
        prefer_reduce_components=prefer_reduce_components,
        reduce_plan=reduce_plan,
        max_points_guard=max_points_guard,
        subsample_strategy=subsample_strategy,
    )


def _prepare_subsample_context(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    max_points_guard: int,
    subsample_strategy: SubsampleStrategy,
    random_state: int | None,
) -> dict[str, Any]:
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    target = dataset.require_target()
    full_train = train_partition_frame(dataset, split_plan)
    _, sub_train, disclosures, warnings = apply_train_subsample(
        split_plan,
        full_train,
        max_points=max_points_guard,
        strategy=subsample_strategy,
        target_column=target,
        random_state=random_state,
    )
    return {
        "train_frame": sub_train,
        "split_plan_eff": split_plan,
        "subsample_disclosures": disclosures,
        "subsample_warnings": warnings,
    }


def _fit_native(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    vectorization: Vectorization,
    homology_dims: Sequence[int],
    knn: int,
    maxdim: int,
    thresh: float | None,
    n_bins: int,
    n_layers: int,
    pixel_size: float | None,
    standardize: bool,
    head: TdaHead,
    task: TdaTask | None,
    columns: list[str] | None,
    random_state: int | None,
    prefer_reduce_components: bool,
    reduce_plan: Any | None,
    max_points_guard: int,
    subsample_strategy: SubsampleStrategy,
    train_frame: Any | None = None,
    split_plan_eff: SplitPlan | None = None,
    subsample_disclosures: Sequence[str] = (),
    subsample_warnings: Sequence[str] = (),
) -> tuple[TdaPlan, TdaFitResult]:
    """Native ripser + persim/in-tree fit path."""
    require_tda_stack(feature="fit_tda")
    assert_fit_partition(split_plan_eff or split_plan, "train")
    sp = split_plan_eff or split_plan
    assert sp is not None

    if int(knn) < 2:
        raise ValidationError("knn must be >= 2 for non-trivial local point clouds.")
    if int(maxdim) < 0:
        raise ValidationError("maxdim must be >= 0.")
    dims = tuple(sorted({int(d) for d in homology_dims}))
    if not dims:
        raise ValidationError("homology_dims must be non-empty.")
    if max(dims) > int(maxdim):
        raise ValidationError(
            f"homology_dims max {max(dims)} exceeds maxdim={maxdim}."
        )
    if int(n_bins) < 2:
        raise ValidationError("n_bins must be >= 2.")
    if int(n_layers) < 1:
        raise ValidationError("n_layers must be >= 1.")

    target = dataset.require_target()
    if train_frame is None:
        ctx = _prepare_subsample_context(
            dataset,
            sp,
            max_points_guard=max_points_guard,
            subsample_strategy=subsample_strategy,
            random_state=random_state,
        )
        train = ctx["train_frame"]
        subsample_disclosures = ctx["subsample_disclosures"]
        subsample_warnings = ctx["subsample_warnings"]
    else:
        train = train_frame

    n_train = int(len(train))
    warnings: list[str] = list(subsample_warnings)
    disclosures: list[str] = list(subsample_disclosures)

    cols, used_reduce, col_disclosures = resolve_tda_columns(
        dataset,
        train,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_column=target,
    )
    disclosures.extend(col_disclosures)

    x_raw = matrix_from_frame(train, cols)
    if standardize:
        x_train, mean, scale = standardize_fit(x_raw)
        disclosures.append(
            "Numeric features standardized with train mean/scale before PH."
        )
    else:
        x_train = x_raw
        mean = scale = None
        disclosures.append("standardize=False: PH uses raw numeric feature scales.")

    knn_eff = int(min(int(knn), n_train))
    if knn_eff < int(knn):
        warnings.append(f"knn reduced from {knn} to {knn_eff} (train size).")

    nn = NearestNeighbors(n_neighbors=knn_eff, metric="euclidean")
    nn.fit(x_train)

    train_diagrams: list[list[np.ndarray]] = []
    for i in range(n_train):
        cloud = local_point_cloud(x_train[i], nn, x_train, knn=knn_eff)
        dgms = compute_rips_diagrams(cloud, maxdim=int(maxdim), thresh=thresh)
        train_diagrams.append(dgms)

    vec_state = fit_vectorizer_state(
        train_diagrams,
        vectorization=vectorization,
        homology_dims=dims,
        n_bins=int(n_bins),
        n_layers=int(n_layers),
        pixel_size=pixel_size,
    )
    feat_names = feature_names_from_state(vec_state)
    x_tda = np.vstack([vectorize_diagrams(dgms, vec_state) for dgms in train_diagrams])

    disclosures.append(
        f"Local Vietoris–Rips (ripser) on knn={knn_eff} train neighbors per row; "
        f"vectorization={vectorization}; homology_dims={list(dims)}."
    )
    disclosures.append(
        "Vectorizer birth/death (or t) ranges fitted on train diagrams only; "
        "holdout transforms reuse the frozen ranges and train NN index."
    )

    head_key = str(head).lower().replace("-", "_")
    resolved_task: str | None = None
    head_est = None
    label_encoder = None
    classes: tuple[Any, ...] = ()
    train_score: float | None = None

    if head_key != "none":
        resolved_task = task or infer_tda_task(train[target])
        if resolved_task == "classification":
            y, label_encoder, classes = encode_classification_targets(train[target])
            head_est = _make_classifier(head_key, random_state=random_state)
            head_est.fit(x_tda, y)
            train_score = float(head_est.score(x_tda, y))
        elif resolved_task == "regression":
            y = regression_targets(train[target])
            head_est = _make_regressor(head_key, random_state=random_state)
            head_est.fit(x_tda, y)
            train_score = float(head_est.score(x_tda, y))
        else:
            raise ValidationError(f"Unknown TDA task {resolved_task!r}.")
        disclosures.append(
            f"Supervised head={head_key} fitted on train topological features only "
            f"(task={resolved_task})."
        )
    else:
        disclosures.append("head='none': TDA transformer only; no supervised head fitted.")

    plan = TdaPlan(
        backend="native",
        vectorization=str(vectorization),
        columns=tuple(cols),
        homology_dims=dims,
        knn=knn_eff,
        maxdim=int(maxdim),
        thresh=None if thresh is None else float(thresh),
        n_bins=int(n_bins),
        n_layers=int(n_layers),
        n_train_rows=n_train,
        feature_dim=int(x_tda.shape[1]),
        feature_names=feat_names,
        task=resolved_task,
        head=head_key,
        used_reduce_components=used_reduce,
        standardize=bool(standardize),
        mean_=None if mean is None else np.asarray(mean, dtype=float),
        scale_=None if scale is None else np.asarray(scale, dtype=float),
        train_x_=np.asarray(x_train, dtype=float),
        nn_=nn,
        vectorizer_state_=vec_state,
        head_estimator_=head_est,
        label_encoder_=label_encoder,
        classes_=classes,
        train_tda_features_=np.asarray(x_tda, dtype=float),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        config={
            "backend": "native",
            "vectorization": str(vectorization),
            "homology_dims": list(dims),
            "knn": knn_eff,
            "maxdim": int(maxdim),
            "thresh": thresh,
            "n_bins": int(n_bins),
            "n_layers": int(n_layers),
            "pixel_size": pixel_size,
            "standardize": bool(standardize),
            "head": head_key,
            "task": resolved_task,
            "random_state": random_state,
            "prefer_reduce_components": prefer_reduce_components,
            "max_points_guard": int(max_points_guard),
            "subsample_strategy": subsample_strategy,
        },
    )
    result = TdaFitResult(
        backend="native",
        vectorization=str(vectorization),
        n_train_rows=n_train,
        feature_dim=int(x_tda.shape[1]),
        homology_dims=dims,
        knn=knn_eff,
        columns=tuple(cols),
        task=resolved_task,
        head=head_key,
        train_score=train_score,
        used_reduce_components=used_reduce,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, result


def _make_classifier(name: str, *, random_state: int | None) -> Any:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression

    if name == "logistic_regression":
        return LogisticRegression(max_iter=500, random_state=random_state)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=1
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=random_state)
    raise ValidationError(
        f"Head {name!r} is not a classification estimator; use "
        "logistic_regression, random_forest, hist_gradient_boosting, or none."
    )


def _make_regressor(name: str, *, random_state: int | None) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge

    if name in {"ridge", "logistic_regression"}:
        return Ridge(random_state=random_state)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=100, random_state=random_state, n_jobs=1
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=random_state)
    raise ValidationError(
        f"Head {name!r} is not a regression estimator; use ridge, "
        "random_forest, hist_gradient_boosting, or none."
    )
