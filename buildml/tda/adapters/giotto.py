"""giotto-tda industry adapter for persistent homology + vectorization."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.neighbors import NearestNeighbors

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.tda.extras import require_giotto
from buildml.tda.features import (
    encode_classification_targets,
    infer_tda_task,
    matrix_from_frame,
    regression_targets,
    resolve_tda_columns,
    standardize_fit,
    train_partition_frame,
)
from buildml.tda.fit import _make_classifier, _make_regressor
from buildml.tda.homology import local_point_cloud
from buildml.tda.results import TdaFitResult, TdaPlan
from buildml.tda.types import TdaHead, TdaTask, Vectorization
from buildml.tda.vectorize import feature_names_from_state


def _require_gtda():
    gtda = require_giotto(feature="fit_tda (giotto backend)")
    return gtda


def compute_giotto_diagrams(
    point_cloud: np.ndarray,
    *,
    homology_dims: Sequence[int],
    maxdim: int,
) -> list[np.ndarray]:
    """Compute VR persistence diagrams for one point cloud via giotto-tda."""
    gtda = _require_gtda()
    from gtda.homology import VietorisRipsPersistence

    cloud = np.asarray(point_cloud, dtype=float)
    if cloud.ndim != 2 or cloud.shape[0] < 2:
        return [np.zeros((0, 2), dtype=float) for _ in range(int(maxdim) + 1)]
    dims = sorted({int(d) for d in homology_dims})
    vr = VietorisRipsPersistence(homology_dimensions=dims, collapse_edges=True)
    # giotto expects (n_samples, n_points, n_features)
    batch = cloud.reshape(1, *cloud.shape)
    raw = vr.fit_transform(batch)[0]
    # raw shape (n_pairs, 3): birth, death, dimension
    out: list[np.ndarray] = []
    for dim in range(int(maxdim) + 1):
        if dim in dims:
            mask = np.isclose(raw[:, 2], float(dim))
            sub = raw[mask, :2]
            finite = sub[np.isfinite(sub).all(axis=1)]
            finite = finite[(finite[:, 1] - finite[:, 0]) > 1e-12]
            out.append(np.asarray(finite, dtype=float))
        else:
            out.append(np.zeros((0, 2), dtype=float))
    return out


def fit_giotto_vectorizer_state(
    train_diagrams: Sequence[Sequence[np.ndarray]],
    *,
    vectorization: str,
    homology_dims: Sequence[int],
    n_bins: int,
    n_layers: int,
) -> dict[str, Any]:
    """Fit giotto vectorizer parameters on train diagrams."""
    gtda = _require_gtda()
    key = str(vectorization).lower().replace("-", "_")
    dims = tuple(int(d) for d in homology_dims)
    stacked = _diagrams_to_giotto_batch(train_diagrams, dims)

    if key == "betti_curve":
        from gtda.diagrams import BettiCurve

        bc = BettiCurve(n_bins=int(n_bins))
        probe = bc.fit_transform(stacked)
        per_dim = int(probe.shape[1] // max(len(dims), 1))
        return {
            "kind": "giotto_betti_curve",
            "homology_dims": dims,
            "n_bins": int(n_bins),
            "per_dim": per_dim,
            "feature_dim": int(probe.shape[1]),
            "giotto_obj": bc,
        }

    if key in {"persistence_image", "persistence_landscape", "landscape"}:
        if key == "persistence_image":
            from gtda.diagrams import PersistenceImage

            vec = PersistenceImage(n_bins=int(n_bins))
        else:
            from gtda.diagrams import PersistenceLandscape

            vec = PersistenceLandscape(n_layers=int(n_layers), n_bins=int(n_bins))
        probe = vec.fit_transform(stacked)
        per_dim = int(probe.shape[1] // max(len(dims), 1))
        kind = (
            "giotto_persistence_image"
            if key == "persistence_image"
            else "giotto_persistence_landscape"
        )
        return {
            "kind": kind,
            "homology_dims": dims,
            "n_bins": int(n_bins),
            "n_layers": int(n_layers),
            "per_dim": per_dim,
            "feature_dim": int(probe.shape[1]),
            "giotto_obj": vec,
        }

    raise ValidationError(
        f"giotto backend does not support vectorization={vectorization!r}. "
        "Choose persistence_image, persistence_landscape, landscape, or betti_curve."
    )


def vectorize_giotto_diagrams(
    diagrams: Sequence[np.ndarray],
    state: dict[str, Any],
) -> np.ndarray:
    """Vectorize one sample's diagrams using a fitted giotto vectorizer."""
    dims = tuple(int(d) for d in state["homology_dims"])
    batch = _diagrams_to_giotto_batch([diagrams], dims)
    obj = state.get("giotto_obj")
    if obj is None:
        raise ValidationError("giotto vectorizer state missing giotto_obj.")
    out = np.asarray(obj.transform(batch)[0], dtype=float)
    target = int(state["feature_dim"])
    if out.size < target:
        out = np.pad(out, (0, target - out.size))
    return out[:target]


def _diagrams_to_giotto_batch(
    train_diagrams: Sequence[Sequence[np.ndarray]],
    dims: Sequence[int],
) -> np.ndarray:
    """Stack diagrams into giotto format (n_samples, n_points, 3)."""
    rows: list[np.ndarray] = []
    for sample in train_diagrams:
        chunks: list[np.ndarray] = []
        for d in dims:
            if d < len(sample):
                dgm = np.asarray(sample[d], dtype=float)
                if dgm.size:
                    extra = np.full((dgm.shape[0], 1), float(d), dtype=float)
                    chunks.append(np.hstack([dgm, extra]))
        if chunks:
            rows.append(np.vstack(chunks))
        else:
            rows.append(np.zeros((1, 3), dtype=float))
    max_pts = max(r.shape[0] for r in rows)
    padded = np.zeros((len(rows), max_pts, 3), dtype=float)
    for i, row in enumerate(rows):
        padded[i, : row.shape[0], :] = row
    return padded


def _optional_mapper_summary(
    x_train: np.ndarray,
    y: np.ndarray | None,
    *,
    random_state: int | None,
) -> dict[str, Any] | None:
    """Build a lightweight KeplerMapper summary on train (giotto-tda)."""
    gtda = _require_gtda()
    try:
        from gtda.mapper import make_mapper_pipeline
        from sklearn.cluster import DBSCAN
        from sklearn.decomposition import PCA
    except ImportError:
        return None

    n = len(x_train)
    if n < 20:
        return None
    cap = min(n, 500)
    rng = np.random.default_rng(random_state)
    if n > cap:
        idx = rng.choice(n, size=cap, replace=False)
        sample = x_train[idx]
        sample_y = None if y is None else y[idx]
    else:
        sample = x_train
        sample_y = y

    n_components = min(3, sample.shape[1])
    pipeline = make_mapper_pipeline(
        filter_func=PCA(n_components=n_components),
        cover=None,
        clusterer=DBSCAN(eps=0.8, min_samples=3),
    )
    graph = pipeline.fit_transform(sample)
    n_nodes = len(getattr(graph, "nodes", lambda: {})())
    n_edges = len(getattr(graph, "edges", lambda: {})())
    summary: dict[str, Any] = {
        "n_train_mapper_points": int(len(sample)),
        "n_mapper_nodes": int(n_nodes),
        "n_mapper_edges": int(n_edges),
        "filter": f"PCA(n_components={n_components})",
        "clusterer": "DBSCAN(eps=0.8, min_samples=3)",
    }
    if sample_y is not None:
        summary["has_labels"] = True
    return summary


def fit_giotto(
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
    subsample_strategy: str = "error",
    mapper: bool = False,
    train_frame: Any | None = None,
    split_plan_eff: SplitPlan | None = None,
    subsample_disclosures: Sequence[str] = (),
    subsample_warnings: Sequence[str] = (),
) -> tuple[TdaPlan, TdaFitResult]:
    """Fit TDA via giotto-tda (buildml[tda-industry])."""
    _ = pixel_size, thresh  # giotto VR uses its own filtration; thresh unused here
    require_giotto(feature="fit_tda backend='giotto'")
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    target = dataset.require_target()
    if train_frame is None:
        from buildml.tda.subsample import apply_train_subsample

        full_train = train_partition_frame(dataset, split_plan)
        _, train, sub_disclosures, sub_warnings = apply_train_subsample(
            split_plan,
            full_train,
            max_points=max_points_guard,
            strategy=subsample_strategy,  # type: ignore[arg-type]
            target_column=target,
            random_state=random_state,
        )
        subsample_disclosures = tuple(sub_disclosures)
        subsample_warnings = tuple(sub_warnings)
    else:
        train = train_frame

    dims = tuple(sorted({int(d) for d in homology_dims}))
    cols, used_reduce, disclosures = resolve_tda_columns(
        dataset,
        train,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_column=target,
    )
    warnings = list(subsample_warnings)
    disclosures = list(subsample_disclosures) + list(disclosures)

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

    n_train = int(len(x_train))
    knn_eff = int(min(int(knn), n_train))
    if knn_eff < int(knn):
        warnings.append(f"knn reduced from {knn} to {knn_eff} (train size).")

    nn = NearestNeighbors(n_neighbors=knn_eff, metric="euclidean")
    nn.fit(x_train)

    train_diagrams: list[list[np.ndarray]] = []
    for i in range(n_train):
        cloud = local_point_cloud(x_train[i], nn, x_train, knn=knn_eff)
        train_diagrams.append(
            compute_giotto_diagrams(cloud, homology_dims=dims, maxdim=int(maxdim))
        )

    vec_state = fit_giotto_vectorizer_state(
        train_diagrams,
        vectorization=str(vectorization),
        homology_dims=dims,
        n_bins=int(n_bins),
        n_layers=int(n_layers),
    )
    feat_names = feature_names_from_state(vec_state)
    x_tda = np.vstack(
        [vectorize_giotto_diagrams(dgms, vec_state) for dgms in train_diagrams]
    )

    disclosures.append(
        f"giotto-tda VietorisRipsPersistence on knn={knn_eff} train neighbors; "
        f"vectorization={vectorization}; homology_dims={list(dims)}."
    )
    disclosures.append(
        "Giotto vectorizer fitted on train diagrams only; holdout transforms "
        "reuse frozen vectorizer + train NN index."
    )

    mapper_summary = None
    if mapper:
        y_arr = None
        if target in train.columns:
            y_arr = train[target].to_numpy()
        mapper_summary = _optional_mapper_summary(
            x_train, y_arr, random_state=random_state
        )
        if mapper_summary:
            disclosures.append(
                f"KeplerMapper summary on train subsample: "
                f"{mapper_summary['n_mapper_nodes']} nodes, "
                f"{mapper_summary['n_mapper_edges']} edges "
                f"(diagnostic only — not used as supervised features)."
            )
        else:
            warnings.append("mapper=True but Mapper pipeline could not be built.")

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
        backend="giotto",
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
        mapper_summary_=mapper_summary,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        config={
            "backend": "giotto",
            "vectorization": str(vectorization),
            "homology_dims": list(dims),
            "knn": knn_eff,
            "maxdim": int(maxdim),
            "n_bins": int(n_bins),
            "n_layers": int(n_layers),
            "standardize": bool(standardize),
            "head": head_key,
            "task": resolved_task,
            "random_state": random_state,
            "prefer_reduce_components": prefer_reduce_components,
            "max_points_guard": int(max_points_guard),
            "mapper": bool(mapper),
        },
    )
    result = TdaFitResult(
        backend="giotto",
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


def transform_diagrams_giotto(
    cloud: np.ndarray,
    *,
    plan: Any,
) -> list[np.ndarray]:
    """Compute giotto diagrams for one local cloud using plan settings."""
    return compute_giotto_diagrams(
        cloud,
        homology_dims=plan.homology_dims,
        maxdim=plan.maxdim,
    )
