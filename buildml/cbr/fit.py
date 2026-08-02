"""Build a case base from Session train only."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.cbr.cases import Case, CaseBase, encode_categoricals
from buildml.cbr.features import (
    classification_accuracy,
    encode_classification_targets,
    matrix_from_frame,
    numeric_ranges,
    regression_metrics,
    regression_targets,
    resolve_categorical_columns,
    resolve_cbr_columns,
    standardize_fit,
    train_partition_frame,
)
from buildml.cbr.catalog import resolve_backend_metric
from buildml.cbr.predict import predict_cbr
from buildml.cbr.results import CbrFitResult, CbrPlan
from buildml.cbr.retrieval_build import build_search_artifacts
from buildml.cbr.types import CbrAdaptMode, CbrBackend, CbrMetric, CbrReuseMode, CbrTask
from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition


def fit_cbr(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    backend: CbrBackend | None = None,
    task: CbrTask | None = None,
    metric: CbrMetric = "euclidean",
    reuse: CbrReuseMode = "distance_weighted",
    adapt: CbrAdaptMode = "none",
    k: int = 5,
    columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    text_columns: list[str] | None = None,
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    standardize: bool = True,
    distance_eps: float = 1e-8,
    random_state: int | None = 0,
    prefer_reduce_components: bool = True,
    reduce_plan: Any | None = None,
    torch_epochs: int = 40,
    torch_learning_rate: float = 1e-3,
    torch_hidden_dim: int = 64,
    torch_embed_dim: int = 32,
    device: str = "cpu",
) -> tuple[CbrPlan, CbrFitResult]:
    """Build a tabular case memory from Session **train** rows.

    Honesty
    -------
    Each train row becomes a case: features + solution (label/outcome).
    Retrieval uses numpy/sklearn-style distances (euclidean / manhattan /
    cosine / mixed Gower-style). Reuse votes or averages neighbor solutions.
    This is **case-based reasoning** for supervised-style tasks — **not** RAG
    (document retrieval for generation), not a vector DB, not a full cognitive
    CBR research platform.

    Leakage: validation/test rows never enter the case base at fit time.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    if int(k) < 1:
        raise ValidationError("k must be >= 1.")
    if float(distance_eps) <= 0:
        raise ValidationError("distance_eps must be > 0.")

    metric_key = str(metric).lower().replace("-", "_")
    text_cols_tuple = tuple(text_columns or ())
    resolved_backend, resolved_metric = resolve_backend_metric(
        backend=backend,
        metric=metric_key,
        text_columns=list(text_cols_tuple) if text_cols_tuple else None,
    )
    metric_key = resolved_metric
    if metric_key not in {"euclidean", "manhattan", "cosine", "mixed"}:
        raise ValidationError(
            f"Unknown CBR metric {metric!r}; expected euclidean, manhattan, "
            "cosine, or mixed."
        )
    reuse_key = str(reuse).lower().replace("-", "_")
    if reuse_key not in {
        "majority",
        "distance_weighted",
        "local_mean",
        "local_ridge",
    }:
        raise ValidationError(
            f"Unknown CBR reuse mode {reuse!r}; expected majority, "
            "distance_weighted, local_mean, or local_ridge."
        )
    adapt_key = str(adapt).lower().replace("-", "_")
    if adapt_key not in {"none", "offset"}:
        raise ValidationError(
            f"Unknown CBR adapt mode {adapt!r}; expected none or offset."
        )

    target = dataset.require_target()
    train = train_partition_frame(dataset, split_plan)
    resolved_task = _resolve_task(dataset, train[target], task)
    _validate_reuse_for_task(reuse_key, resolved_task)

    cols, used_reduce, disclosures = resolve_cbr_columns(
        dataset,
        train,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_column=target,
    )
    cat_cols, cat_notes = resolve_categorical_columns(
        dataset,
        train,
        categorical_columns,
        target_column=target,
        numeric_columns=cols,
    )
    disclosures.extend(cat_notes)
    warnings: list[str] = []

    if metric_key == "mixed" and not cat_cols and not cols:
        raise ValidationError("metric='mixed' requires numeric and/or categorical columns.")
    if metric_key == "mixed" and resolved_backend != "sklearn":
        warnings.append(
            f"metric='mixed' with backend='{resolved_backend}' uses exact sklearn "
            "mixed distances (ANN applies to numeric/embedding search matrix only)."
        )
    if metric_key == "mixed" and not cat_cols:
        disclosures.append(
            "metric='mixed' with numeric-only features degenerates to "
            "range-normalized Manhattan (Gower numeric term)."
        )
    if metric_key != "mixed" and cat_cols:
        warnings.append(
            "categorical_columns were provided but metric is not 'mixed'; "
            "categoricals are stored but ignored by pure numeric metrics."
        )

    n_train = int(len(split_plan.train_indices))
    x_raw = matrix_from_frame(train, cols) if cols else np.zeros((len(train), 0))
    mean = scale = ranges = None
    if cols and standardize and metric_key != "mixed":
        x_num, mean, scale = standardize_fit(x_raw)
        disclosures.append(
            "Numeric features standardized with train mean/scale for distance "
            f"(metric={metric_key})."
        )
    else:
        x_num = x_raw
        if cols and metric_key == "mixed":
            ranges = numeric_ranges(x_raw)
            disclosures.append(
                "mixed metric: numeric |Δ| range-normalized with train ranges; "
                "no z-score standardization."
            )
        elif cols and not standardize:
            disclosures.append(
                "standardize=False: distances use raw numeric feature scales."
            )

    # Categorical codes + vocabularies (train-fit).
    cat_vocabs: list[tuple[Any, ...]] = []
    if cat_cols:
        cat_codes = np.column_stack(
            [
                encode_categoricals(
                    train[c].tolist(),
                    tuple(sorted(train[c].astype(str).unique().tolist())),
                )
                for c in cat_cols
            ]
        )
        for c in cat_cols:
            cat_vocabs.append(tuple(sorted(train[c].astype(str).unique().tolist())))
    else:
        cat_codes = np.zeros((len(train), 0), dtype=int)

    classes: tuple[Any, ...] | None = None
    label_encoder = None
    if resolved_task == "classification":
        y_codes, label_encoder, classes = encode_classification_targets(train[target])
        if len(classes) < 2:
            raise ValidationError("CBR classification requires at least 2 classes.")
        solutions: list[Any] = list(
            label_encoder.inverse_transform(y_codes)
        )
        # Prefer original-ish labels via decode path.
        from buildml.cbr.features import decode_predictions

        solutions = decode_predictions(y_codes, label_encoder)
        y_fit = np.asarray(y_codes, dtype=int)
    else:
        y_num = regression_targets(train[target])
        solutions = [float(v) for v in y_num]
        y_fit = np.asarray(y_num, dtype=float)

    cases: list[Case] = []
    indices = list(train.index)
    for i, idx in enumerate(indices):
        cases.append(
            Case(
                case_id=f"train-{i}",
                row_index=idx,
                solution=solutions[i],
                numeric_features=tuple(float(v) for v in x_num[i]) if cols else (),
                categorical_features=tuple(
                    train[c].iloc[i] for c in cat_cols
                ),
                source="train",
            )
        )

    metric_doc = {
        "euclidean": "L2 distance on (optionally standardized) numeric features.",
        "manhattan": "L1 distance on (optionally standardized) numeric features.",
        "cosine": "1 - cosine similarity on numeric features.",
        "mixed": (
            "Gower-style: mean of range-normalized |Δ| over numerics and "
            "mismatch rate over categoricals, weighted by feature counts."
        ),
    }[metric_key]
    disclosures.append(f"Distance metric: {metric_key} — {metric_doc}")
    disclosures.append(
        "Case base built from Session train only. Holdout is for "
        "retrieve_cases / predict_cbr / evaluate_cbr — never for memory at fit."
    )
    disclosures.append(
        "Honesty: tabular CBR (case→solution) — not RAG document retrieval, "
        "not a vector DB product, not a full cognitive CBR research suite."
    )
    disclosures.append(
        f"Backend={resolved_backend}; reuse mode={reuse_key}; adapt={adapt_key}; "
        f"k={int(k)}."
    )
    if resolved_backend == "sklearn":
        disclosures.append("Retrieval: exact kNN (numpy/sklearn distances).")
    elif resolved_backend == "industry":
        disclosures.append("Retrieval: approximate NN when buildml[cbr-industry] present.")
    elif resolved_backend == "embedding":
        disclosures.append(
            "Retrieval: sentence-transformer case embeddings (buildml[rag|ssl])."
        )
    elif resolved_backend == "torch":
        disclosures.append(
            "Retrieval: learned metric encoder + kNN (buildml[torch])."
        )

    ann_metric = metric_key if metric_key in {"euclidean", "cosine"} else "euclidean"
    if resolved_backend == "sklearn":
        search_matrix = np.asarray(x_num, dtype=float)
        ann_index = None
        ann_library = None
        embedder_id = None
        torch_encoder = None
        artifact_notes: list[str] = []
    else:
        (
            search_matrix,
            ann_index,
            ann_library,
            embedder_id,
            torch_encoder,
            artifact_notes,
        ) = build_search_artifacts(
            backend=resolved_backend,
            train_frame=train,
            numeric_matrix=np.asarray(x_num, dtype=float),
            task=resolved_task,
            metric=ann_metric,
            text_columns=text_cols_tuple,
            text_model_name=text_model_name,
            y_fit=y_fit,
            torch_epochs=torch_epochs,
            torch_learning_rate=torch_learning_rate,
            torch_hidden_dim=torch_hidden_dim,
            torch_embed_dim=torch_embed_dim,
            device=device,
            random_state=random_state,
            n_classes=len(classes) if classes is not None else 2,
        )
    disclosures.extend(artifact_notes)

    case_base = CaseBase(
        cases=tuple(cases),
        numeric_matrix=np.asarray(x_num, dtype=float),
        categorical_matrix=cat_codes,
        numeric_columns=tuple(cols),
        categorical_columns=tuple(cat_cols),
        metric=metric_key,
        numeric_mean_=mean,
        numeric_scale_=scale,
        numeric_ranges_=ranges,
        cat_vocabularies_=tuple(cat_vocabs),
        search_matrix_=search_matrix,
        ann_index_=ann_index,
        ann_library_=ann_library,
        text_embedder_id_=embedder_id,
        torch_encoder_=torch_encoder,
        disclosures=tuple(disclosures),
        n_retained=0,
    )

    config = {
        "task": resolved_task,
        "backend": resolved_backend,
        "metric": metric_key,
        "reuse": reuse_key,
        "adapt": adapt_key,
        "k": int(k),
        "columns": cols,
        "categorical_columns": cat_cols,
        "text_columns": list(text_cols_tuple),
        "text_model_name": text_model_name,
        "standardize": bool(standardize),
        "distance_eps": float(distance_eps),
        "random_state": random_state,
        "prefer_reduce_components": prefer_reduce_components,
        "torch_epochs": int(torch_epochs),
        "torch_learning_rate": float(torch_learning_rate),
        "torch_hidden_dim": int(torch_hidden_dim),
        "torch_embed_dim": int(torch_embed_dim),
        "device": device,
    }
    plan = CbrPlan(
        task=resolved_task,
        backend=resolved_backend,
        metric=metric_key,
        reuse=reuse_key,
        adapt=adapt_key,
        k=int(k),
        columns=tuple(cols),
        categorical_columns=tuple(cat_cols),
        text_columns=text_cols_tuple,
        text_model_name=text_model_name,
        target_column=target,
        n_train_rows=n_train,
        case_base=case_base,
        classes_=classes,
        label_encoder_=label_encoder,
        distance_eps=float(distance_eps),
        standardize=bool(standardize),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        used_reduce_components=used_reduce,
        config=config,
    )

    # Train leave-in score for disclosure (not a holdout claim).
    train_score: float | None = None
    try:
        # Leave-one-out lite on a subsample is expensive; score full train
        # with k neighbors (includes self as nearest — disclose that).
        pred = predict_cbr(
            dataset, plan, split_plan, partition="train", return_traces=False
        )
        if resolved_task == "classification":
            train_score = classification_accuracy(
                train[target].tolist(), list(pred.predictions)
            )
            disclosures_note = (
                "train_score is in-sample (query's own case is typically the "
                "nearest neighbor) — not a holdout metric."
            )
            plan = _plan_with_disclosure(plan, disclosures_note)
        else:
            train_score = regression_metrics(
                regression_targets(train[target]),
                np.asarray(pred.predictions, dtype=float),
            ).get("r2")
            disclosures_note = (
                "train_score (R2) is in-sample — not a holdout metric."
            )
            plan = _plan_with_disclosure(plan, disclosures_note)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Could not compute in-sample train_score: {exc}")

    result = CbrFitResult(
        task=resolved_task,
        backend=resolved_backend,
        metric=metric_key,
        reuse=reuse_key,
        k=int(k),
        n_train_rows=n_train,
        n_cases=case_base.n_cases,
        columns=tuple(cols),
        categorical_columns=tuple(cat_cols),
        target_column=target,
        classes=classes,
        train_score=train_score,
        disclosures=plan.disclosures,
        warnings=tuple(warnings),
    )
    return plan, result


def _plan_with_disclosure(plan: CbrPlan, note: str) -> CbrPlan:
    return CbrPlan(
        task=plan.task,
        backend=plan.backend,
        metric=plan.metric,
        reuse=plan.reuse,
        adapt=plan.adapt,
        k=plan.k,
        columns=plan.columns,
        categorical_columns=plan.categorical_columns,
        text_columns=plan.text_columns,
        text_model_name=plan.text_model_name,
        target_column=plan.target_column,
        n_train_rows=plan.n_train_rows,
        case_base=plan.case_base,
        classes_=plan.classes_,
        label_encoder_=plan.label_encoder_,
        distance_eps=plan.distance_eps,
        standardize=plan.standardize,
        disclosures=tuple([*plan.disclosures, note]),
        warnings=plan.warnings,
        used_reduce_components=plan.used_reduce_components,
        config=plan.config,
    )


def _resolve_task(
    dataset: Dataset, y: Any, task: CbrTask | None
) -> CbrTask:
    if task is not None:
        t = str(task).lower()
        if t not in {"classification", "regression"}:
            raise ValidationError(f"Unknown task {task!r}.")
        return t  # type: ignore[return-value]
    import pandas as pd

    series = y if isinstance(y, pd.Series) else pd.Series(y)
    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 20:
        return "regression"
    return "classification"


def _validate_reuse_for_task(reuse: str, task: str) -> None:
    if task == "classification" and reuse in {"local_mean", "local_ridge"}:
        raise ValidationError(
            f"reuse={reuse!r} is regression-only; use majority or "
            "distance_weighted for classification."
        )
    if task == "regression" and reuse == "majority":
        raise ValidationError(
            "reuse='majority' is classification-only; use distance_weighted, "
            "local_mean, or local_ridge for regression."
        )
