"""Evaluate synthetic utility (TSTR) and fidelity metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from buildml.core.errors import ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, frame_for_partition
from buildml.synthetic.adapters.sdmetrics_eval import sdmetrics_quality_scores
from buildml.synthetic.extras import sdmetrics_available
from buildml.synthetic.features import require_split
from buildml.synthetic.results import SynthesizerPlan, SyntheticEvalResult
from buildml.synthetic.sample import sample_synthetic
from buildml.synthetic.types import EvalBackend, EvalMode


def evaluate_synthetic(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    plan: SynthesizerPlan,
    *,
    mode: EvalMode = "fidelity",
    eval_backend: EvalBackend = "auto",
    partition: str = "test",
    n_synthetic: int | None = None,
    random_state: int = 0,
    estimator: str = "auto",
) -> SyntheticEvalResult:
    """Evaluate a frozen synthesizer.

Modes
-----
fidelity
    Column-wise KS (continuous/integer) and total-variation (categorical),
    plus continuous pairwise correlation L1. When ``eval_backend='sdmetrics'``
    or ``eval_backend='auto'`` with SDMetrics installed, also reports
    SDMetrics QualityReport scores.
tstr
    Train-on-Synthetic, Test-on-Real: fit a simple sklearn estimator on
    synthetic samples, score on the real holdout partition. Discloses that
    this is a utility proxy, not a generative quality certificate.
Never refits the generator on the evaluation partition.

Parameters
----------
dataset:
    BuildML dataset with features, target, and role metadata.
split_plan:
    Train/validation/test split; fit uses train partition only.
plan:
    Fitted plan object carrying model state and feature contract.
mode:
    Anomaly detection mode (``unsupervised`` or ``supervised``).
eval_backend:
    eval backend (EvalBackend).
partition:
    Holdout partition name or ``all`` for the full frame.
n_synthetic:
    n synthetic (int | None).
random_state:
    Seed for stochastic steps (sampling, initialization, bagging).
estimator:
    Fitted model object used for scoring or prediction.

Returns
-------
SyntheticEvalResult
    Serializable result summary (SyntheticEvalResult) for history recording.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    if plan is None or plan.generator_ is None:
        raise ValidationError("No fitted synthesizer. Call fit_synthesizer(...) first.")
    split = require_split(split_plan)
    if mode not in {"fidelity", "tstr"}:
        raise ValidationError(f"Unknown evaluate_synthetic mode: {mode!r}")
    if partition not in {"train", "validation", "test"}:
        raise ValidationError(
            "evaluate_synthetic partition must be train|validation|test."
        )

    real = frame_for_partition(dataset, split, partition)  # type: ignore[arg-type]
    cols = [c for c in plan.columns if c in real.columns]
    if not cols:
        raise ValidationError("No overlapping columns between plan and evaluation frame.")
    real_sub = real[cols].copy()
    n_syn = int(n_synthetic or max(len(real_sub), plan.n_rows_fitted))
    sample = sample_synthetic(plan, n=n_syn, random_state=random_state)
    assert sample.frame is not None
    syn_sub = sample.frame[[c for c in cols if c in sample.frame.columns]].copy()

    disclosures = [
        f"mode={mode!r} comparing synthetic(n={len(syn_sub)}) to real "
        f"partition={partition!r} (n={len(real_sub)}).",
        f"Generator was fitted on train only (n={plan.n_rows_fitted}).",
        "Not a differential-privacy or membership-inference audit.",
    ]
    warnings: list[str] = []
    if partition == "train":
        warnings.append(
            "Evaluating against train measures reconstruction/fidelity to the "
            "fit partition: prefer partition='test' for holdout utility."
        )

    if mode == "fidelity":
        metrics, per_column = _fidelity_metrics(real_sub, syn_sub, plan)
        resolved_eval = _resolve_eval_backend(eval_backend)
        if resolved_eval == "sdmetrics":
            try:
                sdv_meta = _plan_sdv_metadata(plan)
                sd_metrics, sd_warn = sdmetrics_quality_scores(
                    real_sub, syn_sub, metadata=sdv_meta
                )
                metrics.update(sd_metrics)
                warnings.extend(sd_warn)
                disclosures.append(
                    "SDMetrics QualityReport scores appended "
                    "(buildml[synthetic-industry])."
                )
            except (ImportError, OSError, RuntimeError) as exc:
                # Broken torch/sdmetrics wheels (common on Windows): keep builtin
                # fidelity metrics and disclose the skip instead of hard-failing.
                warnings.append(
                    f"eval_backend='sdmetrics' skipped ({type(exc).__name__}: {exc}); "
                    "reporting builtin fidelity metrics only."
                )
                disclosures.append(
                    "SDMetrics path unavailable at runtime; builtin KS/TV/corr metrics used."
                )
        disclosures.append(
            "Fidelity metrics: KS distance (cont/int), total variation (cat), "
            "mean absolute correlation difference (cont)."
        )
        return SyntheticEvalResult(
            mode=mode,
            partition=partition,
            method=plan.method,
            n_real=int(len(real_sub)),
            n_synthetic=int(len(syn_sub)),
            metrics=metrics,
            per_column=per_column,
            disclosures=tuple(disclosures),
            warnings=tuple(warnings),
        )

    # TSTR
    target = plan.target_column
    if target is None:
        for name, role in dataset.roles.items():
            if role == ColumnRole.TARGET:
                target = name
                break
    if target is None or target not in cols:
        raise ValidationError(
            "mode='tstr' requires a target column present in the synthesizer plan "
            "(set a target role before fit_synthesizer, or use method='smote')."
        )
    feature_cols = [c for c in cols if c != target]
    if not feature_cols:
        raise ValidationError("mode='tstr' needs ≥1 feature column.")

    y_real = real_sub[target]
    task = _infer_task(y_real)
    metrics, warn = _tstr_metrics(
        syn_sub,
        real_sub,
        feature_cols=feature_cols,
        target=target,
        task=task,
        estimator=estimator,
        random_state=random_state,
    )
    warnings.extend(warn)
    disclosures.append(
        "TSTR = train estimator on synthetic, evaluate on real holdout "
        "(utility proxy). Also reports TRTR baseline when train labels exist."
    )
    # Optional TRTR baseline from real train
    try:
        train_real = frame_for_partition(dataset, split, "train")
        if target in train_real.columns:
            trtr, _ = _tstr_metrics(
                train_real[cols],
                real_sub,
                feature_cols=feature_cols,
                target=target,
                task=task,
                estimator=estimator,
                random_state=random_state,
            )
            for key, value in trtr.items():
                metrics[f"trtr_{key}"] = value
            if "score" in trtr and "score" in metrics:
                metrics["tstr_gap_vs_trtr"] = float(trtr["score"] - metrics["score"])
    except Exception:
        warnings.append("TRTR baseline unavailable; reporting TSTR only.")

    return SyntheticEvalResult(
        mode=mode,
        partition=partition,
        method=plan.method,
        n_real=int(len(real_sub)),
        n_synthetic=int(len(syn_sub)),
        metrics=metrics,
        per_column={},
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def _fidelity_metrics(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    plan: SynthesizerPlan,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    kind_map = {s.name: s.kind for s in plan.column_specs}
    per_column: dict[str, dict[str, float]] = {}
    ks_vals: list[float] = []
    tv_vals: list[float] = []
    for col in real.columns:
        kind = kind_map.get(col, "continuous")
        if kind in {"continuous", "integer"}:
            a = pd.to_numeric(real[col], errors="coerce").dropna().to_numpy(dtype=float)
            b = pd.to_numeric(syn[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(a) < 2 or len(b) < 2:
                dist = 1.0
            else:
                dist = float(stats.ks_2samp(a, b).statistic)
            per_column[col] = {"ks": dist, "kind": 0.0 if kind == "continuous" else 1.0}
            ks_vals.append(dist)
        else:
            tv = _total_variation(real[col], syn[col])
            per_column[col] = {"tv": tv}
            tv_vals.append(tv)

    cont_cols = [
        c for c in real.columns if kind_map.get(c) in {"continuous", "integer"}
    ]
    corr_l1 = 0.0
    if len(cont_cols) >= 2:
        r_corr = (
            real[cont_cols]
            .apply(pd.to_numeric, errors="coerce")
            .corr()
            .to_numpy(dtype=float)
        )
        s_corr = (
            syn[cont_cols]
            .apply(pd.to_numeric, errors="coerce")
            .corr()
            .to_numpy(dtype=float)
        )
        mask = np.isfinite(r_corr) & np.isfinite(s_corr)
        if mask.any():
            corr_l1 = float(np.mean(np.abs(r_corr[mask] - s_corr[mask])))

    metrics = {
        "mean_ks": float(np.mean(ks_vals)) if ks_vals else float("nan"),
        "mean_tv": float(np.mean(tv_vals)) if tv_vals else float("nan"),
        "corr_l1": float(corr_l1),
        "n_columns_scored": float(len(per_column)),
    }
    return metrics, per_column


def _total_variation(real: pd.Series, syn: pd.Series) -> float:
    r = real.astype("string").fillna("__NA__").value_counts(normalize=True)
    s = syn.astype("string").fillna("__NA__").value_counts(normalize=True)
    keys = sorted(set(r.index) | set(s.index))
    return 0.5 * float(sum(abs(r.get(k, 0.0) - s.get(k, 0.0)) for k in keys))


def _resolve_eval_backend(eval_backend: EvalBackend) -> str:
    if eval_backend == "auto":
        return "sdmetrics" if sdmetrics_available() else "builtin"
    if eval_backend not in {"builtin", "sdmetrics"}:
        raise ValidationError(f"Unknown eval_backend: {eval_backend!r}")
    if eval_backend == "sdmetrics" and not sdmetrics_available():
        from buildml.core.errors import MissingExtraError

        raise MissingExtraError(
            "synthetic-industry",
            "eval_backend='sdmetrics'",
        )
    return eval_backend


def _plan_sdv_metadata(plan: SynthesizerPlan) -> Any | None:
    generator = plan.generator_
    if generator is not None and hasattr(generator, "metadata"):
        return generator.metadata
    return None


def _infer_task(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 20:
        return "regression"
    return "classification"


def _tstr_metrics(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    target: str,
    task: str,
    estimator: str,
    random_state: int,
) -> tuple[dict[str, float], list[str]]:
    warnings: list[str] = []
    x_train = train_frame[feature_cols]
    y_train = train_frame[target]
    x_test = test_frame[feature_cols]
    y_test = test_frame[target]

    numeric = [
        c
        for c in feature_cols
        if pd.api.types.is_numeric_dtype(x_train[c])
    ]
    categorical = [c for c in feature_cols if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValidationError("No usable feature columns for TSTR.")

    pre = ColumnTransformer(transformers)
    if task == "regression":
        model = Ridge(random_state=random_state)
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(x_train, pd.to_numeric(y_train, errors="coerce"))
        pred = pipe.predict(x_test)
        y_true = pd.to_numeric(y_test, errors="coerce").to_numpy(dtype=float)
        score = float(r2_score(y_true, pred))
        return {"score": score, "r2": score, "task_is_classification": 0.0}, warnings

    # classification
    n_classes = int(pd.Series(y_train).nunique(dropna=True))
    if estimator == "ridge" or (estimator == "auto" and n_classes > 10):
        clf: Any = RidgeClassifier(random_state=random_state)
    else:
        clf = LogisticRegression(max_iter=400, random_state=random_state)
    pipe = Pipeline([("pre", pre), ("model", clf)])
    pipe.fit(x_train, y_train.astype(str))
    pred = pipe.predict(x_test)
    y_true = y_test.astype(str).to_numpy()
    acc = float(accuracy_score(y_true, pred))
    try:
        f1 = float(f1_score(y_true, pred, average="macro"))
    except Exception:
        f1 = float("nan")
        warnings.append("macro-F1 unavailable for this label set.")
    return {
        "score": acc,
        "accuracy": acc,
        "macro_f1": f1,
        "task_is_classification": 1.0,
    }, warnings
