"""Tier C: sklearn Pipeline + group-rate twin for adult-fairness-observational."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from buildml import Session
from proofs._lib import (
    extract_buildml_test_metrics,
    load_buildml_results,
    load_fairness_public_dataset,
    metrics_round,
    new_proof_context,
    write_comparison,
)


def _prepare_numeric_frame(frame, feature_cols, target, sensitive):
    """Code-encode categoricals to match the Tier A Session path."""
    out = frame[[*feature_cols, sensitive, target]].copy()
    for col in feature_cols:
        if out[col].dtype == object or str(out[col].dtype) == "category":
            out[col] = out[col].astype("category").cat.codes.astype(float)
        else:
            out[col] = out[col].astype(float)
    out[sensitive] = out[sensitive].astype(str)
    out[target] = out[target].astype(int)
    return out


def _selection_rates(pred, groups: pd.Series) -> dict[str, float]:
    rates: dict[str, float] = {}
    for group in sorted(pd.unique(groups)):
        mask = groups == group
        if int(mask.sum()) == 0:
            continue
        rates[str(group)] = float(pred[mask].mean())
    return rates


def _demographic_parity_difference(rates: dict[str, float]) -> float:
    values = list(rates.values())
    if len(values) < 2:
        return 0.0
    return float(max(values) - min(values))


def _disparate_impact_ratio(rates: dict[str, float]) -> float | None:
    values = list(rates.values())
    if len(values) < 2:
        return None
    high = max(values)
    if high == 0.0:
        return None
    return float(min(values) / high)


def main() -> None:
    ctx = new_proof_context("adult-fairness-observational", seed=42)
    frame_raw, data_meta = load_fairness_public_dataset()
    target = str(data_meta["target"])
    sensitive = str(data_meta["sensitive_column"])
    features = list(data_meta["feature_columns"])

    max_rows = 2500
    if len(frame_raw) > max_rows:
        frame_raw = frame_raw.sample(n=max_rows, random_state=ctx.seed).reset_index(
            drop=True
        )

    frame = _prepare_numeric_frame(frame_raw, features, target, sensitive)
    session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                **{c: "feature" for c in features},
                sensitive: "ignore",
                target: "target",
            }
        )
        .split(
            test_size=0.25,
            validation_size=0.2,
            stratify=True,
            random_state=ctx.seed,
        )
    )
    plan = session.split_plan
    assert plan is not None
    train_idx = list(plan.train_indices)
    val_idx = list(plan.validation_indices)
    test_idx = list(plan.test_indices)

    x_train = frame.loc[train_idx, features]
    y_train = frame.loc[train_idx, target]
    x_test = frame.loc[test_idx, features]
    y_test = frame.loc[test_idx, target]
    groups_test = frame.loc[test_idx, sensitive]

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1200, random_state=ctx.seed)),
        ]
    )
    pipe.fit(x_train, y_train)
    proba = pipe.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    rates = _selection_rates(pred, groups_test)
    di = _disparate_impact_ratio(rates)
    industry_metrics = metrics_round(
        {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred, average="weighted")),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "demographic_parity_difference": _demographic_parity_difference(rates),
            **({"disparate_impact_ratio": di} if di is not None else {}),
        }
    )

    bml_raw = load_buildml_results(ctx.project_dir)
    bml_metrics = extract_buildml_test_metrics(
        bml_raw,
        prefer=("test_metrics",),
        keys=("accuracy", "f1", "roc_auc", "f1_weighted"),
    )
    if "f1" not in bml_metrics and "f1_weighted" in bml_metrics:
        bml_metrics["f1"] = bml_metrics["f1_weighted"]
    fair = bml_raw.get("fairness") or {}
    if isinstance(fair, dict):
        if "demographic_parity_difference" in fair:
            bml_metrics["demographic_parity_difference"] = fair[
                "demographic_parity_difference"
            ]
        if fair.get("disparate_impact_ratio") is not None:
            bml_metrics["disparate_impact_ratio"] = fair["disparate_impact_ratio"]
    bml_metrics = metrics_round(bml_metrics)

    write_comparison(
        ctx,
        buildml={
            "backend": "buildml.Session + session.fairness.evaluate",
            "estimator": "LogisticRegression",
            "test_metrics": bml_metrics,
        },
        industry={
            "backend": "sklearn.Pipeline + group selection rates",
            "estimator": "LogisticRegression",
            "test_metrics": industry_metrics,
            "selection_rate_by_group": rates,
            "leakage_controls": [
                "Imputer + scaler + estimator fit on train indices only",
                "Sensitive column excluded from features",
                "Group rates computed on holdout predictions only",
                "Same SplitPlan indices as BuildML Session",
            ],
        },
        split_counts={
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        delta_keys=(
            "accuracy",
            "f1",
            "roc_auc",
            "demographic_parity_difference",
            "disparate_impact_ratio",
        ),
        extra={
            "evidence_tier": "REAL_PUBLIC_DATASET",
            "sensitive_column": sensitive,
            "loader_selected": data_meta.get("loader_selected"),
        },
    )
    print("adult-fairness-observational Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
