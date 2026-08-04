"""Tier A proof: observational fairness on Adult / credit-g / disclosed proxy."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from sklearn.linear_model import LogisticRegression

from buildml import Session
from proofs._lib import (
    load_fairness_public_dataset,
    metrics_round,
    new_proof_context,
    refuse_perfect_scores,
    write_results,
)


def _prepare_numeric_frame(frame, feature_cols, target, sensitive):
    """Code-encode categoricals for core-install Session classical fit."""
    out = frame[[*feature_cols, sensitive, target]].copy()
    for col in feature_cols:
        if out[col].dtype == object or str(out[col].dtype) == "category":
            out[col] = out[col].astype("category").cat.codes.astype(float)
        else:
            out[col] = out[col].astype(float)
    out[sensitive] = out[sensitive].astype(str)
    out[target] = out[target].astype(int)
    return out


def main() -> None:
    ctx = new_proof_context("adult-fairness-observational", seed=42)
    frame_raw, data_meta = load_fairness_public_dataset()
    target = str(data_meta["target"])
    sensitive = str(data_meta["sensitive_column"])
    features = list(data_meta["feature_columns"])

    # Cap Adult size for CI time while keeping provenance of the full source.
    max_rows = 2500
    if len(frame_raw) > max_rows:
        frame_raw = frame_raw.sample(n=max_rows, random_state=ctx.seed).reset_index(
            drop=True
        )
        data_meta = dict(data_meta)
        data_meta["n_rows_used"] = int(max_rows)
        data_meta["n_rows_source"] = int(data_meta.get("n_rows", max_rows))
        data_meta["n_rows"] = int(max_rows)
        data_meta["subsample"] = {
            "max_rows": max_rows,
            "random_state": ctx.seed,
            "reason": "CI runtime cap",
        }

    frame = _prepare_numeric_frame(frame_raw, features, target, sensitive)

    session = (
        Session.ingest(frame)
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
        .impute(strategy="median")
        .scale(method="standard")
        .fit(LogisticRegression(max_iter=1200, random_state=ctx.seed), task="classification")
    )
    plan = session.split_plan
    assert plan is not None
    test_eval = session.evaluate(partition="test")
    test_metrics = metrics_round(dict(test_eval.metrics))
    refuse_perfect_scores(
        test_metrics,
        keys=("accuracy", "f1", "f1_weighted", "f1_macro", "roc_auc"),
        ceiling=1.0,
        proof_slug="adult-fairness-observational",
        context="fairness public-dataset holdout",
    )

    report = session.fairness.evaluate(
        sensitive_column=sensitive,
        partition="test",
        positive_label=1,
    )
    payload = report.to_dict()
    write_results(
        ctx,
        {
            "status": "completed",
            "evidence_tier": "REAL_PUBLIC_DATASET",
            "data": data_meta,
            "split": {
                "protocol": "stratified_train_validation_test",
                "test_size": 0.25,
                "validation_size": 0.2,
                "stratify": True,
                "random_state": ctx.seed,
                "counts": {
                    "train": len(plan.train_indices),
                    "validation": len(plan.validation_indices),
                    "test": len(plan.test_indices),
                },
            },
            "sensitive_column": sensitive,
            "positive_label": 1,
            "test_metrics": test_metrics,
            "fairness": {
                "n_rows": report.n_rows,
                "groups": list(report.groups),
                "selection_rate_by_group": metrics_round(
                    dict(report.selection_rate_by_group)
                ),
                "demographic_parity_difference": float(
                    report.demographic_parity_difference
                ),
                "disparate_impact_ratio": (
                    None
                    if report.disparate_impact_ratio is None
                    else float(report.disparate_impact_ratio)
                ),
                "equalized_odds_tpr_difference": report.equalized_odds_tpr_difference,
                "equalized_odds_fpr_difference": report.equalized_odds_fpr_difference,
                "disclosures": list(report.disclosures),
                "warnings": list(report.warnings),
            },
            "capability_matrix": session.fairness.capability_matrix(),
            "leakage_controls": [
                "Classifier fitted on train only",
                "Fairness metrics on holdout test predictions only",
                "Sensitive column role=ignore (not a model feature)",
            ],
            "honesty": [
                "Observational disparity report — not a legal audit.",
                "Prefers OpenML Adult, then credit-g; offline CI may use "
                "disclosed breast-cancer radius proxy (see data.proxy_disclosure).",
                "Categorical OpenML features are code-encoded before Session fit "
                "for core-install viability.",
            ],
            "limitations": [
                "Not a legal / regulatory fairness certification",
                "No bias mitigation applied",
                "Adult subsampled to 2500 rows when larger for CI time",
            ],
            "raw_report_keys": sorted(payload.keys()),
        },
    )
    print(
        "adult-fairness-observational OK",
        {
            "loader": data_meta.get("loader_selected"),
            "dp": report.demographic_parity_difference,
            "di": report.disparate_impact_ratio,
            "groups": report.groups,
            "test": test_metrics,
        },
    )


if __name__ == "__main__":
    main()
