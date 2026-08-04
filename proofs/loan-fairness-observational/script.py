"""Tier A proof: observational fairness gaps on a synthetic credit holdout."""

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
    load_credit_approval_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)


def main() -> None:
    ctx = new_proof_context("loan-fairness-observational", seed=42)
    frame, data_meta = load_credit_approval_synthetic(n=900, seed=ctx.seed)
    # region is caller-declared sensitive attribute (not inferred by BuildML).
    features = ["age", "income", "debt_ratio", "employment_years"]
    session = (
        Session.ingest(frame)
        .set_roles(
            {
                **{c: "feature" for c in features},
                "region": "ignore",
                "product": "ignore",
                "approved": "target",
            }
        )
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=ctx.seed)
        .impute(strategy="median")
        .scale(method="standard")
        .fit(LogisticRegression(max_iter=800), task="classification")
    )
    report = session.fairness.evaluate(
        sensitive_column="region",
        partition="test",
        positive_label=1,
    )
    payload = report.to_dict()
    write_results(
        ctx,
        {
            "status": "completed",
            "data": data_meta,
            "sensitive_column": "region",
            "positive_label": 1,
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
                "Fairness metrics computed on holdout test predictions only",
            ],
            "honesty": [
                "Observational disparity report — not a legal audit.",
                "Sensitive groups are caller-declared (region).",
            ],
            "limitations": [
                "Synthetic credit table; region is a stand-in sensitive attribute",
                "No bias mitigation / reweighing applied",
            ],
            "raw_report_keys": sorted(payload.keys()),
        },
    )
    print(
        "loan-fairness-observational OK",
        {
            "dp": report.demographic_parity_difference,
            "di": report.disparate_impact_ratio,
            "groups": report.groups,
        },
    )


if __name__ == "__main__":
    main()
