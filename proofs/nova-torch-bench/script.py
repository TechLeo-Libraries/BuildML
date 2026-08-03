"""Tier B product: Nova Torch Bench.

Composes torch tabular MLP + classical supervised baseline + probabilistic
intervals / calibration. Torch stage skips if unavailable.
"""

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
from buildml.core.errors import MissingExtraError
from proofs._lib import (
    TORCH_STATUS,
    assert_no_test_in_selection,
    load_mortgage_default_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)


FEATURE_NUM = ["ltv", "dti", "credit_score", "note_rate", "term_years"]
FEATURE_CAT = ["property_type"]
TARGET = "defaulted"


def main() -> None:
    ctx = new_proof_context("nova-torch-bench", seed=50)
    frame, data_meta = load_mortgage_default_synthetic(n=1400, seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []

    session = (
        Session.ingest(frame.copy())
        .set_roles(
            {**{c: "feature" for c in FEATURE_NUM + FEATURE_CAT}, TARGET: "target"}
        )
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    split_counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }
    session.impute(strategy="median")
    session.encode(method="onehot")

    # --- Stage 1: torch tabular ---
    if TORCH_STATUS.get("skip_torch_paths", True):
        stages["torch"] = {
            "status": "skipped",
            "error": TORCH_STATUS.get("error") or "torch smoke failed / unavailable",
            "torch": TORCH_STATUS,
        }
        skip_notes.append("torch: unavailable")
    else:
        try:
            session.make_torch_loaders(
                batch_size=64,
                normalize=True,
                seed=ctx.seed,
                task="classification",
            )
            session.fit_torch(
                epochs=3, learning_rate=1e-2, device="cpu", hidden=(64, 32)
            )
            t_val = session.evaluate_torch(partition="validation")
            assert_no_test_in_selection(
                selection_partition="validation", evaluation_partition="test"
            )
            t_test = session.evaluate_torch(partition="test")
            stages["torch"] = {
                "status": "ok",
                "epochs": 3,
                "validation_metrics": metrics_round(
                    dict(getattr(t_val, "metrics", {}) or {})
                ),
                "test_metrics": metrics_round(
                    dict(getattr(t_test, "metrics", {}) or {})
                ),
            }
        except (MissingExtraError, ImportError, TypeError, ValueError) as exc:
            stages["torch"] = {
                "status": "skipped",
                "error": f"{type(exc).__name__}: {exc}",
            }
            skip_notes.append(f"torch: {exc}")
    write_results(ctx, stages["torch"], filename="torch.json")

    # --- Stage 2: classical baseline ---
    c_session = (
        Session.ingest(frame.copy())
        .set_roles(
            {**{c: "feature" for c in FEATURE_NUM + FEATURE_CAT}, TARGET: "target"}
        )
        .inject_split(
            train_indices=list(plan.train_indices),
            validation_indices=list(plan.validation_indices),
            test_indices=list(plan.test_indices),
        )
    )
    c_session.impute(strategy="median")
    c_session.encode(method="onehot")
    c_session.scale(method="standard")
    c_session.fit(
        LogisticRegression(max_iter=1000, random_state=ctx.seed),
        task="classification",
    )
    c_val = c_session.evaluate(partition="validation")
    c_test = c_session.evaluate(partition="test")
    stages["supervised"] = {
        "status": "ok",
        "estimator": "LogisticRegression",
        "validation_metrics": metrics_round(dict(c_val.metrics)),
        "test_metrics": metrics_round(dict(c_test.metrics)),
    }
    write_results(ctx, stages["supervised"], filename="supervised.json")

    # --- Stage 3: probabilistic / calibration on continuous risk proxy ---
    try:
        # Use note_rate regression as a calibrated uncertainty bench on train split.
        hist = frame.loc[list(plan.train_indices)].copy().reset_index(drop=True)
        prob_session = (
            Session.ingest(hist)
            .set_roles(
                {
                    "ltv": "feature",
                    "dti": "feature",
                    "credit_score": "feature",
                    "term_years": "feature",
                    "note_rate": "target",
                    "property_type": "ignore",
                    "defaulted": "ignore",
                }
            )
            .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
        )
        prob_session.impute(strategy="median")
        prob_session.scale(method="standard")
        p_fit = prob_session.fit_probabilistic(
            estimator="bayesian_ridge",
            conformal=True,
            interval_method="both",
            random_state=ctx.seed,
        )
        try:
            intervals = prob_session.predict_interval(partition="test", alpha=0.1)
            interval_payload = metrics_round(
                intervals.to_dict() if hasattr(intervals, "to_dict") else {}
            )
        except Exception as exc:  # noqa: BLE001
            interval_payload = {"error": f"{type(exc).__name__}: {exc}"}
        p_ev = prob_session.evaluate_probabilistic(partition="test")
        stages["probabilistic"] = {
            "status": "ok",
            "fit": metrics_round(p_fit.to_dict() if hasattr(p_fit, "to_dict") else {}),
            "intervals": interval_payload,
            "test_metrics": metrics_round(dict(getattr(p_ev, "metrics", {}) or {})),
            "disclosure": (
                "Probabilistic session built from classical train rows only; "
                "internal split for interval calibration — no outer-test leakage."
            ),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["probabilistic"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"probabilistic: {exc}")
    write_results(ctx, stages["probabilistic"], filename="probabilistic.json")

    summary = {
        "status": "completed",
        "product": "Nova Torch Bench",
        "data": data_meta,
        "split": {"kind": plan.kind, "counts": split_counts, "stratify": True},
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "torch": TORCH_STATUS,
        "leakage_controls": [
            "Stratified split before impute/encode/loaders",
            "Torch normalize stats from train loader only",
            "Classical baseline uses the same inject_split",
            "Probabilistic intervals calibrated on train-derived internal split",
        ],
        "what_fails_if_leakage_ignored": [
            "Torch normalize stats from the full table leak holdout scale",
            "Early-stopping on test epochs cherry-picks the MLP",
            "Interval calibration on outer test reports perfect coverage by construction",
        ],
        "limitations": [
            "Synthetic mortgage labels; 3-epoch MLP smoke only",
            "Torch stage skipped when torch unavailable",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "nova-torch-bench OK",
        {
            "torch": stages["torch"]["status"],
            "supervised_roc": stages["supervised"]["test_metrics"].get("roc_auc"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
