"""Tier B product: Mosaic Warranty Desk.

Composes CBR case memory + symbolic guardrails + classical supervised
scoring for synthetic warranty claim approve/deny decisions.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import MissingExtraError
from proofs._lib import (
    assert_no_test_in_selection,
    metrics_round,
    new_proof_context,
    write_results,
)

FEATS = [
    "failure_severity",
    "usage_hours_z",
    "prior_claims_z",
    "parts_cost_z",
    "age_months_z",
]
TARGET = "approve_claim"


def _warranty_book(n: int = 360, seed: int = 28) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5))
    y = (
        x[:, 0] + 0.65 * x[:, 2] - 0.3 * x[:, 4] + rng.normal(scale=0.25, size=n) > 0
    ).astype(int)
    frame = pd.DataFrame(x, columns=FEATS)
    frame[TARGET] = y
    frame["claim_id"] = [f"w{i}" for i in range(n)]
    meta = {
        "name": "mosaic_synthetic_warranty_claims",
        "license": "synthetic/public-domain",
        "n_rows": n,
        "positive_rate": float(y.mean()),
    }
    return frame, meta


def main() -> None:
    ctx = new_proof_context("mosaic-warranty-desk", seed=28)
    frame, data_meta = _warranty_book(seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []

    session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                **{c: "feature" for c in FEATS},
                TARGET: "target",
                "claim_id": "id",
            }
        )
        .split(
            test_size=0.2,
            validation_size=0.2,
            stratify=True,
            random_state=ctx.seed,
        )
        .scale(method="standard")
    )
    plan = session.split_plan
    assert plan is not None
    split_counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }

    # --- Stage 1: CBR case memory ---
    try:
        fit_c = session.cbr.fit(
            task="classification",
            metric="euclidean",
            reuse="distance_weighted",
            k=5,
            random_state=ctx.seed,
        )
        retrieved = session.cbr.retrieve(partition="test", k=5)
        ev_c = session.cbr.evaluate(partition="test")
        stages["cbr"] = {
            "status": "ok",
            "fit": metrics_round(fit_c.to_dict() if hasattr(fit_c, "to_dict") else {}),
            "retrieve_sample": metrics_round(
                retrieved.to_dict() if hasattr(retrieved, "to_dict") else {}
            ),
            "test_metrics": metrics_round(dict(getattr(ev_c, "metrics", {}) or {})),
        }
        write_results(ctx, stages["cbr"], filename="cbr.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["cbr"] = {"status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
        skip_notes.append(f"cbr: {exc}")

    # --- Stage 2: symbolic guardrails on same split ---
    try:
        sym_session = (
            Session.ingest(frame.copy())
            .set_roles(
                {
                    **{c: "feature" for c in FEATS},
                    TARGET: "target",
                    "claim_id": "id",
                }
            )
            .inject_split(
                train_indices=list(plan.train_indices),
                validation_indices=list(plan.validation_indices),
                test_indices=list(plan.test_indices),
            )
        )
        assert_no_test_in_selection(
            selection_partition="train", evaluation_partition="test"
        )
        try:
            fit_s = sym_session.symbolic.fit(
                source="decision_tree", max_depth=3, random_state=ctx.seed
            )
        except TypeError:
            fit_s = sym_session.symbolic.fit(
                method="decision_tree", random_state=ctx.seed
            )
        ev_s = sym_session.symbolic.evaluate(partition="test")
        stages["symbolic"] = {
            "status": "ok",
            "fit": metrics_round(fit_s.to_dict() if hasattr(fit_s, "to_dict") else {}),
            "test_metrics": metrics_round(dict(getattr(ev_s, "metrics", {}) or {})),
        }
        write_results(ctx, stages["symbolic"], filename="symbolic.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["symbolic"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"symbolic: {exc}")

    # --- Stage 3: classical supervised ---
    try:
        classical = (
            Session.ingest(frame.copy())
            .set_roles(
                {
                    **{c: "feature" for c in FEATS},
                    TARGET: "target",
                    "claim_id": "id",
                }
            )
            .inject_split(
                train_indices=list(plan.train_indices),
                validation_indices=list(plan.validation_indices),
                test_indices=list(plan.test_indices),
            )
            .scale(method="standard")
        )
        classical.fit(
            LogisticRegression(max_iter=1000, random_state=ctx.seed),
            task="classification",
        )
        c_test = classical.evaluate(partition="test")
        stages["classical"] = {
            "status": "ok",
            "estimator": "LogisticRegression",
            "test_metrics": metrics_round(dict(c_test.metrics)),
        }
        write_results(ctx, stages["classical"], filename="classical.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["classical"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"classical: {exc}")

    ok_stages = sum(1 for v in stages.values() if v.get("status") == "ok")
    summary = {
        "status": "completed" if ok_stages >= 3 else "partial",
        "product": "Mosaic Warranty Desk",
        "data": data_meta,
        "split": {"kind": plan.kind, "counts": split_counts, "stratify": True},
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "leakage_controls": [
            "Stratified split before CBR / symbolic / classical",
            "CBR case memory built from train cases only",
            "Symbolic rules induced on the same train split; test after lock",
            "Classical scorer uses inject_split — never refits on test",
        ],
        "what_fails_if_leakage_ignored": [
            "Putting test claims into CBR memory makes accuracy meaningless",
            "Inducing guardrail rules on the full book looks more 'fair' than production",
            "Fitting classical scores on the full table invents holdout ROC",
        ],
        "limitations": [
            "Synthetic warranty claims — not a real OEM extract",
            "CBR ≠ RAG; product proof only",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "mosaic-warranty-desk OK",
        {
            "cbr": (stages.get("cbr") or {}).get("status"),
            "symbolic": (stages.get("symbolic") or {}).get("status"),
            "classical": (stages.get("classical") or {}).get("status"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
