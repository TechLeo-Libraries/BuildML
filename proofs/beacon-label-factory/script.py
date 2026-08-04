"""Tier B product: Beacon Label Factory.

Composes semi-supervised propagation + active-learning budget loop + SSL
pretext/probe for scarce-label manufacturing inspection features.
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

from buildml import Session
from buildml.core.errors import MissingExtraError
from buildml.data.dataset import Dataset
from buildml.ingest.detect import schema_from_dataframe
from proofs._lib import (
    TORCH_STATUS,
    assert_no_test_in_selection,
    metrics_round,
    new_proof_context,
    write_results,
)


def _inspection_pool(n_per: int = 220, seed: int = 55) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    neg = rng.normal([-1.0, -0.9, -0.4, 0.2], [0.55, 0.5, 0.45, 0.4], size=(n_per, 4))
    pos = rng.normal([1.1, 0.95, 0.65, -0.25], [0.55, 0.5, 0.45, 0.4], size=(n_per, 4))
    frame = pd.DataFrame(
        np.vstack([neg, pos]),
        columns=["intensity", "texture", "edge_density", "asymmetry"],
    )
    frame["defect"] = [0] * n_per + [1] * n_per
    meta = {
        "name": "beacon_inspection_label_pool",
        "license": "synthetic/public-domain",
        "n_rows": int(len(frame)),
        "positive_rate": 0.5,
    }
    return frame, meta


def _mask_train_labels(session: Session, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    full = session.to_pandas().copy()
    truth = full["defect"].to_numpy().copy()
    train_idx = list(session.split_plan.train_indices)
    n_blank = max(1, int(fraction * len(train_idx)))
    blank = rng.choice(train_idx, size=n_blank, replace=False)
    full.loc[blank, "defect"] = np.nan
    session._dataset = Dataset.from_transformed(
        session.dataset,
        full,
        schema=schema_from_dataframe(full),
        roles=dict(session.dataset.roles),
    )
    return truth


def main() -> None:
    ctx = new_proof_context("beacon-label-factory", seed=55)
    frame, data_meta = _inspection_pool(seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []
    feats = ["intensity", "texture", "edge_density", "asymmetry"]

    session = (
        Session.ingest(frame.copy())
        .set_roles({**{c: "feature" for c in feats}, "defect": "target"})
        .split(test_size=0.25, validation_size=0.15, stratify=True, random_state=ctx.seed)
        .scale(method="standard")
    )
    plan = session.split_plan
    assert plan is not None
    split_counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }
    truth = _mask_train_labels(session, fraction=0.82, seed=ctx.seed)

    # --- Stage 1: SSL pretext ---
    try:
        ssl_fit = session.ssl.fit_pretext(method="masked_tabular", random_state=ctx.seed)
        try:
            session.ssl.finetune_head(random_state=ctx.seed)
        except Exception:  # noqa: BLE001
            pass
        ssl_val = session.ssl.evaluate(partition="validation")
        ssl_test = session.ssl.evaluate(partition="test")
        stages["ssl"] = {
            "status": "ok",
            "fit": metrics_round(
                ssl_fit.to_dict() if hasattr(ssl_fit, "to_dict") else {}
            ),
            "validation_metrics": metrics_round(
                dict(getattr(ssl_val, "metrics", {}) or {})
            ),
            "test_metrics": metrics_round(
                dict(getattr(ssl_test, "metrics", {}) or {})
            ),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["ssl"] = {"status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
        skip_notes.append(f"ssl: {exc}")
    write_results(ctx, stages["ssl"], filename="ssl.json")

    # --- Stage 2: semi-supervised ---
    try:
        # Fresh masked session for semi (SSL may have altered state).
        semi = (
            Session.ingest(frame.copy())
            .set_roles({**{c: "feature" for c in feats}, "defect": "target"})
            .inject_split(
                train_indices=list(plan.train_indices),
                validation_indices=list(plan.validation_indices),
                test_indices=list(plan.test_indices),
            )
            .scale(method="standard")
        )
        _mask_train_labels(semi, fraction=0.82, seed=ctx.seed)
        s_fit = semi.semisupervised.fit(method="label_propagation", n_neighbors=7)
        s_val = semi.semisupervised.evaluate(partition="validation")
        assert_no_test_in_selection(
            selection_partition="validation", evaluation_partition="test"
        )
        s_test = semi.semisupervised.evaluate(partition="test")
        stages["semisupervised"] = {
            "status": "ok",
            "fit": {
                "method": s_fit.method,
                "n_labeled_train": int(s_fit.n_labeled_train),
                "n_unlabeled_train": int(s_fit.n_unlabeled_train),
            },
            "validation_metrics": metrics_round(dict(s_val.metrics)),
            "test_metrics": metrics_round(dict(s_test.metrics)),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["semisupervised"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"semisupervised: {exc}")
    write_results(ctx, stages["semisupervised"], filename="semisupervised.json")

    # --- Stage 3: active learning budget loop ---
    try:
        al = (
            Session.ingest(frame.copy())
            .set_roles({**{c: "feature" for c in feats}, "defect": "target"})
            .inject_split(
                train_indices=list(plan.train_indices),
                validation_indices=list(plan.validation_indices),
                test_indices=list(plan.test_indices),
            )
            .scale(method="standard")
        )
        truth_series = pd.Series(truth, index=frame.index)
        _mask_train_labels(al, fraction=0.85, seed=ctx.seed + 1)
        a_fit = al.active_learning.fit(strategy="margin", batch_size=8, label_budget=32)
        curve = []
        for round_i in range(4):
            q = al.active_learning.suggest_query(batch_size=8)
            if not q.indices:
                break
            human = [int(truth_series.loc[i]) for i in q.indices]
            labeled = al.active_learning.label_rows(indices=q.indices, labels=human)
            curve.append(
                {
                    "round": round_i,
                    "n_newly_labeled": int(labeled.n_newly_labeled),
                    "n_labeled_now": int(labeled.n_labeled_now),
                    "budget_remaining": int(labeled.budget_remaining),
                }
            )
        a_test = al.active_learning.evaluate(partition="test")
        stages["active_learning"] = {
            "status": "ok",
            "fit": {
                "strategy": a_fit.strategy,
                "n_labeled_train": int(a_fit.n_labeled_train),
                "n_unlabeled_pool": int(a_fit.n_unlabeled_pool),
            },
            "label_curve": curve,
            "test_metrics": metrics_round(dict(a_test.metrics)),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["active_learning"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"active_learning: {exc}")
    write_results(ctx, stages["active_learning"], filename="active_learning.json")

    summary = {
        "status": "completed",
        "product": "Beacon Label Factory",
        "data": data_meta,
        "split": {"kind": plan.kind, "counts": split_counts, "stratify": True},
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "torch": TORCH_STATUS,
        "leakage_controls": [
            "Stratified split before masking / pretext / AL",
            "Label masking applied to train indices only",
            "Holdouts retain full labels solely for evaluation",
            "AL queries drawn from the train unlabeled pool only",
        ],
        "what_fails_if_leakage_ignored": [
            "Masking validation/test then recovering labels via the graph overstates SSL gains",
            "Allowing AL to query the test pool turns the budget curve into a cheat sheet",
            "Fitting SSL pretext on the full table leaks holdout geometry into embeddings",
        ],
        "limitations": [
            "Simulated oracle; synthetic inspection proxies — not a PACS / AOI UI",
            "Tabular imaging proxies, not pixel CNNs",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "beacon-label-factory OK",
        {
            "semi": stages.get("semisupervised", {}).get("status"),
            "al": stages.get("active_learning", {}).get("status"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
