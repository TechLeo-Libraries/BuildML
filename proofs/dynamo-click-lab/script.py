"""Tier B product: Dynamo Click Lab.

Composes online stream conversion scoring + metalearning cold-start across
categories + classical supervised baseline for synthetic clickstream.
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
    TORCH_STATUS,
    metrics_round,
    new_proof_context,
    write_results,
)


def _clickstream(n_per: int = 230, seed: int = 24) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    bounce = rng.normal([-1.0, -0.8, 0.2], 0.55, size=(n_per, 3))
    convert = rng.normal([1.1, 0.9, -0.2], 0.55, size=(n_per, 3))
    frame = pd.DataFrame(
        np.vstack([bounce, convert]),
        columns=["pages_z", "dwell_z", "cart_adds_z"],
    )
    frame["converted"] = [0] * n_per + [1] * n_per
    meta = {
        "name": "dynamo_synthetic_clickstream",
        "license": "synthetic/public-domain",
        "n_rows": int(len(frame)),
        "positive_rate": float(frame["converted"].mean()),
    }
    return frame, meta


def _category_coldstart(seed: int = 26) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for cat in range(16):
        center = rng.normal(size=5) * (0.6 + cat * 0.025)
        for _ in range(28):
            x = center + rng.normal(scale=0.35, size=5)
            y = int((x[0] + 0.35 * x[2] - 0.2 * x[4]) > 0)
            rows.append(
                {
                    **{f"emb{j}": float(x[j]) for j in range(5)},
                    "repurchase": y,
                    "category_id": f"cat{cat}",
                }
            )
    frame = pd.DataFrame(rows)
    meta = {
        "name": "dynamo_synthetic_category_coldstart",
        "license": "synthetic/public-domain",
        "n_rows": int(len(frame)),
        "n_categories": 16,
    }
    return frame, meta


def main() -> None:
    ctx = new_proof_context("dynamo-click-lab", seed=24)
    clicks, click_meta = _clickstream(seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []
    feats = ["pages_z", "dwell_z", "cart_adds_z"]

    session = (
        Session.ingest(clicks.copy())
        .set_roles({**{c: "feature" for c in feats}, "converted": "target"})
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

    # --- Stage 1: online ---
    try:
        online = (
            Session.ingest(clicks.copy())
            .set_roles({**{c: "feature" for c in feats}, "converted": "target"})
            .inject_split(
                train_indices=list(plan.train_indices),
                validation_indices=list(plan.validation_indices),
                test_indices=list(plan.test_indices),
            )
            .scale(method="standard")
        )
        o_fit = online.fit_online(
            estimator="sgd_classifier",
            chunk_size=50,
            n_init=50,
            classes=[0, 1],
        )
        updates = 0
        while True:
            remaining = online.online_plan.n_train_rows - online.online_plan.cursor
            if remaining <= 0:
                break
            online.partial_fit_online(n_rows=min(50, remaining))
            updates += 1
        o_test = online.evaluate_online(partition="test")
        stages["online"] = {
            "status": "ok",
            "data": click_meta,
            "n_init_rows": int(o_fit.n_init_rows),
            "n_updates": updates,
            "test_metrics": metrics_round(dict(o_test.metrics)),
        }
        write_results(ctx, stages["online"], filename="online.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["online"] = {"status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
        skip_notes.append(f"online: {exc}")

    # --- Stage 2: metalearning cold-start ---
    meta_frame, meta_meta = _category_coldstart(seed=ctx.seed + 2)
    try:
        meta_session = (
            Session.ingest(meta_frame)
            .set_roles(
                {
                    **{f"emb{j}": "feature" for j in range(5)},
                    "repurchase": "target",
                    "category_id": "group",
                }
            )
            .group_split(
                test_size=0.25,
                validation_size=0.15,
                random_state=ctx.seed,
                group_column="category_id",
            )
        )
        backend_note = "prototypical"
        try:
            m_fit = meta_session.fit_metalearning(
                method="prototypical",
                task_column="category_id",
                n_way=2,
                k_shot=5,
                n_query=5,
                random_state=ctx.seed,
            )
        except (MissingExtraError, TypeError, ValueError) as exc:
            m_fit = meta_session.fit_metalearning(
                method="warm_start",
                task_column="category_id",
                n_way=2,
                k_shot=5,
                n_query=5,
                random_state=ctx.seed,
            )
            backend_note = f"warm_start_fallback({type(exc).__name__})"
        m_ev = meta_session.evaluate_metalearning(partition="test")
        stages["metalearning"] = {
            "status": "ok",
            "data": meta_meta,
            "backend_note": backend_note,
            "torch": TORCH_STATUS,
            "fit": metrics_round(m_fit.to_dict() if hasattr(m_fit, "to_dict") else {}),
            "test_metrics": metrics_round(dict(getattr(m_ev, "metrics", {}) or {})),
        }
        write_results(ctx, stages["metalearning"], filename="metalearning.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["metalearning"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"metalearning: {exc}")

    # --- Stage 3: classical ---
    try:
        session.fit(
            LogisticRegression(max_iter=1000, random_state=ctx.seed),
            task="classification",
        )
        c_test = session.evaluate(partition="test")
        stages["classical"] = {
            "status": "ok",
            "estimator": "LogisticRegression",
            "test_metrics": metrics_round(dict(c_test.metrics)),
            "split_counts": split_counts,
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
        "product": "Dynamo Click Lab",
        "data": click_meta,
        "split": {"kind": plan.kind, "counts": split_counts, "stratify": True},
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "torch": TORCH_STATUS,
        "leakage_controls": [
            "Online partial_fit consumes train cursor only",
            "Metalearning group_split by category_id; episodic eval on held-out categories",
            "Classical scorer uses the same clickstream stratified split",
            "Test evaluate_online / evaluate_metalearning / evaluate after locks",
        ],
        "what_fails_if_leakage_ignored": [
            "Streaming updates that include test rows make online metrics meaningless",
            "Episodes that mix train and test categories invent cold-start accuracy",
            "Fitting classical scores on the full clickstream invents holdout ROC",
        ],
        "limitations": [
            "Batch chunks, not Kafka/Flink clickstream",
            "Synthetic categories; not production catalog cold-start",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "dynamo-click-lab OK",
        {
            "online": (stages.get("online") or {}).get("status"),
            "metalearning": (stages.get("metalearning") or {}).get("status"),
            "classical": (stages.get("classical") or {}).get("status"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
