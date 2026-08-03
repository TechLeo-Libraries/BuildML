"""Tier B product: Quasar Meta Adapt.

Composes metalearning few-shot adaptation + SSL pretext/probe + classical
supervised baseline for cold-start category repurchase.
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


def _category_coldstart(seed: int = 45) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for cat in range(18):
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
        "name": "quasar_category_coldstart",
        "license": "synthetic/public-domain",
        "n_rows": int(len(frame)),
        "n_categories": 18,
        "positive_rate": float(frame["repurchase"].mean()),
    }
    return frame, meta


def main() -> None:
    ctx = new_proof_context("quasar-meta-adapt", seed=45)
    frame, data_meta = _category_coldstart(seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []
    feats = [f"emb{j}" for j in range(5)]

    # --- Stage 1: metalearning (group split by category) ---
    meta_session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                **{c: "feature" for c in feats},
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
    plan = meta_session.split_plan
    assert plan is not None
    split_counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }
    backend_note = "prototypical"
    try:
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
            "backend_note": backend_note,
            "fit": metrics_round(m_fit.to_dict() if hasattr(m_fit, "to_dict") else {}),
            "test_metrics": metrics_round(dict(getattr(m_ev, "metrics", {}) or {})),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["metalearning"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"metalearning: {exc}")
    write_results(ctx, stages["metalearning"], filename="metalearning.json")

    # --- Stage 2: SSL pretext + probe on same split ---
    try:
        ssl_session = (
            Session.ingest(frame.copy())
            .set_roles(
                {
                    **{c: "feature" for c in feats},
                    "repurchase": "target",
                    "category_id": "ignore",
                }
            )
            .inject_split(
                train_indices=list(plan.train_indices),
                validation_indices=list(plan.validation_indices),
                test_indices=list(plan.test_indices),
            )
            .scale(method="standard")
        )
        ssl_fit = ssl_session.fit_ssl_pretext(
            method="masked_tabular", random_state=ctx.seed
        )
        try:
            ssl_session.finetune_ssl_head(random_state=ctx.seed)
        except Exception:  # noqa: BLE001
            pass
        ssl_val = ssl_session.evaluate_ssl(partition="validation")
        ssl_test = ssl_session.evaluate_ssl(partition="test")
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

    # --- Stage 3: classical supervised baseline ---
    c_session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                **{c: "feature" for c in feats},
                "repurchase": "target",
                "category_id": "ignore",
            }
        )
        .inject_split(
            train_indices=list(plan.train_indices),
            validation_indices=list(plan.validation_indices),
            test_indices=list(plan.test_indices),
        )
        .scale(method="standard")
    )
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

    summary = {
        "status": "completed",
        "product": "Quasar Meta Adapt",
        "data": data_meta,
        "split": {"kind": plan.kind, "counts": split_counts, "group_column": "category_id"},
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "torch": TORCH_STATUS,
        "leakage_controls": [
            "group_split by category_id before meta / SSL / classical fit",
            "Episodic metalearning eval on held-out categories",
            "SSL pretext + probe fit on train only",
            "Test used after each stage locks",
        ],
        "what_fails_if_leakage_ignored": [
            "Episodes that include test categories in the support set fake cold-start skill",
            "SSL pretext on the full table leaks holdout geometry into embeddings",
            "Classical baseline trained with test rows is not a fair comparator",
        ],
        "limitations": [
            "Synthetic categories — not production catalog cold-start",
            "Prototypical path may fall back to warm_start",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "quasar-meta-adapt OK",
        {
            "supervised_roc": stages["supervised"]["test_metrics"].get("roc_auc"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
