"""Tier B product: Aurora Ad Ranker.

Composes learning-to-rank over sponsored ads + classical CTR proxy +
validation-tuned capacity decisions for impression allocation.
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
    extra_available,
    load_ad_ltr_judgments_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)


def _ctr_frame(rank_frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Per query×ad CTR-style classical table (both classes, stratified-friendly)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, row in enumerate(rank_frame.itertuples(index=False)):
        logit = (
            -1.8
            + 0.55 * float(row.rel_feat)
            + 0.35 * float(row.bid)
            + 1.4 * float(row.ctr_prior)
            + 0.35 * float(row.relevance)
            + rng.normal(0, 0.35)
        )
        clicked = int(1 / (1 + np.exp(-logit)) > 0.5)
        rows.append(
            {
                "pair_id": f"p-{i}",
                "rel_feat": float(row.rel_feat),
                "bid": float(row.bid),
                "ctr_prior": float(row.ctr_prior),
                "clicked": clicked,
                "serve_cost": float(1.0 + 0.4 * float(row.bid)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ctx = new_proof_context("aurora-ad-ranker", seed=114)
    rank_frame, data_meta = load_ad_ltr_judgments_synthetic(seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []

    # --- Stage 1: ranking ---
    try:
        lgbm = extra_available("lightgbm")
        rank_session = (
            Session.ingest(rank_frame.copy())
            .set_roles(
                {
                    "query_id": "group",
                    "ad_id": "id",
                    "relevance": "target",
                    "rel_feat": "feature",
                    "bid": "feature",
                    "ctr_prior": "feature",
                }
            )
            .group_split(test_size=0.25, validation_size=0.15, random_state=ctx.seed)
        )
        method = "lambdarank" if lgbm else "pointwise"
        try:
            if lgbm:
                fit_r = rank_session.fit_ranker(
                    method="lambdarank",
                    query_column="query_id",
                    item_column="ad_id",
                    random_state=ctx.seed,
                )
            else:
                raise MissingExtraError("ranking-industry", "lambdarank")
        except (MissingExtraError, TypeError, ValueError):
            fit_r = rank_session.fit_ranker(
                method="pointwise",
                query_column="query_id",
                item_column="ad_id",
                pointwise_estimator="ridge",
                random_state=ctx.seed,
            )
            method = "pointwise"
        ranked = rank_session.rank(partition="test", k=5)
        ev_r = rank_session.evaluate_ranker(partition="test", k=5)
        plan_r = rank_session.split_plan
        assert plan_r is not None
        stages["ranking"] = {
            "status": "ok",
            "method": method,
            "lightgbm_available": lgbm,
            "fit": metrics_round(fit_r.to_dict() if hasattr(fit_r, "to_dict") else {}),
            "rank_sample": metrics_round(
                ranked.to_dict() if hasattr(ranked, "to_dict") else {}
            ),
            "test_metrics": metrics_round(dict(ev_r.metrics)),
            "split_counts": {
                "train": len(plan_r.train_indices),
                "validation": len(plan_r.validation_indices),
                "test": len(plan_r.test_indices),
            },
        }
        write_results(ctx, stages["ranking"], filename="ranking.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["ranking"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"ranking: {exc}")

    # --- Stage 2: classical CTR proxy ---
    ctr = _ctr_frame(rank_frame, seed=ctx.seed)
    try:
        classical = (
            Session.ingest(ctr)
            .set_roles(
                {
                    "pair_id": "id",
                    "rel_feat": "feature",
                    "bid": "feature",
                    "ctr_prior": "feature",
                    "clicked": "target",
                    "serve_cost": "ignore",
                }
            )
            .split(
                test_size=0.25,
                validation_size=0.2,
                stratify=True,
                random_state=ctx.seed,
            )
            .scale(method="standard")
        )
        classical.fit(
            LogisticRegression(max_iter=1000, random_state=ctx.seed),
            task="classification",
        )
        c_test = classical.evaluate(partition="test")
        plan_c = classical.split_plan
        assert plan_c is not None
        stages["classical"] = {
            "status": "ok",
            "estimator": "LogisticRegression",
            "data_n": int(len(ctr)),
            "test_metrics": metrics_round(dict(c_test.metrics)),
            "split_counts": {
                "train": len(plan_c.train_indices),
                "validation": len(plan_c.validation_indices),
                "test": len(plan_c.test_indices),
            },
        }
        write_results(ctx, stages["classical"], filename="classical.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["classical"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"classical: {exc}")
        classical = None
        plan_c = None

    # --- Stage 3: decisions / capacity allocation ---
    try:
        if classical is None or plan_c is None:
            raise ValueError("classical stage unavailable")
        assert_no_test_in_selection(
            selection_partition="validation", evaluation_partition="test"
        )
        thr = classical.fit_decision_policy(
            method="threshold",
            partition="validation",
            fp_cost=1.0,
            fn_cost=3.0,
        )
        thr_test = classical.evaluate_decisions(partition="test")
        alloc_payload: dict = {"alloc_status": "skipped"}
        try:
            knap = classical.fit_decision_policy(
                method="knapsack",
                partition="validation",
                budget=40.0,
                cost_column="serve_cost",
                id_column="pair_id",
                score_source="model_proba",
                knapsack_solver="dp",
            )
            applied = classical.apply_decisions(partition="test")
            alloc_payload = {
                "alloc_status": "ok",
                "knapsack_policy": metrics_round(
                    knap.to_dict() if hasattr(knap, "to_dict") else {}
                ),
                "applied": {
                    "n_selected": int(applied.n_selected),
                    "selected_value": float(applied.selected_value),
                    "selected_cost": float(applied.selected_cost),
                },
            }
        except Exception as exc:  # noqa: BLE001
            topk = classical.fit_decision_policy(
                method="topk",
                partition="validation",
                capacity=25,
                score_source="model_proba",
            )
            applied = classical.apply_decisions(partition="test")
            alloc_payload = {
                "alloc_status": "ok_topk_fallback",
                "error": f"{type(exc).__name__}: {exc}",
                "topk_policy": metrics_round(
                    topk.to_dict() if hasattr(topk, "to_dict") else {}
                ),
                "applied": {
                    "n_selected": int(applied.n_selected),
                    "selected_value": float(applied.selected_value),
                    "selected_cost": float(getattr(applied, "selected_cost", float("nan"))),
                },
            }
        stages["decisions"] = {
            "status": "ok",
            "threshold_policy": metrics_round(
                thr.to_dict() if hasattr(thr, "to_dict") else {}
            ),
            "threshold_test": metrics_round(
                thr_test.to_dict() if hasattr(thr_test, "to_dict") else {}
            ),
            **alloc_payload,
        }
        write_results(ctx, stages["decisions"], filename="decisions.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["decisions"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"decisions: {exc}")

    ok_stages = sum(1 for v in stages.values() if v.get("status") == "ok")
    summary = {
        "status": "completed" if ok_stages >= 3 else "partial",
        "product": "Aurora Ad Ranker",
        "data": data_meta,
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "leakage_controls": [
            "LTR group_split by query_id before ranker fit",
            "Classical CTR split is stratified and disjoint from test",
            "Impression capacity / knapsack tuned on validation only",
            "Test nDCG and decision eval after each stage locks",
        ],
        "what_fails_if_leakage_ignored": [
            "Fitting the ranker on test queries overstates NDCG",
            "Allocating impressions on test invents CTR lift",
            "Tuning serve thresholds on test understates opportunity cost",
        ],
        "limitations": [
            "Synthetic graded ad judgments — not a real auction log",
            "CTR proxy is derived from judgment aggregates",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "aurora-ad-ranker OK",
        {
            "ranking": (stages.get("ranking") or {}).get("status"),
            "classical": (stages.get("classical") or {}).get("status"),
            "decisions": (stages.get("decisions") or {}).get("status"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
