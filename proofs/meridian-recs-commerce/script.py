"""Tier B product: Meridian Recs Commerce.

Composes collaborative recommenders + learning-to-rank for category browse
+ classical supervised purchase propensity (+ optional decision thresholds).
Honest splits; train-only fitting; holdout eval.
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
    load_catalog_interactions_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)


def _purchase_propensity(n: int = 700, seed: int = 41) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    affinity = rng.normal(0, 1, size=n)
    price_sens = rng.normal(0, 1, size=n)
    recency_z = rng.normal(0, 1, size=n)
    session_depth = rng.poisson(3, size=n).astype(float)
    logit = (
        -0.6
        + 0.9 * affinity
        - 0.55 * price_sens
        + 0.4 * recency_z
        + 0.18 * session_depth
        + rng.normal(0, 0.35, size=n)
    )
    purchased = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
    frame = pd.DataFrame(
        {
            "affinity": affinity,
            "price_sens": price_sens,
            "recency_z": recency_z,
            "session_depth": session_depth,
            "purchased": purchased,
            "promo_cost": np.where(purchased == 1, 2.5, 1.0),
            "shopper_id": [f"s-{i}" for i in range(n)],
        }
    )
    meta = {
        "name": "meridian_purchase_propensity",
        "license": "synthetic/public-domain",
        "n_rows": n,
        "positive_rate": float(purchased.mean()),
    }
    return frame, meta


def main() -> None:
    ctx = new_proof_context("meridian-recs-commerce", seed=41)
    stages: dict = {}
    skip_notes: list[str] = []

    # --- Stage 1: collaborative recommenders ---
    rec_frame, rec_meta = load_catalog_interactions_synthetic(seed=ctx.seed)
    try:
        rec_session = (
            Session.ingest(rec_frame)
            .set_roles(
                {
                    "user_id": "id",
                    "item_id": "id",
                    "rating": "target",
                    "category_code": "feature",
                    "price_band": "feature",
                }
            )
            .split(test_size=0.2, validation_size=0.15, random_state=ctx.seed)
        )
        method = "als" if extra_available("implicit") else "item_knn"
        try:
            if method == "als":
                r_fit = rec_session.recommender.fit(
                    method="als",
                    feedback="implicit",
                    user_column="user_id",
                    item_column="item_id",
                    random_state=ctx.seed,
                )
            else:
                raise MissingExtraError("recommenders", "als")
        except (MissingExtraError, TypeError, ValueError):
            r_fit = rec_session.recommender.fit(
                method="item_knn",
                user_column="user_id",
                item_column="item_id",
                n_neighbors=25,
                random_state=ctx.seed,
            )
            method = "item_knn"
        r_val = rec_session.recommender.evaluate(partition="validation", k=5)
        r_test = rec_session.recommender.evaluate(partition="test", k=5)
        stages["recommenders"] = {
            "status": "ok",
            "method": method,
            "data": rec_meta,
            "fit": metrics_round(r_fit.to_dict() if hasattr(r_fit, "to_dict") else {}),
            "validation_metrics": metrics_round(dict(getattr(r_val, "metrics", {}) or {})),
            "test_metrics": metrics_round(dict(getattr(r_test, "metrics", {}) or {})),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["recommenders"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"recommenders: {exc}")
    write_results(ctx, stages["recommenders"], filename="recommenders.json")

    # --- Stage 2: learning-to-rank for browse relevance ---
    ltr_frame, ltr_meta = load_ad_ltr_judgments_synthetic(seed=ctx.seed + 1)
    try:
        ltr_session = (
            Session.ingest(ltr_frame)
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
        rank_method = "lambdarank" if extra_available("lightgbm") else "pointwise"
        try:
            if rank_method == "lambdarank":
                rk_fit = ltr_session.ranking.fit(
                    method="lambdarank",
                    query_column="query_id",
                    item_column="ad_id",
                    random_state=ctx.seed,
                )
            else:
                raise MissingExtraError("ranking-industry", "lambdarank")
        except (MissingExtraError, TypeError, ValueError):
            rk_fit = ltr_session.ranking.fit(
                method="pointwise",
                query_column="query_id",
                item_column="ad_id",
                pointwise_estimator="ridge",
                random_state=ctx.seed,
            )
            rank_method = "pointwise"
        rk_val = ltr_session.ranking.evaluate(partition="validation", k=5)
        rk_test = ltr_session.ranking.evaluate(partition="test", k=5)
        stages["ranking"] = {
            "status": "ok",
            "method": rank_method,
            "data": ltr_meta,
            "fit": metrics_round(rk_fit.to_dict() if hasattr(rk_fit, "to_dict") else {}),
            "validation_metrics": metrics_round(dict(getattr(rk_val, "metrics", {}) or {})),
            "test_metrics": metrics_round(dict(getattr(rk_test, "metrics", {}) or {})),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["ranking"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"ranking: {exc}")
    write_results(ctx, stages["ranking"], filename="ranking.json")

    # --- Stage 3: classical purchase propensity ---
    pur_frame, pur_meta = _purchase_propensity(seed=ctx.seed)
    feats = ["affinity", "price_sens", "recency_z", "session_depth"]
    session = (
        Session.ingest(pur_frame.copy())
        .set_roles(
            {
                **{c: "feature" for c in feats},
                "purchased": "target",
                "promo_cost": "ignore",
                "shopper_id": "id",
            }
        )
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
        .scale(method="standard")
    )
    plan = session.split_plan
    assert plan is not None
    split_counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }
    session.fit(
        LogisticRegression(max_iter=1000, random_state=ctx.seed),
        task="classification",
    )
    s_val = session.evaluate(partition="validation")
    s_test = session.evaluate(partition="test")
    stages["supervised"] = {
        "status": "ok",
        "estimator": "LogisticRegression",
        "data": pur_meta,
        "validation_metrics": metrics_round(dict(s_val.metrics)),
        "test_metrics": metrics_round(dict(s_test.metrics)),
    }
    write_results(ctx, stages["supervised"], filename="supervised.json")

    # --- Stage 4: optional promo decision thresholds ---
    try:
        assert_no_test_in_selection(
            selection_partition="validation", evaluation_partition="test"
        )
        thr = session.decision.fit(
            method="threshold",
            partition="validation",
            fp_cost=1.0,
            fn_cost=3.0,
        )
        thr_test = session.decision.evaluate(partition="test")
        knap_payload: dict = {"status": "skipped"}
        try:
            knap = session.decision.fit(
                method="knapsack",
                partition="validation",
                budget=60.0,
                cost_column="promo_cost",
                id_column="shopper_id",
                score_source="model_proba",
                knapsack_solver="dp",
            )
            applied = session.decision.apply(partition="test")
            knap_payload = {
                "status": "ok",
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
            try:
                topk = session.decision.fit(
                    method="topk",
                    partition="validation",
                    capacity=40,
                    score_source="model_proba",
                )
                applied = session.decision.apply(partition="test")
                knap_payload = {
                    "status": "ok_topk_fallback",
                    "error": f"{type(exc).__name__}: {exc}",
                    "topk_policy": metrics_round(
                        topk.to_dict() if hasattr(topk, "to_dict") else {}
                    ),
                    "applied": {
                        "n_selected": int(applied.n_selected),
                        "selected_value": float(applied.selected_value),
                    },
                }
            except Exception as exc2:  # noqa: BLE001
                knap_payload = {
                    "status": "skipped",
                    "error": f"{type(exc).__name__}: {exc}; fallback: {exc2}",
                }
                skip_notes.append(f"decisions_alloc: {exc}")
        stages["decisions"] = {
            "status": "ok",
            "threshold_policy": metrics_round(
                thr.to_dict() if hasattr(thr, "to_dict") else {}
            ),
            "threshold_test": metrics_round(
                thr_test.to_dict() if hasattr(thr_test, "to_dict") else {}
            ),
            **knap_payload,
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["decisions"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"decisions: {exc}")
    write_results(ctx, stages["decisions"], filename="decisions.json")

    summary = {
        "status": "completed",
        "product": "Meridian Recs Commerce",
        "data": {
            "recommenders": rec_meta,
            "ranking": ltr_meta,
            "supervised": pur_meta,
        },
        "split": {"kind": plan.kind, "counts": split_counts, "stratify": True},
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "leakage_controls": [
            "Interaction / group / stratified splits before any fit",
            "Recommenders and rankers fit on train only",
            "Decision policies tuned on validation only",
            "Test evaluated once per stage after that stage locks",
        ],
        "what_fails_if_leakage_ignored": [
            "Fitting ALS on full interactions leaks test preferences into embeddings",
            "Query-group leakage in LTR inflates nDCG on held-out queries",
            "Tuning promo thresholds on test understates campaign cost",
        ],
        "limitations": [
            "Synthetic catalog / judgments / propensity — not a live commerce stack",
            "Product proof, not a production personalization certification",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "meridian-recs-commerce OK",
        {
            "supervised_roc": stages["supervised"]["test_metrics"].get("roc_auc"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
