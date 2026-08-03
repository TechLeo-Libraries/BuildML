"""Tier A proof: sponsored ad learning-to-rank."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from buildml import Session
from buildml.core.errors import MissingExtraError
from proofs._lib import (
    extra_available,
    load_ad_ltr_judgments_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)


def main() -> None:
    ctx = new_proof_context("sponsored-ad-ltr", seed=114)
    frame, data_meta = load_ad_ltr_judgments_synthetic(seed=ctx.seed)
    lgbm = extra_available("lightgbm")
    session = (
        Session.ingest(frame)
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
            fit = session.fit_ranker(
                method="lambdarank",
                query_column="query_id",
                item_column="ad_id",
                random_state=ctx.seed,
            )
        else:
            raise MissingExtraError("ranking-industry", "lambdarank")
    except (MissingExtraError, TypeError, ValueError):
        fit = session.fit_ranker(
            method="pointwise",
            query_column="query_id",
            item_column="ad_id",
            pointwise_estimator="ridge",
            random_state=ctx.seed,
        )
        method = "pointwise"
    ranked = session.rank(partition="test", k=5)
    ev = session.evaluate_ranker(partition="test", k=5)
    bundle = session.save_ranker_bundle(ctx.artifacts_dir / "ranker_bundle")
    write_results(
        ctx,
        {
            "status": "completed",
            "data": data_meta,
            "lightgbm_available": lgbm,
            "method": method,
            "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
            "rank_sample": metrics_round(
                ranked.to_dict() if hasattr(ranked, "to_dict") else {}
            ),
            "test_metrics": metrics_round(dict(ev.metrics)),
            "bundle_path": str(bundle),
            "leakage_controls": [
                "group_split on query_id",
                "Train-only ranker fit",
                "Test nDCG after lock",
            ],
            "industry_comparison": {
                "status": "filled",
                "note": (
                    "Tier C baseline_industry.py: pointwise Ridge LTR twin on the same split; "
                    "run script then baseline_industry.py for results/comparison.json."
                ),
            },
            "limitations": ["Synthetic graded ad judgments — not a real auction log"],
        },
    )
    print("sponsored-ad-ltr OK", dict(ev.metrics))


if __name__ == "__main__":
    main()
