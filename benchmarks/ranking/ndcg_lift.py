"""nDCG lift benchmark: industry GBDT rankers vs sklearn pointwise baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.ranking.catalog import ranking_capability_matrix
from buildml.ranking.extras import lightgbm_available, ranking_industry_available


def _judgment_frame(n_queries: int = 48, n_items: int = 10, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_queries):
        for item in range(n_items):
            f1 = float(rng.normal(q % 6, 0.65))
            f2 = float(item * 0.4 + rng.normal(0, 0.3))
            bm25 = float(rng.random())
            rel = float(max(0, int(4 - abs(f1 - (q % 6)) + (item % 2 == 0))))
            rows.append(
                {
                    "query_id": f"q{q}",
                    "item_id": f"i{item}",
                    "f1": f1,
                    "f2": f2,
                    "bm25": bm25,
                    "relevance": rel,
                }
            )
    return pd.DataFrame(rows)


def _run(backend: str | None, method: str | None, *, k: int = 5) -> dict[str, object]:
    session = (
        Session.ingest(_judgment_frame())
        .set_roles(
            {
                "query_id": "group",
                "item_id": "id",
                "relevance": "target",
                "f1": "feature",
                "f2": "feature",
                "bm25": "feature",
            }
        )
        .group_split(test_size=0.25, validation_size=0.15, random_state=0)
    )
    fit = session.fit_ranker(
        backend=backend,  # type: ignore[arg-type]
        method=method,  # type: ignore[arg-type]
        query_column="query_id",
        item_column="item_id",
    )
    ev = session.evaluate_ranker(partition="test", k=k)
    return {
        "backend": fit.backend,
        "method": fit.method,
        "n_train_queries": fit.n_train_queries,
        "ndcg_at_k": ev.metrics.get("ndcg_at_k"),
        "map_at_k": ev.metrics.get("map_at_k"),
        "mrr_at_k": ev.metrics.get("mrr_at_k"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BuildML LTR nDCG lift benchmark (industry vs pointwise baseline)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/ranking/results/ndcg_lift.json"),
    )
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)

    matrix = ranking_capability_matrix()
    baseline = _run("sklearn", "pointwise", k=args.k)

    industry_result: dict[str, object] | None = None
    if ranking_industry_available():
        default_method = matrix["default_method_when_installed"]
        if default_method not in {"pointwise", "pairwise"}:
            industry_result = _run("industry", str(default_method), k=args.k)
        elif lightgbm_available():
            industry_result = _run("industry", "lambdarank_lgbm", k=args.k)

    ndcg_baseline = float(baseline.get("ndcg_at_k") or 0.0)
    ndcg_industry = (
        None
        if industry_result is None
        else float(industry_result.get("ndcg_at_k") or 0.0)
    )
    lift = (
        None
        if ndcg_industry is None or ndcg_baseline <= 0
        else float((ndcg_industry - ndcg_baseline) / ndcg_baseline)
    )

    payload = {
        "capability_matrix": {
            "default_backend": matrix["default_backend_when_installed"],
            "default_method": matrix["default_method_when_installed"],
            "industry_available": ranking_industry_available(),
        },
        "k": args.k,
        "baseline_pointwise": baseline,
        "industry_default": industry_result,
        "ndcg_lift_vs_pointwise": lift,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
