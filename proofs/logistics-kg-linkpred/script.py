"""Tier A proof: logistics knowledge-graph link prediction (TransE)."""

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
from proofs._lib import metrics_round, new_proof_context, write_results


def _logistics_triples(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    warehouses = [f"wh{i}" for i in range(18)]
    hubs = [f"hub{i}" for i in range(8)]
    routes = [f"rt{i}" for i in range(14)]
    carriers = [f"car{i}" for i in range(6)]
    triples = []
    for i, wh in enumerate(warehouses):
        triples.append((wh, "ships_via", routes[i % len(routes)]))
        triples.append((routes[i % len(routes)], "serves", hubs[i % len(hubs)]))
        triples.append((carriers[i % len(carriers)], "operates", routes[i % len(routes)]))
        triples.append((wh, "feeds", hubs[i % len(hubs)]))
    for _ in range(40):
        a, b = rng.choice(warehouses, size=2, replace=False)
        triples.append((str(a), "transfers_to", str(b)))
    return (
        pd.DataFrame(triples, columns=["head", "relation", "tail"])
        .drop_duplicates()
        .reset_index(drop=True)
    )


def main() -> None:
    ctx = new_proof_context("logistics-kg-linkpred", seed=115)
    frame = _logistics_triples(seed=ctx.seed)
    session = (
        Session.ingest(frame)
        .set_roles({"head": "id", "relation": "id", "tail": "id"})
        .split(test_size=0.2, validation_size=0.1, random_state=ctx.seed)
    )
    fit = session.kg.fit(
        method="transe",
        head_column="head",
        relation_column="relation",
        tail_column="tail",
        embedding_dim=32,
        epochs=40,
        batch_size=64,
        learning_rate=0.05,
        neg_ratio=2,
        random_state=ctx.seed,
    )
    preds = session.kg.predict_links(
        mode="tail",
        heads=["wh0"],
        relations=["ships_via"],
        k=5,
    )
    ev = session.kg.evaluate(partition="test")
    bundle = session.kg.save_bundle(ctx.artifacts_dir / "kg_bundle")

    # Checkpoint roundtrip: load bundle on a fresh Session and re-score.
    restored = (
        Session.ingest(frame)
        .set_roles({"head": "id", "relation": "id", "tail": "id"})
        .split(test_size=0.2, validation_size=0.1, random_state=ctx.seed)
    )
    restored.kg.load_bundle(bundle, trusted=True)
    ev_reloaded = restored.kg.evaluate(partition="test")

    write_results(
        ctx,
        {
            "status": "completed",
            "data": {
                "name": "synthetic_logistics_kg",
                "license": "synthetic/public-domain",
                "n_triples": int(len(frame)),
                "notes": "Warehouse–route–hub logistics motifs; not a real TMS extract.",
            },
            "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
            "predict_sample": metrics_round(
                preds.to_dict() if hasattr(preds, "to_dict") else {}
            ),
            "test_metrics": metrics_round(dict(getattr(ev, "metrics", {}) or {})),
            "reloaded_test_metrics": metrics_round(
                dict(getattr(ev_reloaded, "metrics", {}) or {})
            ),
            "n_triples_scored": int(getattr(ev, "n_triples_scored", 0) or 0),
            "bundle_path": str(bundle),
            "bundle_roundtrip": {
                "loaded": restored.kg.plan is not None,
                "mrr_match": bool(
                    abs(
                        float(ev.metrics.get("mrr", 0.0))
                        - float(ev_reloaded.metrics.get("mrr", -1.0))
                    )
                    < 1e-9
                ),
            },
            "leakage_controls": [
                "Triple split before fit",
                "Train-only TransE",
                "Test link metrics after lock",
                "Bundle load re-score uses frozen train embeddings only",
            ],
            "industry_comparison": {
                "status": "filled",
                "note": (
                    "Tier C baseline_industry.py: PMI co-occurrence twin on the same split; "
                    "run script then baseline_industry.py for results/comparison.json."
                ),
            },
            "limitations": [
                "Synthetic logistics motifs; not a licensed TMS / network extract",
                "Native numpy TransE path (PyKEEN industry optional, subprocess-gated)",
            ],
        },
    )
    print("logistics-kg-linkpred OK", getattr(ev, "metrics", ev))


if __name__ == "__main__":
    main()
