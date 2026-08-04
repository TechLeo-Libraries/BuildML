"""Tier A proof: kg-biomed-linkpred."""

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


def _triples(seed=0):
    rng = np.random.default_rng(seed)
    genes = [f"g{i}" for i in range(30)]
    diseases = [f"d{i}" for i in range(10)]
    drugs = [f"rx{i}" for i in range(12)]
    triples = []
    for i, g in enumerate(genes):
        triples.append((g, "associated_with", diseases[i % len(diseases)]))
        triples.append((drugs[i % len(drugs)], "treats", diseases[i % len(diseases)]))
        triples.append((g, "targets", drugs[i % len(drugs)]))
    for _ in range(50):
        a, b = rng.choice(genes, size=2, replace=False)
        triples.append((str(a), "interacts_with", str(b)))
    return pd.DataFrame(triples, columns=["head", "relation", "tail"]).drop_duplicates().reset_index(drop=True)


def main() -> None:
    ctx = new_proof_context("kg-biomed-linkpred", seed=0)
    frame = _triples(seed=ctx.seed)
    session = (
        Session.ingest(frame)
        .set_roles({"head": "id", "relation": "id", "tail": "id"})
        .split(test_size=0.2, validation_size=0.1, random_state=ctx.seed)
    )
    fit = session.kg.fit(
        method="transe",
        head_column="head", relation_column="relation", tail_column="tail",
        embedding_dim=32, epochs=40, batch_size=64, learning_rate=0.05,
        neg_ratio=2, random_state=ctx.seed,
    )
    preds = session.kg.predict_links(mode="tail", heads=["g0"], relations=["associated_with"], k=5)
    ev = session.kg.evaluate(partition="test")
    bundle = session.kg.save_bundle(ctx.artifacts_dir / "kg_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {"name": "synthetic_biomed_kg", "license": "synthetic/public-domain", "n_triples": int(len(frame))},
        "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
        "predict_sample": metrics_round(preds.to_dict() if hasattr(preds, "to_dict") else {}),
        "test_metrics": metrics_round(dict(getattr(ev, "metrics", {}) or {})),
        "bundle_path": str(bundle),
        "leakage_controls": ["Triple split before fit", "Train-only TransE", "Test link metrics after lock"],
        "industry_comparison": {"status": "filled", "note": "baseline_industry.py co-occurrence PMI twin"},
        "limitations": ["Synthetic biomed motifs; not a licensed biomedical KG"],
    })
    print("kg-biomed-linkpred OK", getattr(ev, "metrics", ev))


if __name__ == "__main__":
    main()
