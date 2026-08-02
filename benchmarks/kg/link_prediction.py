"""Link-prediction benchmark on synthetic KG triples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.kg.catalog import kg_capability_matrix
from buildml.kg.extras import pykeen_available
from buildml.kg.fit import fit_kg


def _synthetic_kg_frame(
    n_entities: int = 40,
    n_relations: int = 4,
    n_triples: int = 320,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic KG with held-out-style structure for ranking."""
    rng = np.random.default_rng(seed)
    entities = [f"e{i}" for i in range(n_entities)]
    relations = [f"r{i}" for i in range(n_relations)]
    rows: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    while len(rows) < n_triples:
        h = rng.choice(entities)
        r = rng.choice(relations)
        t = rng.choice(entities)
        if h == t:
            continue
        triple = (h, r, t)
        if triple in seen:
            continue
        seen.add(triple)
        rows.append(triple)
    return pd.DataFrame(rows, columns=["head", "relation", "tail"])


def _run_backend(
    backend: str,
    method: str,
    *,
    embedding_dim: int = 32,
    epochs: int = 30,
) -> dict[str, object]:
    frame = _synthetic_kg_frame()
    session = (
        Session.ingest(frame)
        .set_roles({"head": "id", "relation": "id", "tail": "id"})
        .split(test_size=0.2, validation_size=0.1, random_state=0)
    )
    plan, fit = fit_kg(
        session.dataset,
        session.split_plan,
        backend=backend,  # type: ignore[arg-type]
        method=method,  # type: ignore[arg-type]
        head_column="head",
        relation_column="relation",
        tail_column="tail",
        embedding_dim=embedding_dim,
        epochs=epochs,
        batch_size=64,
        random_state=0,
    )
    session._kg_plan = plan
    ev = session.evaluate_kg(partition="test", k=10)
    return {
        "backend": backend,
        "method": method,
        "n_entities": fit.n_entities,
        "n_relations": fit.n_relations,
        "n_train_triples": fit.n_train_triples,
        "embedding_dim": fit.embedding_dim,
        "metrics": dict(ev.metrics),
        "n_triples_scored": ev.n_triples_scored,
        "n_skipped_unknown": ev.n_skipped_unknown,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KG link-prediction benchmark")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/kg/results/link_prediction.json"),
    )
    args = parser.parse_args(argv)

    matrix = kg_capability_matrix()
    runs: list[dict[str, object]] = []

    runs.append(_run_backend("native", "transe"))
    runs.append(_run_backend("native", "distmult"))

    if pykeen_available():
        try:
            from buildml.kg.extras import pykeen_runtime_available

            if pykeen_runtime_available():
                for method in ("transe", "distmult", "rotate", "complex"):
                    runs.append(_run_backend("pykeen", method))
            else:
                runs.append(
                    {
                        "backend": "pykeen",
                        "skipped": True,
                        "reason": "pykeen installed but torch not usable",
                    }
                )
        except Exception as exc:
            runs.append(
                {
                    "backend": "pykeen",
                    "skipped": True,
                    "reason": f"pykeen runtime check failed: {exc}",
                }
            )
    else:
        runs.append(
            {
                "backend": "pykeen",
                "skipped": True,
                "reason": "buildml[kg-industry] (pykeen) not installed",
            }
        )

    payload = {
        "benchmark": "kg_link_prediction",
        "capability_matrix": matrix,
        "pykeen_available": pykeen_available(),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
