"""Tier A proof: peer-lending-graph — P2P lending graph fraud rings."""

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
from proofs._lib import TORCH_STATUS, extra_available, metrics_round, new_proof_context, write_results


def main() -> None:
    ctx = new_proof_context("peer-lending-graph", seed=35)
    rng = np.random.default_rng(ctx.seed)
    n = 130
    # Borrowers in two communities; default denser in community 1 (P2P lending narrative).
    nodes = pd.DataFrame({
        "borrower_id": [f"b{i}" for i in range(n)],
        "credit_z": rng.normal(size=n),
        "income_z": rng.normal(size=n),
        "is_default_ring": [
            1 if i >= n // 2 and rng.random() < 0.38 else 0 for i in range(n)
        ],
    })
    edges = []
    for i in range(n):
        community = i // (n // 2)
        lo, hi = community * (n // 2), (community + 1) * (n // 2)
        for j in rng.choice(range(lo, hi), size=3, replace=True):
            if i != j:
                edges.append((f"b{i}", f"b{int(j)}"))
    edge_frame = pd.DataFrame(edges, columns=["source", "target"]).drop_duplicates()
    session = Session.ingest(nodes)
    session.set_roles({
        "borrower_id": "id",
        "credit_z": "feature",
        "income_z": "feature",
        "is_default_ring": "target",
    })
    session.split(test_size=0.25, validation_size=0.15, stratify=True, random_state=ctx.seed)
    session.graph.set_spec(
        edge_frame,
        source_col="source",
        target_col="target",
        node_id_col="borrower_id",
    )
    fit = session.graph.fit(method="classical", mode="inductive", random_state=ctx.seed)
    ev = session.graph.evaluate(partition="test")
    torch_probe = {"ran": False, "skip": TORCH_STATUS.get("skip_torch_paths", True)}
    if not TORCH_STATUS.get("skip_torch_paths") and extra_available("torch"):
        try:
            session2 = Session.ingest(nodes)
            session2.set_roles({
                "borrower_id": "id",
                "credit_z": "feature",
                "income_z": "feature",
                "is_default_ring": "target",
            })
            session2.inject_split(
                train_indices=list(session.split_plan.train_indices),
                validation_indices=list(session.split_plan.validation_indices),
                test_indices=list(session.split_plan.test_indices),
            )
            session2.graph.set_spec(
                edge_frame,
                source_col="source",
                target_col="target",
                node_id_col="borrower_id",
            )
            f2 = session2.graph.fit(method="gcn", epochs=20, random_state=ctx.seed)
            e2 = session2.graph.evaluate(partition="test")
            torch_probe = {
                "ran": True,
                "fit": metrics_round(f2.to_dict() if hasattr(f2, "to_dict") else {}),
                "test_metrics": metrics_round(dict(getattr(e2, "metrics", {}) or {})),
            }
        except (MissingExtraError, TypeError, ValueError) as exc:
            torch_probe = {"ran": False, "error": f"{type(exc).__name__}: {exc}"}
    bundle = session.graph.save_bundle(ctx.artifacts_dir / "graph_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {
            "name": "synthetic_p2p_lending_graph",
            "license": "synthetic/public-domain",
            "n_nodes": n,
            "n_edges": int(len(edge_frame)),
        },
        "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
        "test_metrics": metrics_round(dict(getattr(ev, "metrics", {}) or {})),
        "torch_gcn_probe": torch_probe,
        "torch": TORCH_STATUS,
        "bundle_path": str(bundle),
        "leakage_controls": [
            "Node split before fit",
            "Classical features from train graph view",
            "Test after lock",
        ],
        "industry_comparison": {
            "status": "filled",
            "note": (
                "Tier C baseline_industry.py: NetworkX degree/clustering + sklearn LR; "
                "run script then baseline_industry.py for results/comparison.json."
            ),
        },
        "limitations": [
            "Synthetic P2P communities; classical primary path",
            "Distinct narrative from graph-fraud-rings (card fraud)",
        ],
    })
    print("peer-lending-graph OK", getattr(ev, "metrics", ev))


if __name__ == "__main__":
    main()
