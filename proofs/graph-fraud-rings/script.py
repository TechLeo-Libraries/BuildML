"""Tier A proof: graph-fraud-rings."""

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
    ctx = new_proof_context("graph-fraud-rings", seed=0)
    rng = np.random.default_rng(ctx.seed)
    n = 120
    # Two communities; fraud denser in community 1
    nodes = pd.DataFrame({
        "node_id": [f"n{i}" for i in range(n)],
        "feat1": rng.normal(size=n),
        "feat2": rng.normal(size=n),
        "is_fraud": [1 if i >= n // 2 and rng.random() < 0.35 else 0 for i in range(n)],
    })
    edges = []
    for i in range(n):
        for j in rng.choice(range((i // (n // 2)) * (n // 2), (i // (n // 2) + 1) * (n // 2)), size=3, replace=True):
            if i != j:
                edges.append((f"n{i}", f"n{int(j)}"))
    edge_frame = pd.DataFrame(edges, columns=["source", "target"]).drop_duplicates()
    session = Session.ingest(nodes)
    session.set_roles({"node_id": "id", "feat1": "feature", "feat2": "feature", "is_fraud": "target"})
    session.split(test_size=0.25, validation_size=0.15, stratify=True, random_state=ctx.seed)
    session.set_graph(
        edge_frame,
        source_col="source",
        target_col="target",
        node_id_col="node_id",
    )
    fit = session.fit_graph(method="classical", mode="inductive", random_state=ctx.seed)
    ev = session.evaluate_graph(partition="test")
    torch_probe = {"ran": False, "skip": TORCH_STATUS.get("skip_torch_paths", True)}
    if not TORCH_STATUS.get("skip_torch_paths") and extra_available("torch"):
        try:
            session2 = Session.ingest(nodes)
            session2.set_roles({"node_id": "id", "feat1": "feature", "feat2": "feature", "is_fraud": "target"})
            session2.inject_split(
                train_indices=list(session.split_plan.train_indices),
                validation_indices=list(session.split_plan.validation_indices),
                test_indices=list(session.split_plan.test_indices),
            )
            session2.set_graph(
                edge_frame,
                source_col="source",
                target_col="target",
                node_id_col="node_id",
            )
            f2 = session2.fit_graph(method="gcn", epochs=20, random_state=ctx.seed)
            e2 = session2.evaluate_graph(partition="test")
            torch_probe = {
                "ran": True,
                "fit": metrics_round(f2.to_dict() if hasattr(f2, "to_dict") else {}),
                "test_metrics": metrics_round(dict(getattr(e2, "metrics", {}) or {})),
            }
        except (MissingExtraError, TypeError, ValueError) as exc:
            torch_probe = {"ran": False, "error": f"{type(exc).__name__}: {exc}"}
    bundle = session.save_graph_bundle(ctx.artifacts_dir / "graph_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {"name": "synthetic_fraud_graph", "license": "synthetic/public-domain", "n_nodes": n, "n_edges": int(len(edge_frame))},
        "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
        "test_metrics": metrics_round(dict(getattr(ev, "metrics", {}) or {})),
        "torch_gcn_probe": torch_probe,
        "torch": TORCH_STATUS,
        "bundle_path": str(bundle),
        "leakage_controls": ["Node split before fit", "Classical features from train graph view", "Test after lock"],
        "industry_comparison": {"status": "stub"},
        "limitations": ["Synthetic communities; classical primary path"],
    })
    print("graph-fraud-rings OK", getattr(ev, "metrics", ev))


if __name__ == "__main__":
    main()
