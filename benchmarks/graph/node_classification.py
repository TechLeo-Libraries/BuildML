"""Node-classification benchmark across graph backends."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.graph.catalog import graph_capability_matrix
from buildml.graph.extras import networkx_available, pyg_available
from buildml.graph.fit import fit_graph


def _community_graph(n_per: int = 40, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = n_per * 2
    labels = np.array([0] * n_per + [1] * n_per)
    x = labels.astype(float).reshape(-1, 1) + rng.normal(scale=0.35, size=(n, 2))
    nodes = pd.DataFrame(
        {"node_id": np.arange(n), "f1": x[:, 0], "f2": x[:, 1], "y": labels}
    )
    edges: list[tuple[int, int]] = []
    for start in (0, n_per):
        members = list(range(start, start + n_per))
        for i in members:
            for j in members:
                if i < j and rng.random() < 0.18:
                    edges.append((i, j))
    for i in range(n_per):
        for j in range(n_per, n):
            if rng.random() < 0.02:
                edges.append((i, j))
    return nodes, pd.DataFrame(edges, columns=["source", "target"])


def _run_case(
    method: str,
    *,
    pyg_model: str | None = None,
    epochs: int = 50,
) -> dict[str, object]:
    nodes, edges = _community_graph()
    session = (
        Session.ingest(nodes)
        .set_roles(
            {"node_id": "id", "f1": "feature", "f2": "feature", "y": "target"}
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
    )
    session.set_graph(edges, node_id_col="node_id", directed=False)
    session.scale(columns=["f1", "f2"], method="standard")
    from buildml.graph.data import build_graph_spec

    spec = session.graph_spec
    assert spec is not None
    kwargs: dict[str, object] = {
        "method": method,
        "mode": "inductive",
        "epochs": epochs,
        "hidden_dim": 16,
        "random_state": 0,
    }
    if pyg_model is not None:
        kwargs["pyg_model"] = pyg_model
    plan, fit = fit_graph(
        session.dataset,
        session.split_plan,
        spec,
        **kwargs,  # type: ignore[arg-type]
    )
    session._graph_plan = plan
    ev = session.evaluate_graph(partition="test")
    label = method if pyg_model is None else f"pyg:{pyg_model}"
    return {
        "label": label,
        "method": method,
        "pyg_model": pyg_model,
        "train_accuracy": fit.train_accuracy,
        "n_train_nodes": fit.n_train_nodes,
        "n_edges_fit": fit.n_edges_fit,
        "metrics": dict(ev.metrics),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Graph node-classification benchmark")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/graph/results/node_classification.json"),
    )
    args = parser.parse_args(argv)

    matrix = graph_capability_matrix()
    runs: list[dict[str, object]] = []

    if networkx_available():
        runs.append(_run_case("classical"))
    else:
        runs.append({"label": "classical", "skipped": "networkx missing"})

    try:
        import torch  # noqa: F401

        runs.append(_run_case("gcn", epochs=40))
    except Exception as exc:
        runs.append({"label": "gcn", "skipped": str(exc)})

    if pyg_available():
        for model in ("gcn", "graphsage", "gat"):
            try:
                runs.append(_run_case("pyg", pyg_model=model, epochs=40))
            except Exception as exc:
                runs.append({"label": f"pyg:{model}", "skipped": str(exc)})
    else:
        runs.append({"label": "pyg", "skipped": "torch-geometric missing"})

    payload = {
        "capability_matrix": matrix,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
