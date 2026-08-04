"""Graph example: set_graph → classical fit → eval → bundle.

Requires: pip install 'buildml[graph]'
Honesty: node classification with NetworkX metrics + sklearn. Optional PyG
path: ``session.graph.fit(method='pyg', pyg_model='graphsage')`` with ``buildml[graph-pyg]``.
Not Neo4j/KG.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def main() -> None:
    rng = np.random.default_rng(3)
    n_per, n = 45, 90
    labels = np.array([0] * n_per + [1] * n_per)
    x = labels.astype(float).reshape(-1, 1) + rng.normal(scale=0.3, size=(n, 2))
    nodes = pd.DataFrame(
        {"node_id": np.arange(n), "f1": x[:, 0], "f2": x[:, 1], "y": labels}
    )
    edges: list[tuple[int, int]] = []
    for start in (0, n_per):
        members = range(start, start + n_per)
        for i in members:
            for j in members:
                if i < j and rng.random() < 0.18:
                    edges.append((i, j))
    edges_df = pd.DataFrame(edges, columns=["source", "target"])

    session = (
        Session.ingest(nodes)
        .set_roles(
            {"node_id": "id", "f1": "feature", "f2": "feature", "y": "target"}
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
    )
    session.graph.set_spec(edges_df, node_id_col="node_id")
    session.scale(columns=["f1", "f2"], method="standard")
    fit = session.graph.fit(method="classical", mode="inductive", random_state=0)
    print("fit", fit.to_dict())
    ev = session.graph.evaluate(partition="validation")
    print("eval", ev.metrics)
    out = Path("artifacts/graph_classical_bundle")
    session.graph.save_bundle(out)
    print("bundle", out)


if __name__ == "__main__":
    main()
