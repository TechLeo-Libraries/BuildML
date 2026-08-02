# Quickstart: Graph ML

Session path for **node classification** over an edge list + node feature
table: `set_graph` → `fit_graph` → `predict_graph` / `evaluate_graph` →
`buildml.graph_bundle.v1`.

**Conventions:** Session rows are **nodes**. Edges are a separate table whose
endpoints match a unique `node_id` column. `Session.split` creates **node**
partitions.

**Leakage modes:**
- `inductive` (default): fit on the train-induced subgraph; score may use
  train↔holdout edges; holdout↔holdout dropped.
- `transductive`: full topology with train-label-only supervision (disclosed).

**Three complete paths:**
1. Classical — NetworkX metrics + sklearn (`pip install 'buildml[graph]'`)
2. Pure-Torch GCN — dense adjacency, no PyG (`pip install 'buildml[torch]'`)
3. PyTorch Geometric — GCN / GraphSAGE / GAT (`pip install 'buildml[graph-pyg]'`)

Honesty: not a Neo4j/KG product, not a full PyG algorithm zoo, not link-prediction
depth in this surface.

**Go deeper:** [Graph deep](graph-deep.md) ·

**Proof:** [graph-fraud-rings](../proofs/graph-fraud-rings/) (+ Tier C networkx+LR). Cross-domain: [aegis-fraud-platform](../proofs/aegis-fraud-platform/).
[Artifacts](artifacts-checkpoints-bundles.md)

```python
import numpy as np
import pandas as pd
from buildml import Session

rng = np.random.default_rng(0)
n_per, n = 40, 80
labels = np.array([0] * n_per + [1] * n_per)
x = labels.astype(float).reshape(-1, 1) + rng.normal(scale=0.3, size=(n, 2))
nodes = pd.DataFrame(
    {"node_id": np.arange(n), "f1": x[:, 0], "f2": x[:, 1], "y": labels}
)
edges = []
for start in (0, n_per):
    members = range(start, start + n_per)
    for i in members:
        for j in members:
            if i < j and rng.random() < 0.2:
                edges.append((i, j))
edges_df = pd.DataFrame(edges, columns=["source", "target"])

session = (
    Session.ingest(nodes)
    .set_roles(
        {"node_id": "id", "f1": "feature", "f2": "feature", "y": "target"}
    )
    .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
)
session.set_graph(edges_df, node_id_col="node_id")
# Scale features only — avoid mutating node_id (set_graph also snapshots ids).
session.scale(columns=["f1", "f2"], method="standard")

fit = session.fit_graph(method="classical", mode="inductive")
print(fit.train_accuracy, fit.n_edges_fit)

# PyG path (when buildml[graph-pyg] installed):
# fit = session.fit_graph(method="pyg", pyg_model="graphsage", epochs=60)

ev = session.evaluate_graph(partition="validation")
print(ev.metrics)

session.save_graph_bundle("artifacts/graph_bundle")
```

| In scope | Out of scope |
| --- | --- |
| Node classification | Neo4j / knowledge-graph product |
| Classical NetworkX + sklearn | Full PyG paper zoo beyond GCN/SAGE/GAT |
| Pure-Torch GCN + PyG GCN/SAGE/GAT | Link prediction product depth |
| Inductive / transductive modes | Graph-level classify zoo |
| Distinct `buildml.graph_bundle.v1` | Silent full-graph train as "inductive" |

Next Phase 2 item after this: **Evolutionary algorithms** (search/HPO backend).
