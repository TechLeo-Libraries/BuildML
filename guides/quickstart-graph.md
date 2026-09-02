# Graph ML quickstart

```bash
pip install "buildml[graph]"
# pure-Torch GCN: pip install "buildml[torch]"
# PyG: pip install "buildml[graph-pyg]"
```

Node classification. Rows are nodes. Edges are a separate table keyed by
`node_id`. `set_spec` first, then a split (node partitions), then fit.
Exactly one target. Default method is `classical` (NetworkX + sklearn).
Default mode is inductive. This is not Neo4j and not `session.kg`.

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
session.graph.set_spec(edges_df, node_id_col="node_id")
# Scale features only: avoid mutating node_id (session.graph.set_spec also snapshots ids).
session.scale(columns=["f1", "f2"], method="standard")

fit = session.graph.fit(method="classical", mode="inductive")
print(fit.train_accuracy, fit.n_edges_fit)

# PyG path (when buildml[graph-pyg] installed):
# fit = session.graph.fit(method="pyg", pyg_model="graphsage", epochs=60)

ev = session.graph.evaluate(partition="validation")
print(ev.metrics)

session.graph.save_bundle("artifacts/graph_bundle")
```

| In scope | Out of scope |
| --- | --- |
| Node classification | Neo4j / knowledge-graph product |
| Classical NetworkX + sklearn | Full PyG paper zoo beyond GCN/SAGE/GAT |
| Pure-Torch GCN + PyG GCN/SAGE/GAT | Link prediction product depth |
| Inductive / transductive modes | Graph-level classify zoo |
| Distinct `buildml.graph_bundle.v1` | Silent full-graph train as "inductive" |

Related next: evolutionary algorithms (search/HPO backend).
