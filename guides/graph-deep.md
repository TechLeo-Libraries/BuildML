# Graph ML deep

```bash
pip install "buildml[graph]"
# pure-Torch GCN: pip install "buildml[torch]"
# PyG GCN / GraphSAGE / GAT: pip install "buildml[graph-pyg]"
```

You have a table of nodes and a separate edge list. You want to classify
those nodes, and you want the holdout neighborhood to stay out of the
fit. Rows are nodes. Edges are keyed by `node_id`, not by DataFrame
position unless those two happen to be the same.

`session.graph.fit()` with no extra knobs is **classical** NetworkX
metrics plus sklearn, in **inductive** mode. That stays classical even
when PyTorch Geometric is sitting on the machine. Ask for `method="gcn"`
or `method="pyg"` when you want a GNN. This is not Neo4j, and it is not
`session.kg`.

Short on-ramp: [graph quickstart](quickstart-graph.md). Proof:
[graph-fraud-rings](../proofs/graph-fraud-rings/).

## A first loop

Exactly one target. Attach structure with `set_spec` before fit. Scale
the feature columns only: `node_id` has the `id` role so a default
`scale()` leaves it alone, which is what you want.

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
session.scale(columns=["f1", "f2"], method="standard")

fit = session.graph.fit(method="classical", mode="inductive")
print(fit.train_accuracy, fit.n_edges_fit)

ev = session.graph.evaluate(partition="validation")
print(ev.metrics)

session.graph.save_bundle("artifacts/graph_bundle")
```

`set_spec` defaults are `source_col="source"`, `target_col="target"`,
`node_id_col="node_id"`, `directed=False`. Endpoints must match the
`node_id` values. Duplicate ids and empty edge lists are refused. If you
later drop or add rows, call `set_spec` again: the spec snapshots ids at
attach time.

`evaluate` and `predict` default to **validation**. Look at test once you
are done choosing.

## Methods

| Method | Extra | What it fits |
| --- | --- | --- |
| `classical` (default) | `buildml[graph]` | Degree, clustering, PageRank, average neighbor degree, betweenness when n is at most 200, concatenated with tabular features, then logistic regression or random forest |
| `gcn` | `buildml[torch]` | 1-2 layer Kipf-Welling GCN on symmetric normalized dense adjacency, train-mask cross-entropy |
| `pyg` | `buildml[graph-pyg]` | PyTorch Geometric `GCNConv` / `SAGEConv` / `GATConv` via `pyg_model`, sparse `edge_index`, train-mask cross-entropy |

`classical_estimator` is `"logistic_regression"` unless you pass
`"random_forest"`. `pyg_model` is `"gcn"` unless you pass `"graphsage"`
or `"gat"`. GAT uses `heads=4` by default. Neural knobs (`hidden_dim=32`,
`n_layers=2`, `epochs=80`, `learning_rate=0.01`) apply to `gcn` and
`pyg` only.

PyG is a separate extra because it pins Torch/CUDA tightly. The
pure-Torch `gcn` path is there for machines that should not take that
stack.

```python
# When buildml[graph-pyg] is installed:
# session.graph.fit(method="pyg", pyg_model="graphsage", epochs=60)
```

The only task on this surface is `node_classification`. Link prediction
and graph-level classify are not here. Knowledge-graph triples belong on
[session.kg](quickstart-kg.md).

## Inductive vs transductive

You choose the mode. The API will not silently widen inductive to the
full graph.

**Inductive** (default): fit edges are train-train only. At score time,
train-holdout edges are kept so a holdout node can still see its labeled
neighbors. Holdout-holdout edges are dropped so unlabeled cliques cannot
invent structure the fit never saw. Isolated nodes under that filter get
zero graph metrics, and the plan discloses it.

**Transductive**: full adjacency at fit and at score. Labels for the
loss, and sklearn fit rows, still come from train only. Holdout features
may participate through edges. That is disclosed. Do not call it
inductive.

Labels never come from validation or test. Fit refuses without a split
and without `set_spec`.

## Size and preprocess

Dense adjacency (`gcn`) and Session materialization refuse more than
5000 nodes. Filter or sample first.

Default `impute` / `encode` / `scale` touch `feature` columns and leave
`id`, `target`, `group`, `time`, `weight`, and `ignore` alone. Pass
`columns=` only when you mean to transform a non-feature column. Scaling
`node_id` after `set_spec` is a good way to desync the spec snapshot
from the frame.

## Bundles

`session.graph.save_bundle` writes `buildml.graph_bundle.v1`: the
`GraphSpec`, the estimator or GNN, and the label encoder. A Session
checkpoint does not embed `GraphPlan`. Reload the table with
`checkpoint_load`, then `session.graph.load_bundle(..., trusted=True)`
for a file you made. Loaders default to `trusted=False`.

## When it refuses

| What you see | What happened |
| --- | --- |
| No split / cannot fit train | `fit` before `split` |
| No graph spec | `fit` before `set_spec` |
| Exactly one target required | Missing target, or more than one |
| `MissingExtraError` for `graph` | NetworkX extra is not installed |
| `MissingExtraError` for `torch` / `graph-pyg` | You asked for `gcn` or `pyg` without that extra |
| More than 5000 nodes | Dense / materialization guard |
| Empty edges / duplicate `node_id` | Spec could not map endpoints to unique nodes |
| Unsupported task | Anything other than `node_classification` |

[Graph quickstart](quickstart-graph.md) ·
[graph-fraud-rings](../proofs/graph-fraud-rings/) ·
[Artifacts](artifacts-checkpoints-bundles.md)
