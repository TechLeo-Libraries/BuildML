# Graph ML (deep)

BuildML’s graph path is a **Session-facing node classification** surface over
an edge list plus a node feature table. It is intentionally not a graph
database and not an exhaustive PyG research suite.

## Mental model

1. Ingest a **node table** (one row per node) and assign roles. Prefer
   `node_id` with role `id`.
2. `set_graph(edges, node_id_col=...)` attaches structure. Endpoints must match
   `node_id` values (not raw row positions unless those *are* your ids).
3. `split` creates **node** partitions (train / validation / test).
4. `fit_graph` learns under an explicit `mode`:
   - **inductive** (default): fit edges = train–train only.
   - **transductive**: full adjacency; loss / sklearn fit rows = train labels
     only (holdout features may participate via edges — disclosed).
5. `predict_graph` / `evaluate_graph` score holdout nodes with a frozen plan.
6. `save_graph_bundle` / `load_graph_bundle` persist `GraphPlan` separately
   from Session checkpoints.

## Paths

| Method | Extra | What it does |
| --- | --- | --- |
| `classical` | `buildml[graph]` (NetworkX) | Degree, clustering, PageRank, avg neighbor degree, optional betweenness (n≤200) + tabular features → logistic regression or random forest |
| `gcn` | `buildml[torch]` | 1–2 layer Kipf–Welling GCN on symmetric normalized dense adjacency; train-mask CE |
| `pyg` | `buildml[graph-pyg]` | PyTorch Geometric GCNConv / SAGEConv / GATConv via `pyg_model=`; sparse `edge_index`; train-mask CE |

### Why a separate `graph-pyg` extra?

PyTorch Geometric couples tightly to specific Torch/CUDA builds and pulls a
heavy stack. Keeping it behind `buildml[graph-pyg]` preserves a light core
install while still shipping industry GNN depth when requested. The pure-Torch
`gcn` path remains available with only `buildml[torch]` for environments that
avoid PyG.

```python
from buildml.graph import graph_capability_matrix

print(graph_capability_matrix()["backends"])
```

## Leakage discipline

- Labels for fitting always come from the train partition.
- Inductive fit drops any edge with a holdout endpoint.
- Inductive score keeps train↔holdout edges (semi-inductive neighborhood) but
  drops holdout↔holdout edges so unlabeled cliques cannot invent structure
  unseen at fit.
- Transductive is available and clearly disclosed — do not call it inductive.

## Bundle boundary

`buildml.graph_bundle.v1` stores `GraphPlan` (GraphSpec + estimator/GCN/PyG +
label encoder). Session checkpoints do **not** embed the graph learner.

## Residuals (honest)

- Node classification only (no link prediction / graph-level classify depth).
- Not Neo4j / KG (separate `buildml.kg` path).
- PyG surface ships GCN / GraphSAGE / GAT only — not GIN, PNA, etc.
- Size guard (≤5000 nodes) for dense adjacency (gcn) and Session materialization.
- Default `scale()` / `encode()` skip `id` / `ignore` / `target` / `group` /
  `time` / `weight` roles. Pass `columns=[...]` only when you intentionally
  want to transform a non-feature column.
