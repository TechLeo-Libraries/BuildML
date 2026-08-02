# Graph ML (deep)

BuildML’s graph path is a **Session-facing node classification** surface over
an edge list plus a node feature table. It is intentionally not a graph
database and not a PyTorch Geometric research suite.

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
| `gcn` | `buildml[torch]` | 1–2 layer Kipf–Welling GCN on symmetric normalized adjacency; train-mask CE |

### Why not PyTorch Geometric?

PyG couples tightly to specific Torch/CUDA builds and is heavy for the core
install story. A small dense-adjacency GCN (Session guard: ≤5000 nodes) is an
honest message-passing path with only `buildml[torch]`. Classical topology
features stay behind `buildml[graph]`. `import buildml` requires neither.

## Leakage discipline

- Labels for fitting always come from the train partition.
- Inductive fit drops any edge with a holdout endpoint.
- Inductive score keeps train↔holdout edges (semi-inductive neighborhood) but
  drops holdout↔holdout edges so unlabeled cliques cannot invent structure
  unseen at fit.
- Transductive is available and clearly disclosed — do not call it inductive.

## Bundle boundary

`buildml.graph_bundle.v1` stores `GraphPlan` (GraphSpec + estimator/GCN +
label encoder). Session checkpoints do **not** embed the graph learner.

## Residuals (honest)

- Node classification only (no link prediction / graph-level classify depth).
- Not Neo4j / KG (separate roadmap item).
- Not GAT / GraphSAGE product zoo (GCN-lite only for the neural path).
- Dense adjacency size guard (≤5000 nodes).
- Default `scale()` may mutate numeric `node_id` columns — call `set_graph`
  first (ids are snapshotted) or `scale(columns=[...features...])` explicitly.
