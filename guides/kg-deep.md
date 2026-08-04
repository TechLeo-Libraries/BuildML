# Knowledge graphs: deep guide

Session-shaped knowledge graphs: triples, embeddings, filtered link
prediction, and symbolic structure queries. This is a **learning/query
path**, not a graph database product.

## What BuildML ships

1. **Triple store from columns**: `head_column`, `relation_column`,
   `tail_column` on Session rows. Unique train triples only.
2. **Embedding backends**
   - **native** (core): pure-numpy **TransE** and **DistMult** with margin
     ranking loss and uniform negative sampling.
   - **pykeen** (`buildml[kg-industry]`): PyKEEN pipeline for **TransE**,
     **DistMult**, **RotatE**, and **ComplEx** on train-only triples.
3. **Link prediction**: `session.kg.score_triples`, `session.kg.predict_links(mode='tail'|'head'|'relation')`.
4. **Evaluation**: filtered **MRR**, **Hits@1/3/K** (head+tail average).
5. **Symbolic query**: `session.kg.query(mode='neighbors'|'typed'|'path')` on
   train adjacency (BFS, not LLM / Cypher).
6. **Bundle**: `buildml.kg_bundle.v1` (`meta.json` + `session.kg.plan.joblib`).
7. **Capability matrix**: `session.kg.capability_matrix()` reports honest backend
   availability and install hints.

## Honesty boundaries

| Claim | Reality |
|-------|---------|
| Neo4j / Cypher product | **No**: in-memory train adjacency + embeddings |
| Graph ML node classify | **Separate**: `session.graph.set_spec` / `session.graph.fit` |
| RAG | **Separate**: chunk embed/retrieve/generate |
| Torch / PyG required (core) | **No**: numpy SGD native fallback |
| PyKEEN industry models | **Optional**: `pip install 'buildml[kg-industry]'` |
| Production KG platform | **No**: Session-scale complete path |

## Backend selection

| Backend | Extra | Methods | Engine |
|---------|-------|---------|--------|
| `native` | none | `transe`, `distmult` | numpy SGD |
| `pykeen` | `kg-industry` | `transe`, `distmult`, `rotate`, `complex` | PyKEEN pipeline |

When `backend=None`, `rotate`/`complex` route to PyKEEN; `transe`/`distmult`
default to `native`. With PyKEEN installed and no explicit backend/method,
the default backend becomes `pykeen`.

```python
# Core path (no extras)
session.kg.fit(backend="native", method="transe", ...)

# Industry path (requires pykeen)
session.kg.fit(backend="pykeen", method="rotate", ...)
```

Inspect availability:

```python
from buildml.kg import kg_capability_matrix
print(kg_capability_matrix())
```

## Negative sampling (disclosed)

**Native:** for each positive train triple, corrupt **head or tail**
(equal probability) by sampling a uniform replacement entity from the
**train** catalog (`neg_ratio` times).

**PyKEEN:** sLCWA/LCWA on the train triple factory; `neg_ratio` is
recorded for parity with native disclosures (PyKEEN controls internal
corruption counts).

Holdout triples are never used as positives or negatives during `session.kg.fit`.
Disclosures on `KgFitResult` record backend, `neg_ratio`, and scoring formula.

## Filtered ranking protocol

For each holdout triple `(h,r,t)` in the train vocab:

1. Score all train entities as tails for `(h,r,?)` and as heads for `(?,r,t)`.
2. Remove other known true triples (train ∪ holdout) from the candidate
   list except the target fill-in.
3. Record 1-indexed ranks; average MRR and Hits@K over head+tail rankings.

OOV entities/relations are skipped and counted in `n_skipped_unknown`.

## Symbolic query vs embeddings

| API | Answers |
|-----|---------|
| `session.kg.predict_links` | Soft completions from embeddings |
| `session.kg.query` | Exact neighbors / typed / shortest path on **train** edges |

`session.kg.query` never invents edges. An empty path means no train path within
`max_hops`, not model failure.

## Leakage checklist

- [ ] `split` (or group_split) before `session.kg.fit`
- [ ] Triple id columns marked `id` / `ignore` so classical `fit()` ignores them
- [ ] Read fit disclosures for backend and negative sampling
- [ ] Evaluate with `session.kg.evaluate`, not training loss alone
- [ ] Reload via `session.kg.load_bundle`: Session checkpoints do not embed `KgPlan`

## API surface

```text
session.kg.fit(backend=..., method=..., head_column=..., ...)
session.kg.score_triples(partition=... | triples=...)
session.kg.predict_links(mode=..., heads=..., relations=..., tails=..., k=...)
session.kg.query(mode=..., entity=..., source=..., target=..., relation=...)
session.kg.evaluate(partition=..., k=...)
session.kg.save_bundle(path) / session.kg.load_bundle(path)
session.kg.capability_matrix()
```

## Benchmark

```bash
python benchmarks/kg/link_prediction.py
```

Writes `benchmarks/kg/results/link_prediction.json` with native runs always
and PyKEEN runs when installed.

## Scope notes

Related domains: recommenders, search/LTR, and this knowledge-graph surface
are shipped with industry extras when installed. Related next: probabilistic ML.
