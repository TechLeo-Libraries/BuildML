# Knowledge graphs — deep guide

Session-shaped knowledge graphs: triples, embeddings, filtered link
prediction, and symbolic structure queries. This is a **learning/query
path**, not a graph database product.

## What BuildML ships

1. **Triple store from columns** — `head_column`, `relation_column`,
   `tail_column` on Session rows. Unique train triples only.
2. **Embeddings** — pure-numpy **TransE** (`−‖h+r−t‖`) and **DistMult**
   (`⟨h,r,t⟩`) with margin ranking loss and uniform negative sampling.
3. **Link prediction** — `score_triples`, `predict_links(mode='tail'|'head'|'relation')`.
4. **Evaluation** — filtered **MRR**, **Hits@1/3/K** (head+tail average).
5. **Symbolic query** — `query_kg(mode='neighbors'|'typed'|'path')` on
   train adjacency (BFS, not LLM / Cypher).
6. **Bundle** — `buildml.kg_bundle.v1` (`meta.json` + `kg_plan.joblib`).

## Honesty boundaries

| Claim | Reality |
|-------|---------|
| Neo4j / Cypher product | **No** — in-memory train adjacency + embeddings |
| Graph ML node classify | **Separate** — `set_graph` / `fit_graph` |
| RAG | **Separate** — chunk embed/retrieve/generate |
| Torch / PyG required | **No** — numpy SGD in core |
| Production KG platform | **No** — Session-scale complete path |

## Negative sampling (disclosed)

For each positive train triple, BuildML corrupts **head or tail**
(equal probability) by sampling a uniform replacement entity from the
**train** catalog (`neg_ratio` times). Holdout triples are never used as
positives or negatives during `fit_kg`. Disclosures on `KgFitResult`
record `neg_ratio` and the scoring formula.

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
| `predict_links` | Soft completions from embeddings |
| `query_kg` | Exact neighbors / typed / shortest path on **train** edges |

`query_kg` never invents edges. An empty path means no train path within
`max_hops`, not model failure.

## Leakage checklist

- [ ] `split` (or group_split) before `fit_kg`
- [ ] Triple id columns marked `id` / `ignore` so classical `fit()` ignores them
- [ ] Read fit disclosures for negative sampling
- [ ] Evaluate with `evaluate_kg`, not training loss alone
- [ ] Reload via `load_kg_bundle` — Session checkpoints do not embed `KgPlan`

## API surface

```text
Session.fit_kg(...)
Session.score_triples(partition=... | triples=...)
Session.predict_links(mode=..., heads=..., relations=..., tails=..., k=...)
Session.query_kg(mode=..., entity=..., source=..., target=..., relation=...)
Session.evaluate_kg(partition=..., k=...)
Session.save_kg_bundle(path) / load_kg_bundle(path)
```

## Tracker

Phase 3 application systems — depth-first:

1. Recommendation systems — **PASS**
2. Search / LTR — **PASS**
3. Knowledge graphs (this guide) — Phase-1 bar
4. Next after KG PASS: **optimisation / decision helpers**
5. Then synthetic-data systems
