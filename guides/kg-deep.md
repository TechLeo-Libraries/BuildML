# Knowledge graphs

```bash
pip install buildml
# RotatE / ComplEx / PyKEEN TransE: pip install "buildml[kg-industry]"
```

You have rows that are triples: who, what relation, whom. You want to
score missing links and ask exact neighborhood questions on the same
Session split you use for everything else. That is this path. It is not
Neo4j, not Cypher, not `session.graph` node classification, and not RAG.

`session.kg.fit` needs `head_column`, `relation_column`, and
`tail_column`. Those names are not inferred. Default method is native
TransE (`embedding_dim=50`, `epochs=40`, `neg_ratio=1`, `norm="l1"`).
`method="transe"` or `"distmult"` with `backend=None` stays native even
when PyKEEN is installed. `method="rotate"` or `"complex"` routes to
PyKEEN and raises if `buildml[kg-industry]` is missing. Pass
`backend="pykeen"` yourself when you want PyKEEN TransE or DistMult.

Fit always uses unique train triples. Holdout never updates embeddings
or vocabularies. You choose the operating point (epochs, dim, k). The
API refuses a missing split, missing triple columns, and a PyKEEN
method without the extra.

Short on-ramp: [KG quickstart](quickstart-kg.md). Proof:
[kg-biomed-linkpred](../proofs/kg-biomed-linkpred/).

## A first loop

```python
import pandas as pd

from buildml import Session

frame = pd.DataFrame(
    [
        ("Alice", "works_at", "Acme"),
        ("Bob", "works_at", "Acme"),
        ("Alice", "knows", "Bob"),
        ("Acme", "located_in", "London"),
        ("Bob", "lives_in", "London"),
        ("Carol", "works_at", "Beta"),
        ("Carol", "knows", "Alice"),
        ("Beta", "located_in", "Paris"),
        ("Alice", "lives_in", "London"),
        ("Bob", "knows", "Carol"),
        ("Carol", "lives_in", "Paris"),
        ("Acme", "knows", "Beta"),
    ],
    columns=["head", "relation", "tail"],
)

session = (
    Session.ingest(frame)
    .set_roles({"head": "id", "relation": "id", "tail": "id"})
    .split(test_size=0.2, validation_size=0.1, random_state=0)
)

fit = session.kg.fit(
    method="transe",
    head_column="head",
    relation_column="relation",
    tail_column="tail",
    embedding_dim=32,
    epochs=40,
    neg_ratio=1,
    random_state=0,
)
print(fit.backend, fit.n_train_triples, fit.n_entities)

preds = session.kg.predict_links(
    mode="tail",
    heads=["Alice"],
    relations=["works_at"],
    k=5,
)
print(preds.predictions)

nbrs = session.kg.query(mode="neighbors", entity="Alice", direction="out")
print(nbrs.results)

ev = session.kg.evaluate(partition="test", k=5)
print(ev.metrics)
```

Mark the triple columns `id` (or `ignore`) so a later classical
`session.fit` does not treat them as numeric features. `split` stores
row positions; `session.kg.fit` materializes unique triples from train
only. `predict_links` ranks completions from embeddings.
`session.kg.query` walks train edges only. `evaluate` defaults to
`partition="test"` and reports filtered MRR, Hits@1/3/K, and mean rank.

## Backends

| Backend | Extra | Methods | Engine |
| --- | --- | --- | --- |
| `native` | none | `transe`, `distmult` | numpy SGD, margin ranking, uniform negatives |
| `pykeen` | `kg-industry` | `transe`, `distmult`, `rotate`, `complex` | PyKEEN pipeline on train triples |

When `backend=None`:

- `transe` / `distmult` → `native`
- `rotate` / `complex` → `pykeen` (needs the extra)
- explicit `backend="pykeen"` with `transe` uses PyKEEN, not native

```python
session.kg.fit(
    backend="native",
    method="transe",
    head_column="head",
    relation_column="relation",
    tail_column="tail",
)

# Needs buildml[kg-industry] and a working torch import.
# session.kg.fit(
#     backend="pykeen",
#     method="rotate",
#     head_column="head",
#     relation_column="relation",
#     tail_column="tail",
# )
```

Native negative sampling: for each positive train triple, corrupt head
or tail (equal chance) by drawing a uniform replacement from the train
entity catalog, `neg_ratio` times. PyKEEN uses its own sLCWA/LCWA on
the train factory; `neg_ratio` is recorded for parity. Holdout triples
are never positives or negatives during fit.

## Scoring, prediction, and query

`session.kg.score_triples` scores a partition or an explicit triple
list with the frozen plan. Unknown entities or relations are skipped
and counted.

`session.kg.predict_links` fills one slot. Default `mode="tail"`.
`k` defaults to 10. `filtered=True` removes other known true triples
from the candidate list except the target fill-in. Modes:

- `tail`: given head and relation, rank tails
- `head`: given relation and tail, rank heads
- `relation`: given head and tail, rank relations

`session.kg.query` is exact structure on train adjacency, not an LLM
and not Cypher:

| Mode | What it answers |
| --- | --- |
| `neighbors` (default) | Incident train edges for `entity` (`direction` `out` / `in` / `both`) |
| `typed` | Neighbors filtered by `relation` |
| `path` | Shortest path from `source` to `target` within `max_hops` (default 3) |

An empty path means no train path within `max_hops`. The query never
invents edges.

## Filtered ranking

For each holdout triple `(h, r, t)` whose tokens sit in the train
vocab:

1. Score all train entities as tails for `(h, r, ?)` and as heads for
   `(?, r, t)`.
2. Drop other known true triples (train union holdout) except the
   target fill-in.
3. Record 1-indexed ranks; average MRR and Hits@K over those rankings.

OOV entities and relations are skipped (`n_skipped_unknown` on the
eval result). Training loss is not a substitute for
`session.kg.evaluate`.

## What the API refuses

- `head_column` / `relation_column` / `tail_column` omitted or not
  distinct
- Fit before `split`
- Native `rotate` / `complex`
- PyKEEN when the extra is missing or torch/PyKEEN fails to import
- Score, predict, query, evaluate, or save without a prior
  `session.kg.fit`

You still decide epochs, dimension, whether to use PyKEEN, and whether
a Hits@K on a tiny graph is worth quoting.

## Bundle

`session.kg.save_bundle` writes `buildml.kg_bundle.v1` (`meta.json` plus
the plan). Session checkpoints do not embed `KgPlan`. Reload with
`session.kg.load_bundle(..., trusted=True)` on a Session that already
has the same split if you want to re-evaluate.

```python
session.kg.save_bundle("artifacts/kg_bundle")
other = (
    Session.ingest(frame)
    .set_roles({"head": "id", "relation": "id", "tail": "id"})
    .split(test_size=0.2, validation_size=0.1, random_state=0)
)
other.kg.load_bundle("artifacts/kg_bundle", trusted=True)
print(other.kg.evaluate(partition="test", k=5).metrics)
```

## Benchmark

```bash
python benchmarks/kg/link_prediction.py
```

Writes `benchmarks/kg/results/link_prediction.json`. Native always;
PyKEEN when the extra imports cleanly.
