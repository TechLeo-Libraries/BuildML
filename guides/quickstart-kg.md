# Quickstart — Knowledge graphs

> **Install (GitHub 2.x):**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> Core path (numpy TransE / DistMult) — no Neo4j, no extra required.
> Industry path: `pip install 'buildml[kg-industry]'` for PyKEEN RotatE/ComplEx.
> See [installation](../docs/installation.rst).

Session knowledge-graph learning on `(head, relation, tail)` triples.
Train-only vocabularies and embeddings, filtered link-prediction metrics
(MRR, Hits@K), and symbolic neighborhood / path / typed queries over the
**train** adjacency.

**Not** a Neo4j / graph-database product. **Not** Graph ML node
classification (`set_graph` / `fit_graph`). **Not** RAG retrieve/generate.

Runnable mirror: [`examples/kg_transe_loop.py`](../examples/kg_transe_loop.py).
Deep guide: [kg-deep.md](kg-deep.md).

---

## Fit → predict / query → evaluate → bundle

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
        # ... more triples for a meaningful split ...
    ],
    columns=["head", "relation", "tail"],
)

session = (
    Session.ingest(frame)
    .set_roles(
        {
            "head": "id",
            "relation": "id",
            "tail": "id",
        }
    )
    .split(test_size=0.2, validation_size=0.1, random_state=0)
)

fit = session.fit_kg(
    method="transe",
    head_column="head",
    relation_column="relation",
    tail_column="tail",
    embedding_dim=32,
    epochs=40,
    neg_ratio=1,
    random_state=0,
)
print(fit.to_dict())

# Link prediction: who might Alice work_at?
preds = session.predict_links(
    mode="tail",
    heads=["Alice"],
    relations=["works_at"],
    k=5,
)
print(preds.predictions)

# Symbolic query over train edges (not LLM)
nbrs = session.query_kg(mode="neighbors", entity="Alice", direction="out")
print(nbrs.results)

path = session.query_kg(mode="path", source="Alice", target="London", max_hops=3)
print(path.results)

ev = session.evaluate_kg(partition="test", k=5)
print(ev.metrics)

session.save_kg_bundle("artifacts/kg_demo_bundle")
```

---

## Leakage rules (read these)

| Rule | Detail |
|------|--------|
| Split required | `fit_kg` calls `assert_can_fit("train")` |
| Train-only materialization | Unique triples, entity/relation vocab, adjacency from train |
| Negative sampling | Uniform head/tail corruption of **train** triples only (`neg_ratio`) |
| Holdout | Never updates embeddings; OOV holdout triples skipped at eval |
| Filtered ranking | Other known true triples removed when ranking candidates |

---

## Distinguish from nearby paths

| Path | What it is |
|------|------------|
| **KG (this)** | Triples → native or PyKEEN embeddings + symbolic query |
| Graph ML | Node table + `set_graph` edges → node classification |
| RAG | Chunk corpus → embed/retrieve/generate |
| Recommenders | User–item interactions → top-K CF |

---

## Tracker

- Recommenders **PASS**; LTR **PASS**
- Knowledge graphs (this guide) — **PASS** (R5.6 industry depth)
- Next: **probabilistic** (R5.7)
