# Quickstart: Knowledge graphs

> **Install:**
> `pip install buildml`
> Core path (numpy TransE / DistMult): no Neo4j, no extra required.
> Industry path: `pip install 'buildml[kg-industry]'` for PyKEEN RotatE/ComplEx.
> See [installation](../docs/installation.rst).

Session knowledge-graph learning on `(head, relation, tail)` triples.
Train-only vocabularies and embeddings, filtered link-prediction metrics
(MRR, Hits@K), and symbolic neighborhood / path / typed queries over the
**train** adjacency.

**Not** a Neo4j / graph-database product. **Not** Graph ML node
classification (`session.graph.set_spec` / `session.graph.fit`). **Not** RAG retrieve/generate.

**Proof:** [kg-biomed-linkpred](../proofs/kg-biomed-linkpred/) (+ Tier C co-occurrence PMI twin).

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
print(fit.to_dict())

# Link prediction: who might Alice work_at?
preds = session.kg.predict_links(
    mode="tail",
    heads=["Alice"],
    relations=["works_at"],
    k=5,
)
print(preds.predictions)

# Symbolic query over train edges (not LLM)
nbrs = session.kg.query(mode="neighbors", entity="Alice", direction="out")
print(nbrs.results)

path = session.kg.query(mode="path", source="Alice", target="London", max_hops=3)
print(path.results)

ev = session.kg.evaluate(partition="test", k=5)
print(ev.metrics)  # mrr, hits_at_1/3/k, mean_rank (filtered ranking)

session.kg.save_bundle("artifacts/kg_demo_bundle")
# Roundtrip: load on a fresh Session with the same split, then re-evaluate
# other.kg.load_bundle("artifacts/kg_demo_bundle", trusted=True)
# other.kg.evaluate(partition="test", k=5)
```

---

## Leakage rules (read these)

| Rule | Detail |
|------|--------|
| Split required | `session.kg.fit` calls `assert_can_fit("train")` |
| Train-only materialization | Unique triples, entity/relation vocab, adjacency from train |
| Negative sampling | Uniform head/tail corruption of **train** triples only (`neg_ratio`) |
| Holdout | Never updates embeddings; OOV holdout triples skipped at eval |
| Filtered ranking | Other known true triples removed when ranking candidates |

---

## Distinguish from nearby paths

| Path | What it is |
|------|------------|
| **KG (this)** | Triples → native or PyKEEN embeddings + symbolic query |
| Graph ML | Node table + `session.graph.set_spec` edges → node classification |
| RAG | Chunk corpus → embed/retrieve/generate |
| Recommenders | User–item interactions → top-K CF |

---

## Scope notes

- Related: recommenders and LTR
- Knowledge graphs (this guide) ship with industry extras when installed
- Related next: probabilistic ML
