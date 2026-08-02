"""Session KG loop: fit_kg → predict_links / query_kg → evaluate_kg → bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session


def _synthetic_triples(seed: int = 0) -> pd.DataFrame:
    """Build a small but connected multi-relation KG with repeated patterns."""
    rng = np.random.default_rng(seed)
    people = [f"p{i}" for i in range(24)]
    orgs = [f"org{i}" for i in range(6)]
    cities = [f"city{i}" for i in range(4)]
    triples: list[tuple[str, str, str]] = []

    for i, person in enumerate(people):
        org = orgs[i % len(orgs)]
        city = cities[i % len(cities)]
        triples.append((person, "works_at", org))
        triples.append((person, "lives_in", city))
        triples.append((org, "located_in", city))
        # Social edges with local structure
        other = people[(i + 1) % len(people)]
        triples.append((person, "knows", other))
        if i % 3 == 0:
            triples.append((person, "knows", people[(i + 2) % len(people)]))

    # Extra noisy but structured triples
    for _ in range(40):
        a, b = rng.choice(people, size=2, replace=False)
        triples.append((str(a), "knows", str(b)))

    frame = pd.DataFrame(triples, columns=["head", "relation", "tail"])
    return frame.drop_duplicates().reset_index(drop=True)


def main() -> None:
    frame = _synthetic_triples()
    session = (
        Session.ingest(frame)
        .set_roles({"head": "id", "relation": "id", "tail": "id"})
        .split(test_size=0.2, validation_size=0.1, random_state=0)
    )

    fit = session.fit_kg(
        method="transe",
        head_column="head",
        relation_column="relation",
        tail_column="tail",
        embedding_dim=32,
        epochs=50,
        batch_size=64,
        learning_rate=0.05,
        neg_ratio=2,
        random_state=0,
    )
    print("fit", {k: fit.to_dict()[k] for k in (
        "method", "n_train_triples", "n_entities", "n_relations",
        "embedding_dim", "epochs_run", "final_loss", "neg_ratio",
    )})

    preds = session.predict_links(
        mode="tail",
        heads=["p0"],
        relations=["works_at"],
        k=5,
    )
    print("predict_links", preds.predictions)

    nbrs = session.query_kg(mode="neighbors", entity="p0", direction="out")
    print("neighbors", nbrs.n_results, nbrs.results[:5])

    typed = session.query_kg(
        mode="typed", entity="p0", relation="works_at", direction="out"
    )
    print("typed", typed.results)

    path = session.query_kg(mode="path", source="p0", target="city0", max_hops=3)
    print("path", path.results)

    ev = session.evaluate_kg(partition="test", k=5)
    print("eval", ev.metrics)

    out = Path("artifacts/kg_demo_bundle")
    session.save_kg_bundle(out)
    other = (
        Session.ingest(frame)
        .set_roles({"head": "id", "relation": "id", "tail": "id"})
        .split(test_size=0.2, validation_size=0.1, random_state=0)
    )
    other.load_kg_bundle(out)
    ev2 = other.evaluate_kg(partition="test", k=5)
    print("reloaded_eval", ev2.metrics)


if __name__ == "__main__":
    main()
