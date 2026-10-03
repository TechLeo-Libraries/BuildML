# ruff: noqa: E501
"""Beginner layers for knowledge graphs and link prediction."""

from __future__ import annotations

from buildml.explain.beginner._builder import ADVANCED, CORE, BeginnerLayer, _index, _layer

KG_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "kg-triples",
        plain=(
            "A knowledge graph stores facts as three-part statements: a head, a relation, and a tail. "
            "'Paris: capital_of: France'. Millions of those, and you have a structured representation of "
            "what you know, which a model can learn patterns from."
        ),
        analogy=(
            "Index cards, each with exactly one fact written in the same fixed format. Using a consistent format makes it possible to connect facts through shared entities."
        ),
        steps=(
            "Shape your data as three columns: head, relation, tail.",
            "Split the triples so some are held out for evaluation.",
            "BuildML builds vocabularies of entities and relations from the training triples only.",
            "It also builds the adjacency structure from those training triples.",
            "Unknown entities or relations have no learned embedding. Scoring reports them with negative-infinity scores; link prediction skips unknown query values and records disclosures.",
        ),
        use=(
            "When your domain is naturally relational: products and categories, people and organizations, genes and diseases.",
            "When you want to infer facts that were never explicitly recorded.",
        ),
        avoid=(
            "Do not force tabular data into triples when rows are independent; you gain nothing and lose the convenience of a table.",
            "Do not use it when relations are numeric quantities rather than named types: 'price 42' is a column, not a relation.",
        ),
        myths=(
            (
                "A knowledge graph is a database.",
                "The storage question is separate. Here it is a training set of triples, and the goal is learning patterns rather than answering lookups.",
            ),
            (
                "Every fact should be a triple.",
                "Attributes with continuous values, timestamps, and free text usually belong in ordinary columns. Triples are for named relationships between named things.",
            ),
        ),
        example=(
            "from pathlib import Path",
            "import numpy as np",
            "import pandas as pd",
            "from buildml import Session",
            "",
            "rng = np.random.default_rng(42)",
            'Path("artifacts").mkdir(exist_ok=True)',
            "# A small artificial graph demonstrates the API, not link-prediction quality.",
            'entities = ["paris", "berlin", "rome", "madrid", "france", "germany", "italy", "spain"]',
            'triples = [(entities[i], "linked_to", entities[(i + offset) % 8]) for offset in (1, 2, 3) for i in range(8)]',
            'frame = pd.DataFrame(triples, columns=["subject", "predicate", "object"])',
            'session = Session.ingest(frame).set_roles({"subject": "id", "predicate": "id", "object": "id"})',
            "session.inject_split(train_indices=list(range(16)), validation_indices=list(range(16, 20)), test_indices=list(range(20, 24)))",
            'session.kg.fit(head_column="subject", relation_column="predicate", tail_column="object", backend="native", method="transe", embedding_dim=8, epochs=5, random_state=42)',
            "print(session.kg.plan.n_entities, session.kg.plan.n_relations)",
        ),
        check=(
            "How many distinct entities and relations does your training set contain?",
            "What fraction of your evaluation triples mention an entity never seen in training?",
        ),
        tools=("fit_kg", "score_triples", "predict_links", "query_kg"),
        terms=("knowledge graph", "graph", "node", "edge"),
        difficulty=CORE,
    ),
    _layer(
        "kg-transe-distmult",
        plain=(
            "Knowledge-graph embedding gives every entity and every relation a short list of numbers, "
            "arranged so that true facts score highly and false ones do not. TransE treats a relation as a "
            "translation: head plus relation should land near tail. DistMult uses a multiplicative score instead."
        ),
        analogy=(
            "Placing every city on a map so that 'is north of' becomes an actual upward step. Once the "
            "layout works, you can check a claim geometrically rather than by looking it up."
        ),
        steps=(
            "Every entity and relation starts with random numbers.",
            "For each recorded training triple, the native model samples candidate negatives by replacing the tail. A sampled candidate is not necessarily a false fact.",
            "Margin ranking training pushes true triples to score above their corrupted versions.",
            "Repeat over many passes; the embeddings settle into a layout that encodes the relational structure.",
            "The negative-sampling scheme is disclosed, because it materially affects the result.",
        ),
        use=(
            "TransE for relations that behave like one-to-one translations.",
            "DistMult when relations are symmetric: it cannot represent direction, which is sometimes exactly right.",
        ),
        avoid=(
            "TransE can struggle with one-to-many relations because one translation must score several tails. Compare alternative embeddings on held-out triples.",
            "Do not use DistMult for asymmetric relations; it scores 'A parent_of B' and 'B parent_of A' identically.",
        ),
        myths=(
            (
                "A higher-dimensional embedding is always better.",
                "Beyond a point you are fitting noise and paying for memory. Choose the embedding dimension using validation results and memory constraints.",
            ),
            (
                "A corrupted triple is definitely false.",
                "It might be an unrecorded true fact. This is why filtered evaluation exists: it removes known-true triples from the corrupted candidates.",
            ),
        ),
        example=(
            "from pathlib import Path",
            "import numpy as np",
            "import pandas as pd",
            "from buildml import Session",
            "",
            "rng = np.random.default_rng(42)",
            'Path("artifacts").mkdir(exist_ok=True)',
            "# A small artificial graph demonstrates the API, not link-prediction quality.",
            'entities = ["paris", "berlin", "rome", "madrid", "france", "germany", "italy", "spain"]',
            'triples = [(entities[i], "linked_to", entities[(i + offset) % 8]) for offset in (1, 2, 3) for i in range(8)]',
            'frame = pd.DataFrame(triples, columns=["subject", "predicate", "object"])',
            'session = Session.ingest(frame).set_roles({"subject": "id", "predicate": "id", "object": "id"})',
            "session.inject_split(train_indices=list(range(16)), validation_indices=list(range(16, 20)), test_indices=list(range(20, 24)))",
            'session.kg.fit(head_column="subject", relation_column="predicate", tail_column="object", backend="native", method="distmult", embedding_dim=8, neg_ratio=2, epochs=5, random_state=42)',
            'print(session.kg.evaluate(partition="test", k=3).metrics)',
        ),
        check=(
            "Are your important relations symmetric or asymmetric?",
            "Does any relation map one head to many tails?",
        ),
        tools=("fit_kg", "score_triples", "evaluate_kg", "predict_links"),
        terms=("knowledge graph", "embedding", "extra", "link prediction"),
        difficulty=ADVANCED,
    ),
    _layer(
        "kg-link-prediction",
        plain=(
            "Link prediction fills in a blank. Given 'Paris: capital_of: ?', the model ranks every "
            "candidate entity by how plausible that completed fact would be. You can leave any of the three "
            "slots blank."
        ),
        analogy=(
            "A crossword clue with the pattern visible. You do not know the answer, but you can rank the "
            "candidates by how well each fits what you already know."
        ),
        steps=(
            "Use `session.kg.score_triples` when you have a complete fact and want its plausibility.",
            "Use `session.kg.predict_links` with one slot blank to get a ranked list of completions.",
            "The candidates come from the training vocabulary: nothing outside it can be proposed.",
            "Read the ranking as a shortlist for review, not as a set of asserted facts.",
            "Evaluate with filtered MRR and Hits@K, which ignore other known-true triples in the candidate list.",
        ),
        use=(
            "Recommending plausible connections for a human curator to confirm.",
            "Finding gaps in a knowledge base that should probably be filled.",
        ),
        avoid=(
            "Do not treat a top-ranked prediction as a fact; these models are pattern matchers with no notion of truth.",
            "Do not predict links involving entities absent from training: there is no embedding for them.",
        ),
        myths=(
            (
                "A high score means the fact is true.",
                "It means the fact fits the geometric pattern the model learned. Plausible-looking falsehoods score highly by construction.",
            ),
            (
                "Unfiltered and filtered metrics are roughly the same.",
                "Unfiltered ranking penalizes the model for placing *other true facts* above the target. Filtered metrics remove known true alternatives; the difference depends on the graph and the filter set. BuildML filters against its recorded training catalog.",
            ),
        ),
        example=(
            "from pathlib import Path",
            "import numpy as np",
            "import pandas as pd",
            "from buildml import Session",
            "",
            "rng = np.random.default_rng(42)",
            'Path("artifacts").mkdir(exist_ok=True)',
            "# A small artificial graph demonstrates the API, not link-prediction quality.",
            'entities = ["paris", "berlin", "rome", "madrid", "france", "germany", "italy", "spain"]',
            'triples = [(entities[i], "linked_to", entities[(i + offset) % 8]) for offset in (1, 2, 3) for i in range(8)]',
            'frame = pd.DataFrame(triples, columns=["subject", "predicate", "object"])',
            'session = Session.ingest(frame).set_roles({"subject": "id", "predicate": "id", "object": "id"})',
            "session.inject_split(train_indices=list(range(16)), validation_indices=list(range(16, 20)), test_indices=list(range(20, 24)))",
            'session.kg.fit(head_column="subject", relation_column="predicate", tail_column="object", backend="native", method="transe", embedding_dim=8, epochs=5, random_state=42)',
            'scores = session.kg.score_triples(triples=[("paris", "linked_to", "france")])',
            'candidates = session.kg.predict_links(heads=["paris"], relations=["linked_to"], k=3)',
            "print(candidates.predictions)",
            'print(session.kg.evaluate(partition="test", k=3).metrics)',
        ),
        check=(
            "Are your reported metrics filtered or unfiltered?",
            "Who reviews a predicted link before it becomes a recorded fact?",
        ),
        tools=("predict_links", "score_triples", "evaluate_kg", "fit_kg"),
        terms=("link prediction", "MRR", "knowledge graph", "embedding"),
        difficulty=CORE,
    ),
    _layer(
        "kg-symbolic-query",
        plain=(
            "Not every question needs a learned model. `session.kg.query` walks the recorded training triples "
            "directly: who is connected to this entity, which entities have this specific relation, what is "
            "the shortest chain of relations between these two. Exact answers from stored facts, no "
            "embedding involved."
        ),
        analogy=(
            "Looking something up in the index versus asking someone who has read the book to guess. The "
            "index is exact and limited to what is written down."
        ),
        steps=(
            "Neighbour queries return everything directly connected to an entity.",
            "Typed queries fix the entity and the relation and return the matching tails.",
            "Path queries find the shortest chain of relations linking two entities.",
            "All of it walks the training adjacency, so unrecorded facts simply do not appear.",
            "Combine with link prediction when you need plausible-but-unrecorded answers too.",
        ),
        use=(
            "When you need exact, explainable answers about what is actually recorded.",
            "For debugging your graph: path queries quickly reveal whether two entities are connected at all.",
        ),
        avoid=(
            "Do not use it to infer missing facts; it only reports what is stored. That is what link prediction is for.",
            "Do not run unbounded path searches on a huge dense graph without a depth limit.",
        ),
        myths=(
            (
                "This is a language-model question-answering system.",
                "It is deterministic graph traversal. No text is generated, nothing is inferred, and the answer is exactly reproducible.",
            ),
            (
                "An empty result means the fact is false.",
                "It means the fact is not recorded in your training triples. It does not establish that the relationship is false.",
            ),
        ),
        example=(
            "from pathlib import Path",
            "import numpy as np",
            "import pandas as pd",
            "from buildml import Session",
            "",
            "rng = np.random.default_rng(42)",
            'Path("artifacts").mkdir(exist_ok=True)',
            "# A small artificial graph demonstrates the API, not link-prediction quality.",
            'entities = ["paris", "berlin", "rome", "madrid", "france", "germany", "italy", "spain"]',
            'triples = [(entities[i], "linked_to", entities[(i + offset) % 8]) for offset in (1, 2, 3) for i in range(8)]',
            'frame = pd.DataFrame(triples, columns=["subject", "predicate", "object"])',
            'session = Session.ingest(frame).set_roles({"subject": "id", "predicate": "id", "object": "id"})',
            "session.inject_split(train_indices=list(range(16)), validation_indices=list(range(16, 20)), test_indices=list(range(20, 24)))",
            'session.kg.fit(head_column="subject", relation_column="predicate", tail_column="object", backend="native", method="transe", embedding_dim=8, epochs=5, random_state=42)',
            'print(session.kg.query(mode="neighbors", entity="paris"))',
            'print(session.kg.query(mode="typed", entity="paris", relation="linked_to"))',
            'print(session.kg.query(mode="path", source="paris", target="berlin", max_hops=4))',
        ),
        check=(
            "Do you need what is recorded, or what is plausible?",
            "How deep can a path search go before it becomes too slow on your graph?",
        ),
        tools=("query_kg", "predict_links", "fit_kg", "score_triples"),
        terms=("knowledge graph", "graph", "node", "edge", "link prediction"),
        difficulty=CORE,
    ),
    _layer(
        "kg-bundle-boundary",
        plain=(
            "The knowledge-graph plan: embeddings, entity and relation vocabularies, and the training "
            "adjacency: saves as its own bundle. It is distinct from Session checkpoints, from graph-ML "
            "bundles, and from RAG bundles, even though all four involve some notion of connected information."
        ),
        analogy=(
            "A library catalogue, a reading list, a card index, and a filing cabinet all organize "
            "information. Handing someone the wrong one does not help them find the book."
        ),
        steps=(
            "Fit a knowledge-graph model so a plan exists.",
            "Call `session.kg.save_bundle(path)` to store the embeddings, vocabularies, and adjacency.",
            "Reload with `session.kg.load_bundle(path)`.",
            "Score, predict, or query against the restored plan.",
            "Keep checkpoints and other domain bundles separate.",
        ),
        use=(
            "When link prediction or querying runs in a service outside your notebook.",
            "When the entity vocabulary must be pinned so entity identifiers stay stable across versions.",
        ),
        avoid=(
            "Do not mix this up with graph ML, which classifies nodes in a feature graph rather than completing triples.",
            "Do not extend the vocabulary by editing the bundle; new entities need a refit.",
        ),
        myths=(
            (
                "Graph ML and knowledge graphs are the same surface.",
                "Graph ML predicts labels for nodes that carry features. Knowledge-graph embedding predicts whether a relation holds between two entities. Different inputs, different outputs, different bundles.",
            ),
            (
                "New entities can be added incrementally.",
                "Each entity needs a learned embedding. A new one has none, so it needs retraining or a dedicated cold-start strategy.",
            ),
        ),
        example=(
            "from pathlib import Path",
            "import numpy as np",
            "import pandas as pd",
            "from buildml import Session",
            "",
            "rng = np.random.default_rng(42)",
            'Path("artifacts").mkdir(exist_ok=True)',
            "# A small artificial graph demonstrates the API, not link-prediction quality.",
            'entities = ["paris", "berlin", "rome", "madrid", "france", "germany", "italy", "spain"]',
            'triples = [(entities[i], "linked_to", entities[(i + offset) % 8]) for offset in (1, 2, 3) for i in range(8)]',
            'frame = pd.DataFrame(triples, columns=["subject", "predicate", "object"])',
            'session = Session.ingest(frame).set_roles({"subject": "id", "predicate": "id", "object": "id"})',
            "session.inject_split(train_indices=list(range(16)), validation_indices=list(range(16, 20)), test_indices=list(range(20, 24)))",
            'session.kg.fit(head_column="subject", relation_column="predicate", tail_column="object", backend="native", method="transe", embedding_dim=8, epochs=5, random_state=42)',
            'session.kg.save_bundle("artifacts/knowledge-graph")',
            'service = Session().kg.load_bundle("artifacts/knowledge-graph", trusted=True)',
            'print(service.kg.predict_links(heads=["paris"], relations=["linked_to"], k=3).predictions)',
        ),
        check=(
            "Which of the four bundle types does your question actually need?",
            "How often do new entities appear, and what is your plan for them?",
        ),
        tools=("save_kg_bundle", "load_kg_bundle", "predict_links", "checkpoint_save"),
        terms=("bundle", "checkpoint", "knowledge graph", "graph", "RAG"),
        difficulty=CORE,
    ),
)

__all__ = ["KG_BEGINNER"]
