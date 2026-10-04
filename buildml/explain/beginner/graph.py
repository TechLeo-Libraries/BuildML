# ruff: noqa: E501
"""Beginner layers for graph machine learning."""

from __future__ import annotations

from buildml.explain.beginner._builder import ADVANCED, CORE, BeginnerLayer, _index, _layer

GRAPH_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "graph-data-model",
        plain=(
            "Graph machine learning needs two things: a table where each row is an entity (a node) and a "
            "separate list of connections between them (edges). BuildML keeps them separate: your Session "
            "frame holds the node features, and you attach the edge list with `session.graph.set_spec`."
        ),
        analogy=(
            "A staff directory and an org chart. The directory lists everyone's details; the chart says who "
            "reports to whom. You need both to reason about the organization."
        ),
        steps=(
            "Make sure each row has a stable identifier column: that is the node ID.",
            "Prepare an edge list: two columns naming the source and target node IDs.",
            "Call `session.graph.set_spec(edges, node_id_col=...)` to attach it.",
            "BuildML checks that edge endpoints refer to real nodes.",
            "Now graph operations can combine each node's own features with information from its neighbours.",
        ),
        use=(
            "When relationships genuinely carry signal: fraud rings, citation networks, social influence, supply chains.",
            "When a node's neighbours tell you something its own attributes do not.",
        ),
        avoid=(
            "Do not reach for graph methods when your rows are independent; you add substantial complexity for nothing.",
            "Do not use this as a graph database: BuildML does machine learning on graphs, it does not store or query them at scale.",
        ),
        myths=(
            (
                "Graph machine learning requires a graph database.",
                "It requires a node table and an edge list. Those are two ordinary dataframes.",
            ),
            (
                "Any dataset with relationships needs graph methods.",
                "If the relationship can be summarized into a column: 'number of connections', 'household size': an ordinary model with that column is simpler and often just as good.",
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
            '# Requires: python -m pip install "buildml[graph]"',
            "n = 60",
            'nodes_frame = pd.DataFrame({"account_id": np.arange(n), "x1": rng.normal(size=n), "x2": rng.normal(size=n)})',
            'nodes_frame["is_fraud"] = (nodes_frame.x1 + nodes_frame.x2 > 0).astype(int)',
            'edge_frame = pd.DataFrame([(i, (i + offset) % n) for i in range(n) for offset in (1, 2, 5)], columns=["source", "target"])',
            'session = Session.ingest(nodes_frame).set_roles({"account_id": "id", "x1": "feature", "x2": "feature", "is_fraud": "target"})',
            "session.split(test_size=0.2, validation_size=0.2, stratify=True, random_state=42)",
            'session.graph.set_spec(edges=edge_frame, node_id_col="account_id")',
            'session.graph.fit(method="classical", random_state=42)',
            'print(session.graph.evaluate(partition="validation").metrics)',
        ),
        check=(
            "Does every edge endpoint correspond to a row in your node table?",
            "Would a simple 'degree' column capture most of what the graph offers?",
        ),
        tools=("set_graph", "fit_graph", "predict_graph", "evaluate_graph"),
        terms=("graph", "node", "edge", "network"),
        difficulty=CORE,
    ),
    _layer(
        "graph-inductive-transductive",
        plain=(
            "Two ways to split a graph, and they mean different things. Inductive hides the evaluation "
            "nodes entirely during training: the model never sees them or their connections. Transductive "
            "lets the model see the whole structure but hides the evaluation nodes' labels."
        ),
        analogy=(
            "Inductive: training on one office and being tested on a branch you have never visited. "
            "Transductive: you have walked the whole building and know the layout: you just have not been "
            "told what happens in certain rooms."
        ),
        steps=(
            "Decide which situation matches deployment.",
            "For inductive, BuildML fits on the subgraph induced by the training nodes only, dropping edges that reach out of it.",
            "For transductive, the full topology is visible during training but only training-node labels supervise the loss.",
            "Score the held-out nodes.",
            "State which mode you used, because the two are not comparable.",
        ),
        use=(
            "Inductive when new nodes will arrive after deployment: new users, new accounts, new products.",
            "Transductive when the graph is fixed and you are filling in missing labels within it.",
        ),
        avoid=(
            "Do not report a transductive score as evidence the model will handle new nodes; it never had to.",
            "Inspect isolated nodes after an inductive split and compare against a model using node features alone.",
        ),
        myths=(
            (
                "Transductive learning leaks.",
                "It uses structure, not labels, from the evaluation nodes. That is legitimate *if* deployment also has the full graph. If new nodes arrive later, it is over-optimistic.",
            ),
            (
                "Inductive and transductive scores are comparable.",
                "The protocols expose different information during training. Neither guarantees a higher score; compare results only when their deployment assumptions and evaluation protocols align.",
            ),
        ),
        example=(
            '# Requires: python -m pip install "buildml[torch]"',
            "from pathlib import Path",
            "import numpy as np",
            "import pandas as pd",
            "from buildml import Session",
            "",
            "rng = np.random.default_rng(42)",
            'Path("artifacts").mkdir(exist_ok=True)',
            '# Requires: python -m pip install "buildml[graph]"',
            "n = 60",
            'nodes_frame = pd.DataFrame({"account_id": np.arange(n), "x1": rng.normal(size=n), "x2": rng.normal(size=n)})',
            'nodes_frame["is_fraud"] = (nodes_frame.x1 + nodes_frame.x2 > 0).astype(int)',
            'edge_frame = pd.DataFrame([(i, (i + offset) % n) for i in range(n) for offset in (1, 2, 5)], columns=["source", "target"])',
            'session = Session.ingest(nodes_frame).set_roles({"account_id": "id", "x1": "feature", "x2": "feature", "is_fraud": "target"})',
            "session.split(test_size=0.2, validation_size=0.2, stratify=True, random_state=42)",
            'session.graph.set_spec(edges=edge_frame, node_id_col="account_id")',
            'session.graph.fit(method="gcn", mode="inductive", epochs=3, random_state=42)',
            'print(session.graph.evaluate(partition="test").metrics)',
            "print(session.graph.plan.mode, session.graph.plan.disclosures)",
        ),
        check=(
            "Will new nodes appear after deployment?",
            "How many edges did inductive splitting have to drop?",
        ),
        tools=("fit_graph", "evaluate_graph", "set_graph", "split"),
        terms=("inductive", "transductive", "graph", "node", "leakage"),
        difficulty=ADVANCED,
    ),
    _layer(
        "graph-classical-features",
        plain=(
            "You do not need a neural network to use a graph. Compute a handful of classical structural "
            "measures for each node: how many connections it has, how tightly its neighbours interconnect, "
            "how central it is: append them as columns, and feed the result to any ordinary model."
        ),
        analogy=(
            "Describing someone by how many colleagues they have, whether their colleagues know each other, "
            "and how many messages flow through them. A few numbers, and an ordinary model can use them."
        ),
        steps=(
            "BuildML computes node-level metrics with NetworkX: degree, clustering coefficient, PageRank, betweenness, and similar.",
            "Those metrics become extra numeric columns beside your existing features.",
            "Fit an ordinary scikit-learn classifier on the combined table.",
            "Read feature importance to see whether the structural columns actually mattered.",
            "Compute the metrics under your split discipline so evaluation nodes do not shape training features.",
        ),
        use=(
            "As your first graph attempt: it is fast, interpretable, and often captures most of the available signal.",
            "When your graph is small enough for exact centrality computation.",
        ),
        avoid=(
            "Do not use it when the signal lies in multi-hop patterns that summary statistics cannot express; that is where graph neural networks earn their cost.",
            "Do not compute betweenness on a very large graph: it is expensive and will dominate your runtime.",
        ),
        myths=(
            (
                "Graph neural networks always beat classical features.",
                "On small or moderately connected graphs, degree plus PageRank plus a gradient-boosting model is a very strong and much cheaper baseline.",
            ),
            (
                "Structural features are safe from leakage.",
                "PageRank computed over the full graph absorbs the structure of evaluation nodes. Under inductive assumptions, that is leakage.",
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
            '# Requires: python -m pip install "buildml[graph]"',
            "n = 60",
            'nodes_frame = pd.DataFrame({"account_id": np.arange(n), "x1": rng.normal(size=n), "x2": rng.normal(size=n)})',
            'nodes_frame["is_fraud"] = (nodes_frame.x1 + nodes_frame.x2 > 0).astype(int)',
            'edge_frame = pd.DataFrame([(i, (i + offset) % n) for i in range(n) for offset in (1, 2, 5)], columns=["source", "target"])',
            'session = Session.ingest(nodes_frame).set_roles({"account_id": "id", "x1": "feature", "x2": "feature", "is_fraud": "target"})',
            "session.split(test_size=0.2, validation_size=0.2, stratify=True, random_state=42)",
            'session.graph.set_spec(edges=edge_frame, node_id_col="account_id")',
            'session.graph.fit(method="classical", include_graph_metrics=True, classical_estimator="random_forest", random_state=42)',
            'print(session.graph.evaluate(partition="validation").metrics)',
        ),
        check=(
            "Do the structural columns appear in your top feature importances?",
            "Over which nodes were your centrality measures computed?",
        ),
        tools=("fit_graph", "evaluate_graph", "feature_importance", "set_graph"),
        terms=("graph", "PageRank", "node", "feature importance"),
        difficulty=CORE,
    ),
    _layer(
        "graph-pyg",
        plain=(
            "PyTorch Geometric provides graph neural network layers. With the optional extra installed, BuildML can build GCN, GraphSAGE, or GAT models through it: architectures that let each node's prediction depend on a learned combination of its neighbours."
        ),
        analogy=(
            "Rather than counting how many colleagues someone has, you learn what to take from each "
            "colleague, and then what to take from *their* colleagues. Depth lets influence travel."
        ),
        steps=(
            "Install `pip install buildml[graph-pyg]`.",
            "Choose GCN for degree-normalized aggregation, GraphSAGE for learned neighborhood aggregation, or GAT for attention-weighted aggregation.",
            "Set `n_layers` to 1 or 2, the values supported by this adapter. Each layer adds one message-passing step.",
            "Train with a mask so only training-node labels contribute to the loss.",
            "Evaluate on the held-out node mask.",
        ),
        use=(
            "When multi-hop structure genuinely matters and classical features have plateaued.",
            "When you want to compare PyG convolution layers on a graph that fits in memory. This adapter processes the graph in full; it does not expose neighbor sampling.",
        ),
        avoid=(
            "The adapter supports one or two layers. More message passing is not automatically better; aggregation can make node representations less distinct.",
            "Do not use it on a graph with very few labelled nodes; graph neural networks are data-hungry like any neural network.",
        ),
        myths=(
            (
                "More layers means more context and better results.",
                "Each layer widens the receptive field and blurs it. Over-smoothing means deep graph networks often perform worse than shallow ones.",
            ),
            (
                "Graph neural networks understand the graph.",
                "They learn to aggregate neighbour features. If your edges are noisy or meaningless, aggregation spreads the noise rather than filtering it.",
            ),
        ),
        example=(
            '# Requires: python -m pip install "buildml[graph-pyg]"',
            "from pathlib import Path",
            "import numpy as np",
            "import pandas as pd",
            "from buildml import Session",
            "",
            "rng = np.random.default_rng(42)",
            'Path("artifacts").mkdir(exist_ok=True)',
            '# Requires: python -m pip install "buildml[graph]"',
            "n = 60",
            'nodes_frame = pd.DataFrame({"account_id": np.arange(n), "x1": rng.normal(size=n), "x2": rng.normal(size=n)})',
            'nodes_frame["is_fraud"] = (nodes_frame.x1 + nodes_frame.x2 > 0).astype(int)',
            'edge_frame = pd.DataFrame([(i, (i + offset) % n) for i in range(n) for offset in (1, 2, 5)], columns=["source", "target"])',
            'session = Session.ingest(nodes_frame).set_roles({"account_id": "id", "x1": "feature", "x2": "feature", "is_fraud": "target"})',
            "session.split(test_size=0.2, validation_size=0.2, stratify=True, random_state=42)",
            'session.graph.set_spec(edges=edge_frame, node_id_col="account_id")',
            'session.graph.fit(method="pyg", pyg_model="graphsage", n_layers=2, hidden_dim=16, epochs=3, random_state=42)',
            'print(session.graph.evaluate(partition="test").metrics)',
        ),
        check=(
            "How many hops away is the information you believe matters?",
            "How many labelled nodes are in your training mask?",
        ),
        tools=("fit_graph", "evaluate_graph", "predict_graph", "set_graph"),
        terms=("GNN", "graph", "node", "neural network", "extra"),
        difficulty=ADVANCED,
    ),
    _layer(
        "graph-gcn",
        plain=(
            "BuildML also ships a compact graph convolutional network written directly in PyTorch, with no PyTorch Geometric required. It is a one- or two-layer GCN using a normalized adjacency matrix."
        ),
        analogy=(
            "A simple recipe with three ingredients that gets you most of the way, rather than the "
            "professional kitchen version that needs specialist equipment."
        ),
        steps=(
            "The adjacency matrix is normalized so nodes with many connections do not dominate.",
            "Each layer mixes a node's own features with the average of its neighbours' features.",
            "One layer sees direct neighbours; two layers see neighbours of neighbours.",
            "Train with a mask so only training nodes contribute to the loss.",
            "Evaluate on the held-out mask exactly as with the PyG path.",
        ),
        use=(
            "When you want a graph neural network without adding the PyTorch Geometric dependency.",
            "On moderately sized graphs where a dense adjacency matrix still fits in memory.",
        ),
        avoid=(
            "Do not use it on very large graphs; the dense normalized adjacency does not scale the way sampled approaches do.",
            "Use the PyG adapter if you need GraphSAGE or GAT; this implementation supports GCN.",
        ),
        myths=(
            (
                "A hand-written GCN is a toy.",
                "GCN is the standard baseline in the literature and frequently competitive with more elaborate architectures on node classification.",
            ),
            (
                "Adjacency normalization is a detail.",
                "Without it, high-degree nodes swamp the aggregation and training becomes unstable. It is central to why GCN works.",
            ),
        ),
        example=(
            '# Requires: python -m pip install "buildml[torch]"',
            "from pathlib import Path",
            "import numpy as np",
            "import pandas as pd",
            "from buildml import Session",
            "",
            "rng = np.random.default_rng(42)",
            'Path("artifacts").mkdir(exist_ok=True)',
            '# Requires: python -m pip install "buildml[graph]"',
            "n = 60",
            'nodes_frame = pd.DataFrame({"account_id": np.arange(n), "x1": rng.normal(size=n), "x2": rng.normal(size=n)})',
            'nodes_frame["is_fraud"] = (nodes_frame.x1 + nodes_frame.x2 > 0).astype(int)',
            'edge_frame = pd.DataFrame([(i, (i + offset) % n) for i in range(n) for offset in (1, 2, 5)], columns=["source", "target"])',
            'session = Session.ingest(nodes_frame).set_roles({"account_id": "id", "x1": "feature", "x2": "feature", "is_fraud": "target"})',
            "session.split(test_size=0.2, validation_size=0.2, stratify=True, random_state=42)",
            'session.graph.set_spec(edges=edge_frame, node_id_col="account_id")',
            'session.graph.fit(method="gcn", n_layers=2, hidden_dim=16, epochs=3, random_state=42)',
            'print(session.graph.evaluate(partition="validation").metrics)',
        ),
        check=(
            "How many nodes does your graph have, and will a dense adjacency fit?",
            "Does two-hop information help, or is one layer enough?",
        ),
        tools=("fit_graph", "evaluate_graph", "predict_graph", "set_graph"),
        terms=("GNN", "graph", "node", "neural network"),
        difficulty=ADVANCED,
    ),
    _layer(
        "graph-bundle-boundary",
        plain=(
            "The fitted graph model saves as a graph bundle holding the model, the node feature contract, "
            "and the split mode. Session checkpoints hold your node table and workflow state, not the graph "
            "plan."
        ),
        analogy=(
            "The org chart analysis you produced is a different document from the staff directory it was "
            "based on."
        ),
        steps=(
            "Fit a graph model so a plan exists.",
            "Call `session.graph.save_bundle(path)`.",
            "Attach the graph with `session.graph.set_spec`, then load a trusted bundle with `session.graph.load_bundle(path, trusted=True)`. Setting a graph spec clears any previously loaded plan.",
            "Predict for nodes, remembering that inductive and transductive plans expect different things.",
            "Keep checkpoints separate for the node data.",
        ),
        use=(
            "When node scoring runs on a schedule against a refreshed graph.",
            "When the split mode must travel with the model so its scores stay interpretable.",
        ),
        avoid=(
            "Do not apply a transductive plan to a graph with new nodes without re-reading its disclosures.",
            "Do not assume the bundle contains the edge list; you supply the graph at load time.",
        ),
        myths=(
            (
                "The bundle stores the graph.",
                "It stores the fitted model and its contract. The graph is data, and it changes; that is why you attach it fresh.",
            ),
            (
                "Any graph with matching node IDs will work.",
                "The model's behaviour depends on the structure it was trained under. A radically different topology gives predictions you have not validated.",
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
            '# Requires: python -m pip install "buildml[graph]"',
            "n = 60",
            'nodes_frame = pd.DataFrame({"account_id": np.arange(n), "x1": rng.normal(size=n), "x2": rng.normal(size=n)})',
            'nodes_frame["is_fraud"] = (nodes_frame.x1 + nodes_frame.x2 > 0).astype(int)',
            'edge_frame = pd.DataFrame([(i, (i + offset) % n) for i in range(n) for offset in (1, 2, 5)], columns=["source", "target"])',
            'session = Session.ingest(nodes_frame).set_roles({"account_id": "id", "x1": "feature", "x2": "feature", "is_fraud": "target"})',
            "session.split(test_size=0.2, validation_size=0.2, stratify=True, random_state=42)",
            'session.graph.set_spec(edges=edge_frame, node_id_col="account_id")',
            'session.graph.fit(method="classical", random_state=42)',
            'session.graph.save_bundle("artifacts/graph-model")',
            "job = Session.ingest(nodes_frame).set_roles(dict(session.dataset.roles))",
            'job.graph.set_spec(edges=edge_frame, node_id_col="account_id")',
            'job.graph.load_bundle("artifacts/graph-model", trusted=True)',
            'print(job.graph.predict(partition="all"))',
        ),
        check=(
            "Was your plan fitted inductively or transductively, and does today's graph match that assumption?",
            "Where does the edge list come from at scoring time?",
        ),
        tools=("save_graph_bundle", "load_graph_bundle", "set_graph", "checkpoint_save"),
        terms=("bundle", "checkpoint", "graph", "inductive", "transductive"),
        difficulty=CORE,
    ),
)

__all__ = ["GRAPH_BEGINNER"]
