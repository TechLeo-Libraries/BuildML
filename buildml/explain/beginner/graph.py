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
            "session.set_roles({'account_id': 'id', 'is_fraud': 'target'})",
            "session.graph.set_spec(edges=edge_frame, node_id_col='account_id')",
            "session.graph.fit(method='classical', random_state=0)",
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
            "Do not use inductive splitting on a graph so sparse that removing cross-boundary edges leaves isolated nodes.",
        ),
        myths=(
            (
                "Transductive learning leaks.",
                "It uses structure, not labels, from the evaluation nodes. That is legitimate *if* deployment also has the full graph. If new nodes arrive later, it is over-optimistic.",
            ),
            (
                "Inductive and transductive scores are comparable.",
                "Transductive is systematically higher because the model had more information. Comparing them across papers or experiments is a common mistake.",
            ),
        ),
        example=(
            "session.graph.fit(method='gcn', mode='inductive', random_state=0)",
            "session.graph.evaluate(partition='test')",
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
            "session.graph.fit(",
            "    method='classical',",
            "    include_graph_metrics=True,",
            "    classical_estimator='random_forest',",
            "    random_state=0,",
            ")",
            "session.graph.evaluate(partition='validation')",
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
            "PyTorch Geometric is the standard library for graph neural networks. With the optional extra "
            "installed, BuildML can build GCN, GraphSAGE, or GAT models through it: architectures that let "
            "each node's prediction depend on a learned combination of its neighbours."
        ),
        analogy=(
            "Rather than counting how many colleagues someone has, you learn what to take from each "
            "colleague, and then what to take from *their* colleagues. Depth lets influence travel."
        ),
        steps=(
            "Install `pip install buildml[graph-pyg]`.",
            "Choose an architecture: GCN averages neighbours uniformly, GraphSAGE samples them, GAT learns attention weights over them.",
            "Set the number of layers: this is how many hops of influence the model can see. Two is typical.",
            "Train with a mask so only training-node labels contribute to the loss.",
            "Evaluate on the held-out node mask.",
        ),
        use=(
            "When multi-hop structure genuinely matters and classical features have plateaued.",
            "On large graphs where GraphSAGE's neighbour sampling makes training feasible.",
        ),
        avoid=(
            "Do not stack many layers: beyond three or four, every node's representation converges to the same thing, a failure called over-smoothing.",
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
            "# pip install \"buildml[graph-pyg]\"",
            "session.graph.fit(",
            "    method='pyg', pyg_model='graphsage',",
            "    n_layers=2, hidden_dim=64, epochs=200, random_state=0,",
            ")",
            "session.graph.evaluate(partition='test')",
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
            "BuildML also ships a compact graph convolutional network written directly in PyTorch, with no "
            "PyTorch Geometric required. It is a one- or two-layer GCN using a normalized adjacency matrix "
            ": enough for many node-classification problems and far lighter to install."
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
            "Do not expect the architectural variety of PyG: this is GCN, not a menu of designs.",
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
            "session.graph.fit(",
            "    method='gcn', n_layers=2, hidden_dim=32,",
            "    epochs=200, random_state=0,",
            ")",
            "session.graph.evaluate(partition='validation')",
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
            "Reload with `session.graph.load_bundle(path)` and reattach a graph with `session.graph.set_spec`.",
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
            "session.graph.save_bundle('artifacts/fraud-graph')",
            "job = Session.ingest(nodes_frame).graph.load_bundle('artifacts/fraud-graph')",
            "job.graph.set_spec(edges=todays_edges, node_id_col='account_id')",
            "job.graph.predict()",
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
