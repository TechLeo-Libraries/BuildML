# ruff: noqa: E501
"""Graph ML concept notes (node classification + leakage modes)."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

GRAPH_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="graph-data-model",
            title="Session rows are nodes; edges are attached separately",
            summary="Graph ML uses a node feature table plus an edge list keyed by node_id: not a Neo4j product.",
            definition=(
                "Each Session dataset row is a node. set_graph attaches an edge "
                "list whose endpoints match a unique node_id column. "
                "Session.split creates node partitions."
            ),
            intuition=(
                "Think of a spreadsheet of people (nodes) and a separate friend "
                "list (edges). You split people into train/val/test, then learn "
                "labels using both their attributes and who they connect to."
            ),
            formal_idea=(
                "G=(V,E) with node features X_v and labels Y_v for a subset of V."
            ),
            why_it_matters=(
                "Without a clear node/edge convention, leakage and id mismatches are silent.",
                "Triple-based KG learning is a separate Session path (buildml.kg), not Neo4j.",
            ),
            how_buildml_uses=(
                "Session.set_graph → fit_graph / predict_graph / evaluate_graph.",
            ),
            interpretation_rules=(
                "Read n_edges and disclosures after set_graph.",
                "node_id must be unique and match edge endpoints.",
            ),
            assumptions=("Caller supplies a coherent edge list for the node table.",),
            failure_modes=("Orphan edges; duplicate node ids; missing node_id column.",),
            anti_patterns=(
                "Treating this as a knowledge-graph database or fit_kg triples path.",
                "Using row index as edge endpoints without a stable id column.",
            ),
            worked_example_pattern=(
                "set_graph(edges_df, node_id_col='node_id') → "
                "fit_graph(method='classical', mode='inductive').",
            ),
            related_concepts=(
                "graph-inductive-transductive",
                "graph-classical-features",
                "graph-gcn",
                "graph-pyg",
                "graph-bundle-boundary",
                "leakage-boundary",
            ),
        ),
        _note(
            key="graph-inductive-transductive",
            title="Inductive vs transductive graph splits",
            summary="Default inductive fits on the train-induced subgraph; transductive uses full topology with train-label-only supervision.",
            definition=(
                "Inductive: message-passing / NetworkX metrics for fit use only "
                "edges with both endpoints in train. Scoring may use train↔holdout "
                "edges; holdout↔holdout edges are dropped. Transductive: full "
                "graph topology participates; labels still train-only."
            ),
            intuition=(
                "Inductive asks: can we generalize to new nodes without seeing "
                "their mutual edges at train time? Transductive is the classic "
                "semi-supervised GNN setting on one fixed graph."
            ),
            formal_idea=(
                "Inductive: A_fit = A[train,train]. Transductive: A_full with "
                "loss on train mask only."
            ),
            why_it_matters=(
                "Using test edges during train message passing is a common silent leak.",
            ),
            how_buildml_uses=(
                "fit_graph(mode='inductive'|'transductive') records disclosures.",
            ),
            interpretation_rules=(
                "Read mode and disclosures before comparing numbers across papers.",
            ),
            assumptions=("Node split is created before fit_graph.",),
            failure_modes=("Inductive with no train–train edges.",),
            anti_patterns=(
                "Calling a full-graph train 'inductive'.",
                "Using holdout labels in the loss.",
            ),
            worked_example_pattern=(
                "fit_graph(method='gcn', mode='inductive') → evaluate_graph('test').",
            ),
            related_concepts=(
                "graph-data-model",
                "graph-gcn",
                "leakage-boundary",
            ),
        ),
        _note(
            key="graph-classical-features",
            title="Classical graph metrics + sklearn on nodes",
            summary="NetworkX degree/clustering/PageRank/etc. concatenated with tabular node features, then a sklearn classifier.",
            definition=(
                "Classical path requires buildml[graph] (NetworkX). Metrics are "
                "computed on the mode-filtered edge set and concatenated with "
                "numeric node features for logistic regression or random forest."
            ),
            intuition=(
                "Hand-engineered topology stats plus node attributes, then a "
                "normal classifier: no neural message passing."
            ),
            formal_idea="x_v = [x_tab_v || φ(G)_v]; predict with sklearn.",
            why_it_matters=(
                "Gives a dependency-light baseline that still respects graph structure.",
            ),
            how_buildml_uses=(
                "fit_graph(method='classical', include_graph_metrics=True).",
            ),
            interpretation_rules=(
                "Betweenness is skipped for large graphs (n>200).",
            ),
            assumptions=("NetworkX optional extra installed.",),
            failure_modes=("Missing buildml[graph]; no features and metrics disabled.",),
            anti_patterns=("Computing global centrality on the full graph while claiming inductive.",),
            worked_example_pattern=(
                "pip install 'buildml[graph]'; fit_graph(method='classical').",
            ),
            related_concepts=(
                "graph-data-model",
                "graph-inductive-transductive",
                "graph-bundle-boundary",
            ),
        ),
        _note(
            key="graph-pyg",
            title="PyTorch Geometric GCN / GraphSAGE / GAT",
            summary=(
                "Industry GNN path via torch_geometric.nn with train-mask "
                "cross-entropy: requires buildml[graph-pyg]."
            ),
            definition=(
                "PyG path requires buildml[graph-pyg] (torch-geometric + torch). "
                "Supports GCNConv, SAGEConv, GATConv via pyg_model= "
                "gcn|graphsage|gat. Same inductive/transductive edge filters "
                "as classical and pure-Torch GCN."
            ),
            intuition=(
                "Use PyG when you want standard conv layers and sparse "
                "edge_index message passing without reimplementing GNN blocks."
            ),
            formal_idea="H' = Conv(H, edge_index); loss on train mask only.",
            why_it_matters=(
                "Bridges Session graph ML to industry PyG stacks while "
                "preserving leakage disclosures.",
            ),
            how_buildml_uses=(
                "fit_graph(method='pyg', pyg_model='graphsage', mode='inductive').",
            ),
            interpretation_rules=(
                "Read pyg_model and method=pyg in disclosures; not full PyG zoo.",
            ),
            assumptions=("Numeric node features; graph-pyg extra installed.",),
            failure_modes=("Missing graph-pyg; broken torch wheel.",),
            anti_patterns=(
                "Expecting every PyG algorithm (GIN, PNA, etc.) on this surface.",
                "Ignoring inductive vs transductive mode.",
            ),
            worked_example_pattern=(
                "pip install 'buildml[graph-pyg]'; "
                "fit_graph(method='pyg', pyg_model='gat', heads=4).",
            ),
            related_concepts=(
                "graph-inductive-transductive",
                "graph-gcn",
                "graph-bundle-boundary",
                "leakage-boundary",
            ),
        ),
        _note(
            key="graph-gcn",
            title="Pure-Torch GCN (no PyTorch Geometric)",
            summary="A 1–2 layer GCN with symmetric normalized adjacency implemented in Torch: PyG is optional via method='pyg'.",
            definition=(
                "GCN path requires buildml[torch]. Uses D^{-1/2}(A+I)D^{-1/2} "
                "and train-mask cross-entropy. Dense adjacency is limited to "
                "≤5000 nodes in this Session surface."
            ),
            intuition=(
                "Each layer mixes a node's features with its neighbors, then a "
                "linear map: Kipf & Welling style, small and honest."
            ),
            formal_idea="H' = Â H W with Â = D^{-1/2}(A+I)D^{-1/2}.",
            why_it_matters=(
                "Avoids PyG's heavy CUDA/Torch coupling while still shipping a real GNN path.",
            ),
            how_buildml_uses=("fit_graph(method='gcn', epochs=..., hidden_dim=...).",),
            interpretation_rules=(
                "Read train_loss_last and train_accuracy; not a research zoo.",
            ),
            assumptions=("Numeric tabular node features present; torch installed.",),
            failure_modes=("Missing torch; zero feature columns; huge graphs.",),
            anti_patterns=(
                "Calling this the only neural graph path (see graph-pyg for PyG).",
                "Ignoring inductive vs transductive disclosures.",
            ),
            worked_example_pattern=(
                "pip install 'buildml[torch]'; fit_graph(method='gcn', mode='inductive').",
            ),
            related_concepts=(
                "graph-inductive-transductive",
                "graph-pyg",
                "graph-bundle-boundary",
                "leakage-boundary",
            ),
        ),
        _note(
            key="graph-bundle-boundary",
            title="Graph bundles are not Session checkpoints",
            summary="buildml.graph_bundle.v1 stores GraphPlan separately from Session checkpoints.",
            definition=(
                "save_graph_bundle / load_graph_bundle persist GraphSpec + fitted "
                "estimator/GCN. Session checkpoints do not embed the graph learner."
            ),
            intuition=(
                "Reload your table workflow from a checkpoint; reload the graph "
                "model from its own bundle."
            ),
            formal_idea="Distinct artifact formats; complementary, not interchangeable.",
            why_it_matters=("Prevents silent format confusion across domains.",),
            how_buildml_uses=(
                "Session.save_graph_bundle / load_graph_bundle.",
            ),
            interpretation_rules=("Confirm meta.json format buildml.graph_bundle.v1.",),
            assumptions=("Bundle directory writable/readable.",),
            failure_modes=("Missing meta.json or graph_plan.joblib.",),
            anti_patterns=("Expecting checkpoint_load to restore GraphPlan.",),
            worked_example_pattern=(
                "save_graph_bundle('artifacts/graph_bundle') → load_graph_bundle(...).",
            ),
            related_concepts=("graph-data-model", "leakage-boundary"),
        ),
    )
}
