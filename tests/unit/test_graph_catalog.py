"""Unit tests for graph capability catalog."""

from __future__ import annotations

from buildml.graph.catalog import graph_capability_matrix


def test_graph_capability_matrix_shape() -> None:
    matrix = graph_capability_matrix()
    assert set(matrix["backends"]) == {"classical", "gcn", "pyg"}
    assert "node_classification" in matrix["tasks"]
    assert "install_hints" in matrix
    assert matrix["install_hints"]["graph-pyg"].startswith("pip install")
