"""Run the documented PyG workflow in a process isolated from native crashes."""

import importlib.util

import pytest

from scripts.check_documentation_examples import ROOT, execute, inventory


@pytest.mark.skipif(
    importlib.util.find_spec("torch_geometric") is None,
    reason="Requires buildml[graph-pyg]",
)
def test_pyg_teaching_example_trains_and_evaluates():
    example = next(
        example for example in inventory([ROOT / "buildml/explain/beginner/graph.py"])
        if example.kind == "teaching" and 'method="pyg"' in example.source
    )
    result = execute(example, timeout=240)
    assert result["status"] == "passed", result
