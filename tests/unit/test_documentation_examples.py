"""Documentation examples must provide their own imports and data."""

import ast

import pytest

from scripts.check_documentation_examples import (
    PYTHON,
    ROOT,
    Example,
    blocks,
    execute,
    inventory,
    python_source,
    static_check,
)
from scripts.check_teaching_examples import _attribute_path, _split_receivers


def test_teaching_contract_tracks_fluent_splits_and_excludes_dataframe_methods():
    tree = ast.parse("session = Session.ingest(frame).set_roles(roles).group_split(test_size=0.2)\n")
    assert "session" in _split_receivers(tree)
    call = ast.parse("session.to_pandas().drop(columns='target')", mode="eval").body
    assert _attribute_path(call.func) is None
    assert _attribute_path(call.func.value.func) == ["session", "to_pandas"]


def test_missing_imports_in_standalone_cv_example_are_reported():
    result = static_check("session = Session.ingest(frame)\nmodel = LogisticRegression()")
    assert result["status"] == "undefined-names"
    assert result["undefined_names"] == ["LogisticRegression", "Session", "frame"]


def test_nested_global_references_are_checked_without_flagging_locals():
    result = static_check("def run(items):\n    return [unknown(item) for item in items]\n")
    assert result["undefined_names"] == ["unknown"]


def test_fence_and_rst_inventory_keeps_non_python_blocks():
    found = blocks("```bash\npip install buildml\n```\n\n.. code-block:: python\n\n   print(1)\n", "guide.md")
    assert [(e.language, e.source) for e in found] == [("bash", "pip install buildml"), ("python", "print(1)")]


def test_each_execution_starts_with_empty_globals():
    first = Example("one.md", 1, "python", "remembered = 42", "fence")
    second = Example("two.md", 1, "python", "print(remembered)", "fence")
    assert execute(first, 10)["status"] == "passed"
    assert execute(second, 10)["status"] == "failed"


def test_rst_literal_examples_are_inventoried_without_losing_unknown_blocks():
    found = blocks("Example::\n\n    import pandas as pd\n    frame = pd.DataFrame()\n\nOutput::\n\n    +---+\n    | x |\n    +---+\n", "guide.rst")
    assert [(e.kind, e.language) for e in found] == [("literal", "python"), ("literal", "")]
    assert found[0].line == 3
    assert static_check(found[0].source)["status"] == "static-pass"


def test_invalid_console_blocks_cannot_pass_as_empty_scripts():
    for example in (
        Example("api.py", 1, "pycon", "malformed doctest", "invalid-doctest"),
        Example("guide.md", 1, "pycon", "not a console session", "fence"),
    ):
        with pytest.raises(ValueError):
            python_source(example)


README_EXAMPLES = [e for e in inventory([ROOT / "README.md"]) if e.language in PYTHON]


@pytest.mark.parametrize("example", README_EXAMPLES, ids=lambda e: f"readme-line-{e.line}")
def test_readme_python_block_runs_independently(example):
    assert static_check(example.source)["status"] == "static-pass"
    result = execute(example, 90)
    assert result["status"] == "passed", result
