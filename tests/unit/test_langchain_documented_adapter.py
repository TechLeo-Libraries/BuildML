"""The advertised optional extra includes the QA chain used by the adapter."""

import importlib.util

import pytest

from scripts.check_documentation_examples import ROOT, execute, inventory


@pytest.mark.skipif(importlib.util.find_spec("langchain_classic") is None, reason="Requires buildml[rag-advanced]")
def test_langchain_docstring_example_runs_without_a_provider_account():
    example = next(
        example for example in inventory([ROOT / "buildml/rag/adapters/langchain.py"])
        if "FakeListLLM" in example.source
    )
    result = execute(example, timeout=180)
    assert result["status"] == "passed", result
    assert "Cancel from account settings." in result["stdout"]


def test_langchain_chat_provider_uses_installed_classic_chain():
    pytest.importorskip("langchain_classic")
    pytest.importorskip("langchain_community")
    from types import SimpleNamespace

    from langchain_core.language_models.fake import FakeListLLM

    from buildml.rag.adapters.langchain import LangChainGroundedAdapter

    adapter = LangChainGroundedAdapter(FakeListLLM(responses=["Refunds take 30 days."]))
    response = adapter.as_chat_provider().chat([
        SimpleNamespace(role="system", content="Refunds take 30 days."),
        SimpleNamespace(role="user", content="How long do refunds take?"),
    ])
    assert response.content == "Refunds take 30 days."
