"""Phase C AI depth: RAG/DL tools, multi-step plans, registry completeness."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.results import PlanResult, PlanStep
from buildml.ai.tools import build_default_registry, registered_tool_names
from buildml.rag.generate import EchoGroundedProvider

_TORCH_SPEC = importlib.util.find_spec("torch") is not None


def _require_torch_or_skip() -> None:
    """Skip when Torch is installed but not importable in this process (e.g. AV)."""
    try:
        from buildml.core.errors import MissingExtraError
        from buildml.dl.extras import require_torch

        require_torch(feature="pytest Torch AI Phase C")
    except (MissingExtraError, ImportError, OSError) as exc:
        pytest.skip(f"torch not importable in-process: {exc}")


def test_default_registry_includes_rag_and_dl_tools() -> None:
    names = set(registered_tool_names())
    for required in (
        "rag_retrieve",
        "rag_generate",
        "rag_ingest_corpus",
        "rag_embed_and_index",
        "make_torch_loaders",
        "make_text_torch_loaders",
        "fit_torch",
        "evaluate_torch",
        "cross_validate_torch",
        "fit",
        "evaluate",
        "split",
    ):
        assert required in names


def test_registry_tools_have_session_methods_or_builtin() -> None:
    """Every tool either maps to a Session method or is a known builtin."""
    builtins = {"describe_dataset", "ai_status"}
    session_methods = {name for name in dir(Session) if not name.startswith("_")}
    registry = build_default_registry()
    for tool in registry.tools:
        if tool.name in builtins:
            continue
        method = tool.session_method or tool.name
        assert method in session_methods, f"{tool.name} missing Session.{method}"


def test_ai_execute_rag_generate_flow() -> None:
    docs = [
        "Pandas DataFrames store tabular rows and columns for analysis.",
        "BuildML Session orchestrates ingest, prep, fit, and evaluate.",
    ]
    session = (
        Session()
        .rag_ingest_corpus(docs)
        .rag_embed_and_index(embedder="hashing")
        .ai_configure(provider="mock")
    )
    # Prefer echo grounded provider for deterministic citations in generate tool path.
    session._ai_provider = EchoGroundedProvider()
    proposal = session.ai_execute("rag_retrieve", {"query": "What is Session?", "k": 2})
    # read-only auto-confirms
    assert getattr(proposal, "executed", False) or getattr(proposal, "tool_call", None)
    result = session.ai_execute(
        "rag_generate", {"query": "What does BuildML Session do?", "k": 2}, confirm=True
    )
    assert result.executed
    assert session.rag_generate_result is not None
    assert session.rag_generate_result.n_citations >= 1


def test_ai_run_plan_multi_step_classical() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": rng.normal(size=80),
            "b": rng.normal(size=80),
            "y": np.asarray([0, 1] * 40, dtype=np.int64),
        }
    )
    session = Session.ingest(df).ai_configure(provider="mock")
    plan = PlanResult(
        goal="Prepare and split",
        steps=(
            PlanStep(
                operation="set_roles",
                description="Assign roles",
                rationale="Need target",
                prerequisites=(),
                expected_changes=("roles assigned",),
                parameters={"mapping": {"a": "feature", "b": "feature", "y": "target"}},
            ),
            PlanStep(
                operation="split",
                description="Create split",
                rationale="Need holdout",
                prerequisites=(),
                expected_changes=("split created",),
                parameters={"test_size": 0.25, "random_state": 0},
            ),
        ),
        current_state_summary="raw",
        assumptions=(),
        raw_response="{}",
    )
    result = session.ai_run_plan(
        plan,
        confirmations={0: True, 1: True},
        auto_confirm_read_only=True,
    )
    assert result.completed_steps == 2
    assert session.split_plan is not None


@pytest.mark.skipif(not _TORCH_SPEC, reason="torch not installed")
def test_ai_execute_fit_torch_tool() -> None:
    _require_torch_or_skip()
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=70),
            "x2": rng.normal(size=70),
            "y": np.asarray([0, 1] * 35, dtype=np.int64),
        }
    )
    session = (
        Session.ingest(df)
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
        .ai_configure(provider="mock")
    )
    loaders = session.ai_execute("make_torch_loaders", {"batch_size": 16}, confirm=True)
    assert loaders.executed
    fitted = session.ai_execute(
        "fit_torch", {"epochs": 1, "device": "cpu"}, confirm=True
    )
    assert fitted.executed
    assert session.dl_train_result is not None
