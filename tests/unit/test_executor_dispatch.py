"""Integration tests for executor tool dispatch against real Session methods.

These tests verify that M2 tool dispatch handlers in executor.py correctly
invoke the actual Session methods without signature mismatches or TypeErrors.
"""

from __future__ import annotations

import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import (
    ExecutorProposal,
    ExecutorResult,
    _resolve_estimator,
    execute_tool,
    propose_tool_execution,
)
from buildml.ai.tools import build_default_registry
from buildml.ai.types import ToolCall
from buildml.core.errors import ValidationError


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """Tiny DataFrame for dispatch testing."""
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60],
        "income": [50000.0, 60000.0, None, 80000.0, 90000.0, 100000.0, 110000.0, 120000.0],
        "category": ["A", "B", "A", "B", "A", "B", "A", "B"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
    })


@pytest.fixture
def session_with_roles(tiny_df: pd.DataFrame) -> Session:
    """Session with roles set, ready for split."""
    session = Session.ingest(tiny_df)
    session.set_roles({"age": "feature", "income": "feature", "category": "feature", "target": "target"})
    return session


@pytest.fixture
def session_with_split(session_with_roles: Session) -> Session:
    """Session with split, ready for prep/fit."""
    session_with_roles.split(test_size=0.25, random_state=42)
    return session_with_roles


@pytest.fixture
def session_ready_for_fit(session_with_split: Session) -> Session:
    """Session with split and encoded, ready for fit."""
    session_with_split.impute(strategy="median")
    session_with_split.encode(method="onehot", columns=["category"])
    return session_with_split


@pytest.fixture
def registry():
    """Default tool registry."""
    return build_default_registry()


class TestReadToolDispatch:
    """Tests for read-only tool dispatch: head, walkthrough, ai_status."""

    def test_head_dispatch(self, session_with_roles: Session, registry) -> None:
        """head tool dispatches correctly to Session.head()."""
        session = session_with_roles
        session.ai_configure(provider="mock")

        proposal = propose_tool_execution("head", {"n": 3}, registry)
        result = execute_tool(session, proposal, confirmed=True, registry=registry)

        assert result.executed
        assert result.error is None
        assert hasattr(result.result, "shape") or isinstance(result.result, pd.DataFrame)
        assert len(result.result) == 3

    def test_head_default_n(self, session_with_roles: Session, registry) -> None:
        """head tool uses default n=5 when not specified."""
        session = session_with_roles
        session.ai_configure(provider="mock")

        proposal = propose_tool_execution("head", {}, registry)
        result = execute_tool(session, proposal, confirmed=True, registry=registry)

        assert result.executed
        assert len(result.result) == 5

    def test_walkthrough_dispatch(self, session_with_roles: Session, registry) -> None:
        """walkthrough tool dispatches correctly to Session.walkthrough()."""
        session = session_with_roles
        session.ai_configure(provider="mock")

        proposal = propose_tool_execution("walkthrough", {}, registry)
        result = execute_tool(session, proposal, confirmed=True, registry=registry)

        assert result.executed
        assert result.error is None
        assert result.result is not None

    def test_ai_status_dispatch(self, session_with_roles: Session, registry) -> None:
        """ai_status tool dispatches correctly to Session.ai_status()."""
        session = session_with_roles
        session.ai_configure(provider="mock")

        proposal = propose_tool_execution("ai_status", {}, registry)
        result = execute_tool(session, proposal, confirmed=True, registry=registry)

        assert result.executed
        assert result.error is None
        assert isinstance(result.result, dict)
        assert "enabled" in result.result
        assert "provider" in result.result


class TestSplitDispatch:
    """Tests for split tool dispatch."""

    def test_split_dispatch_basic(self, session_with_roles: Session, registry) -> None:
        """split tool dispatches correctly with default args."""
        session = session_with_roles
        session.ai_configure(provider="mock")

        proposal = propose_tool_execution("split", {"test_size": 0.25}, registry)
        result = execute_tool(session, proposal, confirmed=True, registry=registry)

        assert result.executed
        assert result.error is None
        assert "split_created" in result.result
        assert session._split_plan is not None

    def test_split_with_validation(self, session_with_roles: Session, registry) -> None:
        """split tool handles validation_size correctly."""
        session = session_with_roles
        session.ai_configure(provider="mock")

        proposal = propose_tool_execution(
            "split",
            {"test_size": 0.2, "validation_size": 0.1, "stratify": False},
            registry,
        )
        result = execute_tool(session, proposal, confirmed=True, registry=registry)

        assert result.executed
        assert result.error is None
        assert len(result.state_changes) >= 2  # train/test + validation

    def test_split_stratify_default_is_false(self, session_with_roles: Session, registry) -> None:
        """split defaults stratify to False (matching Session signature)."""
        session = session_with_roles
        session.ai_configure(provider="mock")

        # Don't pass stratify - should use default False
        proposal = propose_tool_execution("split", {"test_size": 0.25}, registry)
        result = execute_tool(session, proposal, confirmed=True, registry=registry)

        assert result.executed
        # Verify split was created without error (stratify=False works with any target)


class TestPrepToolDispatch:
    """Tests for prep tool dispatch: impute, encode, scale."""

    def test_impute_dispatch_correct_signature(
        self, session_with_split: Session, registry
    ) -> None:
        """impute tool uses correct Session signature (strategy, not numeric/categorical)."""
        session = session_with_split
        session.ai_configure(provider="mock")

        # Use the corrected schema: single 'strategy' param
        proposal = propose_tool_execution(
            "impute",
            {"strategy": "mean"},
            registry,
        )
        result = execute_tool(session, proposal, confirmed=True, registry=registry)

        assert result.executed, f"impute failed: {result.error}"
        assert result.error is None
        assert "imputed" in result.result

    def test_impute_with_columns(
        self, session_with_split: Session, registry
    ) -> None:
        """impute tool correctly passes columns parameter."""
        session = session_with_split
        session.ai_configure(provider="mock")

        proposal = propose_tool_execution(
            "impute",
            {"strategy": "median", "columns": ["income"]},
            registry,
        )
        result = execute_tool(session, proposal, confirmed=True, registry=registry)

        assert result.executed, f"impute with columns failed: {result.error}"
        assert result.error is None

    def test_encode_dispatch(self, session_with_split: Session, registry) -> None:
        """encode tool dispatches correctly."""
        session = session_with_split
        session.ai_configure(provider="mock")

        proposal = propose_tool_execution(
            "encode",
            {"method": "onehot", "columns": ["category"]},
            registry,
        )
        result = execute_tool(session, proposal, confirmed=True, registry=registry)

        assert result.executed, f"encode failed: {result.error}"
        assert result.error is None
        assert "encoded" in result.result

    def test_scale_dispatch(self, session_with_split: Session, registry) -> None:
        """scale tool dispatches correctly."""
        session = session_with_split
        session.ai_configure(provider="mock")

        proposal = propose_tool_execution(
            "scale",
            {"method": "standard"},
            registry,
        )
        result = execute_tool(session, proposal, confirmed=True, registry=registry)

        assert result.executed, f"scale failed: {result.error}"
        assert result.error is None
        assert "scaled" in result.result


class TestFitEvaluateDispatch:
    """Tests for fit/evaluate tool dispatch."""

    def test_fit_dispatch_with_string_estimator(
        self, session_ready_for_fit: Session, registry
    ) -> None:
        """fit tool resolves string estimator name to actual class."""
        session = session_ready_for_fit
        session.ai_configure(provider="mock")

        proposal = propose_tool_execution(
            "fit",
            {"estimator": "LogisticRegression", "task": "classification"},
            registry,
        )
        result = execute_tool(session, proposal, confirmed=True, registry=registry)

        assert result.executed, f"fit failed: {result.error}"
        assert result.error is None
        assert result.result["fitted"]
        assert result.result["estimator"] == "LogisticRegression"

    def test_fit_with_hyperparameters(
        self, session_ready_for_fit: Session, registry
    ) -> None:
        """fit tool passes hyperparameters to estimator constructor."""
        session = session_ready_for_fit
        session.ai_configure(provider="mock")

        proposal = propose_tool_execution(
            "fit",
            {
                "estimator": "RandomForestClassifier",
                "hyperparameters": {"n_estimators": 10, "max_depth": 3},
            },
            registry,
        )
        result = execute_tool(session, proposal, confirmed=True, registry=registry)

        assert result.executed, f"fit with hyperparams failed: {result.error}"
        assert result.error is None
        assert result.result["estimator"] == "RandomForestClassifier"

    def test_fit_unknown_estimator_rejected(self, registry) -> None:
        """fit tool rejects unknown estimator names."""
        with pytest.raises(ValidationError, match="Unknown estimator"):
            _resolve_estimator("MagicClassifier", {})

    def test_evaluate_dispatch(self, session_ready_for_fit: Session, registry) -> None:
        """evaluate tool dispatches correctly after fit."""
        session = session_ready_for_fit
        session.ai_configure(provider="mock")

        # First fit
        from sklearn.linear_model import LogisticRegression
        session.fit(LogisticRegression(), task="classification")

        # Then evaluate via tool dispatch
        proposal = propose_tool_execution(
            "evaluate",
            {"partition": "test"},
            registry,
        )
        result = execute_tool(session, proposal, confirmed=True, registry=registry)

        assert result.executed, f"evaluate failed: {result.error}"
        assert result.error is None


class TestDestructiveToolConfirmation:
    """Tests that destructive tools require confirmation."""

    def test_drop_columns_requires_confirm(
        self, session_with_roles: Session, registry
    ) -> None:
        """drop_columns requires explicit confirmation."""
        session = session_with_roles
        session.ai_configure(provider="mock")

        proposal = propose_tool_execution(
            "drop_columns",
            {"columns": ["age"]},
            registry,
        )

        # Must require confirmation
        assert proposal.requires_confirmation

        # Without confirmation, should not execute
        result = execute_tool(session, proposal, confirmed=False, registry=registry)
        assert not result.executed
        assert "confirmation" in result.error.lower()

        # Verify column not dropped
        assert "age" in session.dataset.frame.columns

    def test_drop_columns_executes_with_confirm(
        self, session_with_roles: Session, registry
    ) -> None:
        """drop_columns executes when confirmed."""
        session = session_with_roles
        session.ai_configure(provider="mock")

        proposal = propose_tool_execution(
            "drop_columns",
            {"columns": ["category"]},
            registry,
        )
        result = execute_tool(session, proposal, confirmed=True, registry=registry)

        assert result.executed
        assert result.error is None
        assert "category" not in session.dataset.frame.columns


class TestEstimatorResolver:
    """Tests for _resolve_estimator helper."""

    def test_resolve_logistic_regression(self) -> None:
        """Resolves LogisticRegression by name."""
        est = _resolve_estimator("LogisticRegression", {})
        assert est.__class__.__name__ == "LogisticRegression"

    def test_resolve_case_insensitive(self) -> None:
        """Estimator resolution is case-insensitive."""
        est = _resolve_estimator("randomforestclassifier", {})
        assert est.__class__.__name__ == "RandomForestClassifier"

    def test_resolve_with_hyperparams(self) -> None:
        """Hyperparameters are passed to constructor."""
        est = _resolve_estimator("RandomForestClassifier", {"n_estimators": 5})
        assert est.n_estimators == 5

    def test_resolve_instance_passthrough(self) -> None:
        """Already-instantiated estimator passes through."""
        from sklearn.linear_model import Ridge
        original = Ridge(alpha=2.0)
        result = _resolve_estimator(original, {})
        assert result is original

    def test_resolve_none_raises(self) -> None:
        """None estimator raises ValidationError."""
        with pytest.raises(ValidationError, match="requires an estimator"):
            _resolve_estimator(None, {})

    def test_resolve_invalid_hyperparams(self) -> None:
        """Invalid hyperparameters raise ValidationError."""
        with pytest.raises(ValidationError, match="Invalid hyperparameters"):
            _resolve_estimator("LogisticRegression", {"not_a_real_param": 123})


class TestOtherToolDispatch:
    """Tests for other tool dispatch handlers."""

    def test_describe_dataset_dispatch(
        self, session_with_roles: Session, registry
    ) -> None:
        """describe_dataset dispatches to Session.metadata()."""
        session = session_with_roles
        session.ai_configure(provider="mock")

        proposal = propose_tool_execution("describe_dataset", {}, registry)
        result = execute_tool(session, proposal, confirmed=True, registry=registry)

        assert result.executed
        assert isinstance(result.result, dict)
        assert "has_dataset" in result.result

    def test_workflow_status_dispatch(
        self, session_with_roles: Session, registry
    ) -> None:
        """workflow_status dispatches to Session.workflow()."""
        session = session_with_roles
        session.ai_configure(provider="mock")

        proposal = propose_tool_execution("workflow_status", {}, registry)
        result = execute_tool(session, proposal, confirmed=True, registry=registry)

        assert result.executed
        assert result.result is not None

    def test_eda_summary_dispatch(
        self, session_with_roles: Session, registry
    ) -> None:
        """eda_summary dispatches to Session.eda()."""
        session = session_with_roles
        session.ai_configure(provider="mock")

        proposal = propose_tool_execution("eda_summary", {}, registry)
        result = execute_tool(session, proposal, confirmed=True, registry=registry)

        assert result.executed
        assert result.error is None

    def test_checkpoint_save_dispatch(
        self, session_with_split: Session, registry, tmp_path
    ) -> None:
        """checkpoint_save dispatches to Session.checkpoint_save()."""
        session = session_with_split
        session.ai_configure(provider="mock")

        checkpoint_path = str(tmp_path / "test_checkpoint.buildml")
        proposal = propose_tool_execution(
            "checkpoint_save",
            {"path": checkpoint_path},
            registry,
        )
        result = execute_tool(session, proposal, confirmed=True, registry=registry)

        assert result.executed, f"checkpoint_save failed: {result.error}"
        assert result.error is None
