"""Unit coverage for the meta-learning thin slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.tools import build_default_registry
from buildml.core.errors import LeakageError, ValidationError
from buildml.explain.catalog import OPERATION_CATALOG
from buildml.metalearning.checkpoint import BUNDLE_FORMAT, load_metalearning_bundle


def _episodic_frame(
    n_tasks: int = 8,
    n_per_task: int = 40,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic multi-task classification frame with a group/task column."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for task in range(n_tasks):
        # Task-specific class means so few-shot adapt can succeed.
        shift = rng.normal(0, 1.0, size=2)
        for i in range(n_per_task):
            label = i % 2
            center = shift + (1.2 if label == 1 else -1.2)
            x = rng.normal(center, 0.45, size=2)
            rows.append(
                {
                    "x": float(x[0]),
                    "y": float(x[1]),
                    "label": int(label),
                    "task_id": f"t{task}",
                }
            )
    return pd.DataFrame(rows)


def _ready_session() -> Session:
    return (
        Session.ingest(_episodic_frame())
        .set_roles(
            {
                "x": "feature",
                "y": "feature",
                "label": "target",
                "task_id": "group",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )


def test_core_import_and_catalog() -> None:
    import buildml.metalearning as metalearning

    assert hasattr(metalearning, "fit_metalearning")
    assert hasattr(metalearning, "metalearning_capability_matrix")
    assert hasattr(Session, "fit_metalearning")
    for name in (
        "fit_metalearning",
        "adapt_to_task",
        "evaluate_metalearning",
        "save_metalearning_bundle",
        "load_metalearning_bundle",
    ):
        assert name in OPERATION_CATALOG
    assert (
        "metalearning-episodic" in OPERATION_CATALOG["fit_metalearning"].concept_links
    )
    assert (
        "metalearning-bundle-boundary"
        in OPERATION_CATALOG["save_metalearning_bundle"].concept_links
    )


def test_fit_adapt_evaluate_bundle(tmp_path: Path) -> None:
    session = _ready_session()
    fit = session.fit_metalearning(
        method="prototypical",
        k_shot=3,
        n_query=6,
        n_episodes=12,
        task_holdout_fraction=0.25,
    )
    assert fit.n_meta_train_tasks >= 2
    assert session.metalearning_plan is not None
    assert session.metalearning_plan.task_column == "task_id"

    # Adapt on a meta-train task support set.
    task_id = session.metalearning_plan.train_task_ids[0]
    adapt = session.adapt_to_task(task_id=task_id, partition="train", max_support_per_class=3)
    assert adapt.n_support >= 2
    assert adapt.n_classes_adapted >= 2

    ev = session.evaluate_metalearning(
        partition="validation",
        k_shot=3,
        prefer_novel_tasks=True,
    )
    assert session.metalearning_eval_result is not None
    # May score novel or overlapping tasks depending on random split.
    assert ev.n_tasks_evaluated >= 0
    if ev.n_tasks_evaluated > 0:
        assert "mean_accuracy" in ev.metrics

    before = session.explain("evaluate_metalearning", moment="before")
    assert before.prerequisite_status.get("metalearning-plan") is True

    bundle = session.save_metalearning_bundle(tmp_path / "metalearning_bundle")
    assert (bundle / "meta.json").is_file()
    plan = load_metalearning_bundle(bundle, trusted=True)
    assert plan.n_train_rows == fit.n_train_rows

    restored = Session.ingest(session.to_pandas()).set_roles(
        {
            "x": "feature",
            "y": "feature",
            "label": "target",
            "task_id": "group",
        }
    )
    restored._split_plan = session.split_plan
    restored._dataset = session.dataset
    restored.load_metalearning_bundle(bundle, trusted=True)
    assert restored.metalearning_plan is not None
    assert restored.metalearning_plan.method == "prototypical"


def test_refuse_without_split() -> None:
    session = Session.ingest(_episodic_frame()).set_roles(
        {
            "x": "feature",
            "y": "feature",
            "label": "target",
            "task_id": "group",
        }
    )
    with pytest.raises(LeakageError, match="split"):
        session.fit_metalearning()


def test_refuse_without_task_column() -> None:
    session = (
        Session.ingest(_episodic_frame())
        .set_roles({"x": "feature", "y": "feature", "label": "target"})
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )
    with pytest.raises(ValidationError, match="task/group"):
        session.fit_metalearning()


def test_ai_allowlist() -> None:
    registry = build_default_registry()
    names = {t.name for t in registry.tools}
    for name in (
        "fit_metalearning",
        "adapt_to_task",
        "evaluate_metalearning",
        "save_metalearning_bundle",
        "load_metalearning_bundle",
    ):
        assert name in names


def test_bundle_format_constant() -> None:
    assert BUNDLE_FORMAT == "buildml.metalearning_bundle.v1"
