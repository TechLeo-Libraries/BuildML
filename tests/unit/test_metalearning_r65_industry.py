"""R6.5 industry-depth coverage for meta-learning backends."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.dl.extras import torch_spec_available
from buildml.metalearning import (
    list_metalearning_methods,
    metalearning_capability_matrix,
)
from buildml.metalearning.catalog import resolve_backend_method


def _frame(n_tasks: int = 8, n_per_task: int = 40, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for task in range(n_tasks):
        shift = rng.normal(0, 0.8, size=3)
        for i in range(n_per_task):
            label = i % 2
            center = shift + (1.0 if label else -1.0)
            x = rng.normal(center, 0.4, size=3)
            rows.append(
                {
                    "f0": float(x[0]),
                    "f1": float(x[1]),
                    "f2": float(x[2]),
                    "label": int(label),
                    "task_id": f"t{task}",
                }
            )
    return pd.DataFrame(rows)


def _session() -> Session:
    return (
        Session.ingest(_frame())
        .set_roles(
            {
                "f0": "feature",
                "f1": "feature",
                "f2": "feature",
                "label": "target",
                "task_id": "group",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=0)
        .scale(method="standard")
    )


def test_capability_matrix_exports() -> None:
    matrix = metalearning_capability_matrix()
    assert "sklearn" in matrix["backends"]
    assert matrix["backends"]["sklearn"]["available"] is True
    assert "prototypical" in matrix["backends"]["sklearn"]["methods"]
    assert "held_out_task_ids" in matrix["episodic_protocol"]
    sklearn_methods = list_metalearning_methods(backend="sklearn")
    assert "warm_start" in sklearn_methods


def test_resolve_backend_method_defaults() -> None:
    backend, method = resolve_backend_method(backend=None, method="prototypical")
    assert backend == "sklearn"
    assert method == "prototypical"


def test_invalid_backend_method_pair() -> None:
    with pytest.raises(ValidationError, match="not valid for backend"):
        resolve_backend_method(backend="sklearn", method="maml")


def _skip_if_torch_unusable() -> None:
    if not torch_spec_available():
        pytest.skip("torch not installed")
    try:
        from buildml.dl.extras import require_torch

        require_torch(feature="test")
    except MissingExtraError:
        pytest.skip("torch not usable")


@pytest.mark.skipif(torch_spec_available(), reason="torch spec present")
def test_torch_backend_missing_extra() -> None:
    with pytest.raises(MissingExtraError):
        resolve_backend_method(backend="torch", method="prototypical_torch")


def test_held_out_task_ids_preserved() -> None:
    session = _session()
    fit = session.fit_metalearning(
        method="prototypical",
        k_shot=3,
        n_episodes=6,
        task_holdout_fraction=0.25,
        prefer_reduce_components=False,
    )
    plan = session.metalearning_plan
    assert plan is not None
    assert fit.n_held_out_tasks == len(plan.held_out_task_ids)
    assert fit.backend == "sklearn"


@pytest.mark.skipif(not torch_spec_available(), reason="torch spec absent")
def test_torch_prototypical_smoke() -> None:
    _skip_if_torch_unusable()
    session = _session()
    fit = session.fit_metalearning(
        backend="torch",
        method="prototypical_torch",
        k_shot=3,
        n_episodes=4,
        meta_epochs=3,
        prefer_reduce_components=False,
    )
    assert fit.backend == "torch"
    assert fit.method == "prototypical_torch"
    assert session.metalearning_plan.meta_learner_ is not None
    ev = session.evaluate_metalearning(partition="test", k_shot=3)
    assert ev.method == "prototypical_torch"


@pytest.mark.skipif(not torch_spec_available(), reason="torch spec absent")
def test_industry_maml_smoke() -> None:
    _skip_if_torch_unusable()
    session = _session()
    fit = session.fit_metalearning(
        backend="industry",
        method="maml",
        k_shot=3,
        n_episodes=4,
        meta_epochs=3,
        inner_steps=2,
        prefer_reduce_components=False,
    )
    assert fit.backend == "industry"
    assert fit.method == "maml"
    adapt = session.adapt_to_task(
        task_id=session.metalearning_plan.train_task_ids[0],
        partition="train",
        max_support_per_class=3,
    )
    assert adapt.adapted_estimator_ is not None
