"""Torch search derives task and evidence from the eligible training rows."""

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import LeakageError, ValidationError
from buildml.data.splits import SplitPlan
from buildml.dl.search import nested_cv_torch, search_torch


def _task_session(poison):
    row = np.arange(64, dtype=float)
    train = np.arange(0, 64, 2)
    validation = np.arange(1, 32, 2)
    test = np.arange(33, 64, 2)
    target = np.zeros(64, dtype=float)
    target[train] = np.arange(len(train)) % 2
    target[validation] = np.arange(len(validation)) % 2
    target[test] = np.arange(len(test)) % 2
    for name, selected in (("validation", validation), ("test", test)):
        if poison in {name, "both"}:
            target[selected] = row[selected] + 0.25
    frame = pd.DataFrame(
        {"x": np.sin(row), "z": np.cos(row / 2), "target": target}, index=np.arange(64) // 3
    )
    return (
        Session.ingest(frame)
        .set_roles({"x": "feature", "z": "feature", "target": "target"})
        .inject_split(
            train_indices=train.tolist(),
            validation_indices=validation.tolist(),
            test_indices=test.tolist(),
        )
    )


@pytest.mark.parametrize("route", ("search", "nested_cv"))
@pytest.mark.parametrize("poison", (None, "validation", "test", "both"))
@pytest.mark.parametrize("normalize", (False, True))
def test_heldout_labels_cannot_select_task_or_change_real_fold_scores(route, poison, normalize):
    torch = pytest.importorskip("torch")
    base = _task_session(None)
    altered = _task_session(poison)
    models = []

    def factory(width, task, classes, params):
        model = torch.nn.Linear(width, classes if task == "classification" else 1)
        models.append(model)
        return model

    seed = 19
    options = dict(n_folds=2) if route == "search" else dict(outer_cv=2, inner_cv=2)
    options.update(
        param_grid={"learning_rate": [0.01]},
        epochs=1,
        batch_size=64,
        device="cpu",
        normalize=normalize,
        stratify=False,
        seed=seed,
        module_factory=factory,
    )
    control = getattr(base.dl, route)(**options)
    models.clear()
    result = getattr(altered.dl, route)(**options)
    assert result.task == control.task == "classification"
    assert result.scoring_metric == control.scoring_metric == "accuracy"
    assert result.held_out_partitions == ("validation", "test")
    assert result.to_dict() == control.to_dict()
    assert len(models) == (2 if route == "search" else 6)

    train = altered.partition("train")
    matrix = train[["x", "z"]].to_numpy()
    y = train["target"].to_numpy(dtype=int)
    folds = np.array_split(np.random.default_rng(seed).permutation(len(train)), 2)
    oracle = []
    for i, val in enumerate(folds):
        model = models[i if route == "search" else 3 * i + 2]
        design = matrix[val]
        if normalize:
            fit = matrix[folds[1 - i]]
            design = (design - fit.mean(axis=0)) / fit.std(axis=0)
        logits = design @ model.weight.detach().numpy().T + model.bias.detach().numpy()
        logits -= logits.max(axis=1, keepdims=True)
        loss = np.log(np.exp(logits).sum(axis=1)) - logits[np.arange(len(val)), y[val]]
        oracle.append((np.mean(logits.argmax(axis=1) == y[val]), loss.mean()))
    if route == "search":
        observed = result.trials[0].mean_metrics
        assert observed["accuracy"] == pytest.approx(np.mean([a for a, _ in oracle]))
        assert observed["loss"] == pytest.approx(np.mean([loss for _, loss in oracle]), rel=2e-6)
    else:
        for outer, (accuracy, loss) in zip(result.outer_folds, oracle, strict=True):
            assert outer.outer_metrics["accuracy"] == pytest.approx(accuracy)
            assert outer.outer_metrics["loss"] == pytest.approx(loss, rel=2e-6)


@pytest.mark.parametrize("route", ("search", "nested_cv", "cross_validate"))
def test_nonfinite_torch_fold_evidence_does_not_publish_a_result(route):
    torch = pytest.importorskip("torch")
    row = np.arange(32, dtype=float)
    x, z = np.sin(row), np.cos(row)
    session = Session.ingest(pd.DataFrame({"x": x, "z": z, "target": 1.5 * x + 0.4 * z})).set_roles(
        {"x": "feature", "z": "feature", "target": "target"}
    )

    def factory(width, task, classes, params=None):
        return torch.nn.Linear(width, 1)

    options = dict(outer_cv=2, inner_cv=2) if route == "nested_cv" else dict(n_folds=2)
    options.update(epochs=1, batch_size=64, device="cpu", task="regression", module_factory=factory)

    def run(learning_rate):
        config = (
            {"learning_rate": learning_rate}
            if route == "cross_validate"
            else {"param_grid": {"learning_rate": [learning_rate]}}
        )
        return getattr(session.dl, route)(**config, **options)

    normal = run(0.01)
    history = session.history
    assert np.isfinite(
        normal.best_metrics["mse"] if route == "search" else normal.mean_metrics["mse"]
    )
    with pytest.raises(ValidationError, match="non-finite"):
        run(1e20)
    assert session.history == history


@pytest.mark.parametrize("search", (search_torch, nested_cv_torch))
@pytest.mark.parametrize(
    ("case", "error", "message"),
    (
        ("negative_train", ValidationError, "train positions must be integers"),
        ("negative_test", ValidationError, "test positions must be integers"),
        ("out_of_bounds", ValidationError, "train positions must be integers"),
        ("duplicate_train", ValidationError, "train positions must be unique"),
        ("overlap", LeakageError, "Train and validation partitions overlap"),
        ("empty_train", ValidationError, "Train partition is empty"),
    ),
)
def test_restored_split_refuses_ambiguous_positions_before_fitting(search, case, error, message):
    torch = pytest.importorskip("torch")
    session = _task_session(None)
    snapshot = session.split_plan.to_dict()
    train = snapshot["train_indices"]
    if case == "negative_train":
        train[0] = -1  # Same row as the final, positive test position in Pandas.
    elif case == "negative_test":
        snapshot["test_indices"][-1] = -64  # Aliases the first train position.
    elif case == "out_of_bounds":
        train[0] = 64
    elif case == "duplicate_train":
        train[0] = train[1]
    elif case == "overlap":
        train[0] = snapshot["validation_indices"][0]
    else:
        train.clear()
    restored = SplitPlan.from_dict(snapshot)
    widths = []

    def factory(width, task, classes, params):
        widths.append(width)
        return torch.nn.Linear(width, classes)

    folds = {"n_folds": 2} if search is search_torch else {"outer_cv": 2, "inner_cv": 2}
    with pytest.raises(error, match=message):
        search(
            session.dataset,
            split_plan=restored,
            param_grid={"learning_rate": [0.01]},
            epochs=1,
            batch_size=64,
            device="cpu",
            stratify=False,
            module_factory=factory,
            **folds,
        )
    assert widths == []


@pytest.mark.parametrize(
    ("route", "folds", "message"),
    (
        ("search", {"n_folds": 1}, "n_folds must be >= 2"),
        ("search", {"n_folds": 33}, "Need at least 33 rows"),
        ("nested_cv", {"outer_cv": 1}, "n_folds must be >= 2"),
        ("nested_cv", {"outer_cv": 2, "inner_cv": 17}, "Need at least 17 rows"),
    ),
)
def test_fold_counts_are_checked_against_eligible_rows(route, folds, message):
    pytest.importorskip("torch")
    session = _task_session(None)
    history = session.history

    def factory(*args):
        pytest.fail("Invalid fold request trained a model")

    with pytest.raises(ValidationError, match=message):
        getattr(session.dl, route)(
            param_grid={"learning_rate": [0.01]}, module_factory=factory, device="cpu", **folds
        )
    assert session.history == history
