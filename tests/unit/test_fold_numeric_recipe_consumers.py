"""Fold-local numeric policies must keep their declared numerical meaning."""

import bisect
import copy
import hashlib
import math

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.preprocess import FoldLocalPreprocessor, PreprocessRecipe, build_fold_preprocessor

BACKINGS = ("pandas", "pandas-parquet", "polars-lazy", "duckdb-lazy")


def _frame():
    t = np.arange(36)
    return pd.DataFrame(
        {
            "x": (t % 9 - 4).astype(float) + np.sin(t) / 5,
            "constant": np.where(t < 24, 2.0, np.tile([-5.0, 2.0, 9.0], 12)),
            "keep": np.cos(t / 3),
            "target": 1 + np.sin(t / 4) + t / 17,
        },
        index=[i // 2 for i in t],
    )


def _session(frame, backing, directory):
    engine = backing.split("-")[0]
    if engine != "pandas":
        pytest.importorskip(engine)
    if backing == "pandas":
        session = Session.ingest(frame)
    else:
        path = directory / "rows.parquet"
        frame.to_parquet(path, index=False)
        session = Session.ingest(
            path, engine=engine, mode="memory" if engine == "pandas" else "lazy"
        )
    return session.set_roles(
        {"x": "feature", "constant": "feature", "keep": "feature", "target": "target"}
    ).inject_split(
        train_indices=list(range(24)),
        validation_indices=list(range(24, 30)),
        test_indices=list(range(30, 36)),
    )


def _quantile(values, fraction):
    ordered = sorted(float(x) for x in values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo])


def _fences(values, method):
    if method == "iqr":
        q1, q3 = _quantile(values, 0.25), _quantile(values, 0.75)
        return q1 - 1.25 * (q3 - q1), q3 + 1.25 * (q3 - q1)
    mean = sum(float(v) for v in values) / len(values)
    sd = math.sqrt(sum((float(v) - mean) ** 2 for v in values) / len(values))
    return mean - 1.75 * sd, mean + 1.75 * sd


def _design(train, query, method, action, binning):
    learned, replay = train.copy(), query.copy()
    for col in ("x", "constant"):
        lo, hi = _fences(train[col], method)
        if action == "cap":
            for frame in (learned, replay):
                frame[col] = [min(hi, max(lo, float(v))) for v in frame[col]]
    if binning == "none":
        return replay.to_numpy(dtype=float), list(replay.columns)
    strategy, encoding = binning.split("-")
    names, blocks = ["keep"], [replay.keep.to_numpy(dtype=float).reshape(-1, 1)]
    for col in ("x", "constant"):
        values = learned[col].tolist()
        count = min(3, len(set(values)))
        if count < 2:
            cuts = []
        elif strategy == "quantile":
            edges = sorted({_quantile(values, j / count) for j in range(count + 1)})
            cuts = edges[1:-1] if len(edges) >= 3 else []
        else:
            low, high = min(values), max(values)
            cuts = [low + (high - low) * j / count for j in range(1, count)]
        codes = [bisect.bisect_left(cuts, float(v)) for v in replay[col]]
        if encoding == "ordinal":
            names.append(col + "_bin")
            blocks.append(np.asarray(codes, dtype=float).reshape(-1, 1))
        else:
            names.extend(f"{col}_bin_{j}" for j in range(len(cuts) + 1))
            blocks.append(
                np.asarray([[float(c == j) for j in range(len(cuts) + 1)] for c in codes])
            )
    return np.column_stack(blocks), names


def _recipe(method, action, binning):
    return PreprocessRecipe(
        impute=None,
        outliers=method,
        outlier_action=action,
        outlier_columns=("x", "constant"),
        iqr_multiplier=np.float64(1.25),
        zscore_threshold=np.float32(1.75),
        binning=None if binning == "none" else binning.split("-")[0],
        binning_encode_as="ordinal" if binning == "none" else binning.split("-")[1],
        binning_columns=("x", "constant"),
        n_bins=np.int64(3),
    )


def _ridge(train, target, query):
    center = train.mean(axis=0)
    x = train - center
    average = target.mean()
    coef = np.linalg.solve(x.T @ x + 0.7 * np.eye(x.shape[1]), x.T @ (target - average))
    return (query - center) @ coef + average


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("method", ["iqr", "zscore"])
@pytest.mark.parametrize("action", ["cap", "detect"])
@pytest.mark.parametrize("binning", ["none", "quantile-ordinal", "uniform-onehot"])
def test_native_cv_and_saved_regressors_match_fold_numeric_identities(
    backing, method, action, binning, tmp_path
):
    source = _frame()
    session = _session(source, backing, tmp_path)
    frame = session.dataset.to_pandas()
    before = frame.copy(deep=True)
    train, query = source.iloc[:24].drop(columns="target"), source.iloc[24:].drop(columns="target")
    target = source.target.iloc[:24].to_numpy()
    recipe = _recipe(method, action, binning)
    recipe_before = copy.deepcopy(recipe.to_dict())
    pipeline = Pipeline(
        [("features", FoldLocalPreprocessor(recipe)), ("model", Ridge(alpha=0.7, solver="svd"))]
    ).fit(train, target)
    learned, _ = _design(train, train, method, action, binning)
    expected, names = _design(train, query, method, action, binning)
    actual = pipeline.named_steps["features"].transform(query)
    assert actual.index.equals(query.index) and list(actual.columns) == names
    np.testing.assert_allclose(actual.to_numpy(), expected, atol=1e-12, rtol=1e-12)
    prediction = _ridge(learned, target, expected)
    np.testing.assert_allclose(pipeline.predict(query), prediction, atol=1e-10, rtol=1e-10)
    path = tmp_path / "regressor.joblib"
    state = joblib.hash(pipeline)
    joblib.dump(pipeline, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    restored = joblib.load(path)
    np.testing.assert_allclose(restored.predict(query), prediction, atol=1e-10, rtol=1e-10)
    assert joblib.hash(restored) == state == joblib.hash(pipeline)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    result = session.classical.cv_score(
        Ridge(alpha=0.7, solver="svd"),
        task="regression",
        cv=KFold(3, shuffle=False),
        scoring_metric="rmse",
        preprocess=recipe,
    )
    x, y = frame.iloc[:24].drop(columns="target"), frame.target.iloc[:24].to_numpy()
    scores = []
    for report, (train_pos, eval_pos) in zip(
        result.folds, KFold(3, shuffle=False).split(x), strict=True
    ):
        a, _ = _design(x.iloc[train_pos], x.iloc[train_pos], method, action, binning)
        b, _ = _design(x.iloc[train_pos], x.iloc[eval_pos], method, action, binning)
        predicted = _ridge(a, y[train_pos], b)
        score = math.sqrt(sum((predicted - y[eval_pos]) ** 2) / len(eval_pos))
        assert report.metrics["rmse"] == pytest.approx(score, abs=1e-10)
        scores.append(score)
    assert result.mean_metrics["rmse"] == pytest.approx(sum(scores) / len(scores), abs=1e-10)
    assert result.std_metrics["rmse"] == pytest.approx(np.std(scores, ddof=1), abs=1e-10)
    assert recipe.to_dict() == recipe_before
    pd.testing.assert_frame_equal(session.dataset.to_pandas(), before)


BAD_OPTIONS = (
    [{"outliers": value} for value in ("typo", "IQR", "", True, [], {}, np.nan, 42)]
    + [{"binning": value} for value in ("typo", "Quantile", "", True, [], {}, np.nan, 42)]
    + [
        {"binning": "uniform", "binning_encode_as": value}
        for value in ("bad", "", True, None, [], {}, np.nan)
    ]
    + [
        {"outliers": "iqr", "outlier_action": value}
        for value in ("drop", "", True, None, [], {}, np.nan)
    ]
    + [
        {"outliers": method, name: value}
        for method, name in (("iqr", "iqr_multiplier"), ("zscore", "zscore_threshold"))
        for value in (True, np.bool_(True), np.nan, np.inf, -np.inf, 0, -1, "1.5", None, 10**400)
    ]
)


@pytest.mark.parametrize("options", BAD_OPTIONS)
def test_invalid_numeric_policy_is_refused_at_the_public_fold_boundary(options):
    frame = _frame().iloc[:24].drop(columns="target")
    before = frame.copy(deep=True)
    recipe = PreprocessRecipe(impute=None, **options)
    with pytest.raises(
        ValidationError, match="[Oo]utlier|[Bb]inning|iqr_multiplier|zscore_threshold"
    ):
        build_fold_preprocessor(frame, recipe)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize(
    "options",
    [
        {"outliers": "typo"},
        {"binning": "typo"},
        {"binning": "uniform", "binning_encode_as": "typo"},
        {"outliers": "iqr", "iqr_multiplier": np.nan},
        {"outliers": "zscore", "zscore_threshold": True},
        {"outliers": "iqr", "outlier_action": []},
    ],
)
def test_invalid_cv_recipe_preserves_previous_result_and_population(backing, options, tmp_path):
    session = _session(_frame(), backing, tmp_path)
    prior = session.classical.cv_score(Ridge(alpha=0.7), task="regression", cv=3)
    before, history = session.dataset.to_pandas().copy(deep=True), copy.deepcopy(session.history)
    split = session.split_plan
    with pytest.raises(
        ValidationError, match="[Oo]utlier|[Bb]inning|iqr_multiplier|zscore_threshold"
    ):
        session.classical.cv_score(
            Ridge(alpha=0.7),
            task="regression",
            cv=3,
            preprocess=PreprocessRecipe(impute=None, **options),
        )
    assert session._last_cv is prior and session.split_plan is split and session.history == history
    pd.testing.assert_frame_equal(session.dataset.to_pandas(), before)


@pytest.mark.parametrize("step", ["iqr", "zscore", "uniform", "quantile"])
@pytest.mark.parametrize("phase", ["fit", "replay"])
@pytest.mark.parametrize("storage", ["complex", "object"])
def test_complex_samples_cannot_lose_their_imaginary_part_in_numeric_recipes(step, phase, storage):
    frame = _frame().iloc[:24].drop(columns="target")
    options = (
        {"outliers": step, "outlier_columns": ("x",)}
        if step in {"iqr", "zscore"}
        else {"binning": step, "binning_columns": ("x",)}
    )
    recipe = PreprocessRecipe(impute=None, **options)
    dirty = frame.copy()
    dirty["x"] = dirty.x.astype(complex) + 2j
    if storage == "object":
        dirty["x"] = dirty.x.astype(object)
    before = dirty.copy(deep=True)
    with pytest.raises(ValidationError, match="[Rr]eal|[Cc]omplex"):
        if phase == "fit":
            build_fold_preprocessor(dirty, recipe)
        else:
            build_fold_preprocessor(frame, recipe).transform(dirty)
    pd.testing.assert_frame_equal(dirty, before)


def test_disabled_numeric_steps_do_not_validate_inactive_knobs():
    frame = _frame().iloc[:24].drop(columns="target")
    recipe = PreprocessRecipe(
        impute="median",
        outliers=None,
        iqr_multiplier=None,
        zscore_threshold=np.nan,
        binning=None,
        binning_encode_as="unused",
    )
    np.testing.assert_array_equal(build_fold_preprocessor(frame, recipe).transform(frame), frame)


@pytest.mark.parametrize("step", ["iqr", "zscore", "uniform", "quantile"])
def test_numeric_steps_require_numeric_training_columns(step):
    options = {"outliers": step} if step in {"iqr", "zscore"} else {"binning": step}
    with pytest.raises(ValidationError, match="numeric columns"):
        build_fold_preprocessor(
            pd.DataFrame({"label": ["red", "blue"]}), PreprocessRecipe(impute=None, **options)
        )
