"""Fold-generated features keep column ownership and numerical meaning."""

import bisect
import calendar
from datetime import datetime, timedelta

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

KINDS = (
    "date",
    "clock",
    "count",
    "tfidf",
    "quantile-ordinal",
    "quantile-onehot",
    "uniform-ordinal",
    "uniform-onehot",
)
BACKINGS = (
    "pandas",
    "pandas-parquet",
    "polars-memory",
    "polars-lazy",
    "duckdb-memory",
    "duckdb-lazy",
)


def _recipe(kind, scale=False):
    options = dict(impute=None, scale="standard" if scale else None)
    if kind in {"date", "clock"}:
        options.update(
            dates=True,
            date_columns=("when",),
            date_include_time=kind == "clock",
            date_drop_original=True,
        )
    elif kind in {"count", "tfidf"}:
        options.update(text=kind, text_columns=("review",), text_max_features=16)
    else:
        strategy, encoding = kind.split("-")
        options.update(
            binning=strategy, binning_encode_as=encoding, binning_columns=("x",), n_bins=np.int64(3)
        )
    return PreprocessRecipe(**options)


def _frame(kind, n=48, duplicate=False):
    positions = np.arange(n)
    frame = pd.DataFrame({"keep": np.sin(positions / 3) + 0.1 * positions})
    if kind in {"date", "clock"}:
        frame["when"] = [
            datetime(2024, 1, 28, 11, 12, 13) + timedelta(days=int(i), seconds=int(37 * i))
            for i in positions
        ]
    elif kind in {"count", "tfidf"}:
        frame["review"] = np.tile(["red moon", "blue moon", "red star", "blue star"], n // 4)
    else:
        frame["x"] = (positions % 11) ** 2 + positions / 10
    if duplicate:
        frame.index = [f"row-{i // 2}" for i in positions]
    return frame


def _quantile(values, fraction):
    ordered = sorted(values)
    location = (len(ordered) - 1) * fraction
    lo = int(location)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (location - lo)


def _bins(train, values, strategy, encoding, source="x"):
    finite = [float(v) for v in train if pd.notna(v) and np.isfinite(v)]
    count = min(3, len(set(finite))) if len(set(finite)) >= 2 else 2
    if strategy == "quantile":
        edges = sorted(set(_quantile(finite, i / count) for i in range(count + 1)))
    else:
        edges = sorted(
            set(min(finite) + i * (max(finite) - min(finite)) / count for i in range(count + 1))
        )
    inner = edges[1:-1] if len(edges) >= 3 else []
    codes = [np.nan if pd.isna(v) else bisect.bisect_left(inner, v) for v in values]
    if encoding == "ordinal":
        return np.array(codes, dtype=float)[:, None], [source + "_bin"]
    result = np.full((len(values), len(inner) + 1), np.nan)
    for i, code in enumerate(codes):
        if not np.isnan(code):
            result[i] = 0.0
            result[i, int(code)] = 1.0
    return result, [f"{source}_bin_{i}" for i in range(len(inner) + 1)]


def _design(train, query, kind, scale=False):
    def raw(values):
        keep = values[["keep"]].to_numpy(dtype=float)
        if kind in {"date", "clock"}:
            rows = []
            suffixes = [
                "year",
                "month",
                "day",
                "dayofweek",
                "dayofyear",
                "quarter",
                "is_month_start",
                "is_month_end",
            ]
            if kind == "clock":
                suffixes += ["hour", "minute", "second"]
            for value in values.when:
                date = value.to_pydatetime() if hasattr(value, "to_pydatetime") else value
                row = [
                    date.year,
                    date.month,
                    date.day,
                    date.weekday(),
                    date.timetuple().tm_yday,
                    (date.month - 1) // 3 + 1,
                    int(date.day == 1),
                    int(date.day == calendar.monthrange(date.year, date.month)[1]),
                ]
                if kind == "clock":
                    row += [date.hour, date.minute, date.second]
                rows.append(row)
            block, names = np.asarray(rows, dtype=float), ["when_" + name for name in suffixes]
        elif kind in {"count", "tfidf"}:
            vocabulary = sorted({token for document in train.review for token in document.split()})
            block = np.array(
                [
                    [document.split().count(token) for token in vocabulary]
                    for document in values.review
                ],
                dtype=float,
            )
            if kind == "tfidf":
                df = np.array(
                    [
                        sum(token in document.split() for document in train.review)
                        for token in vocabulary
                    ]
                )
                block *= np.log((len(train) + 1) / (df + 1)) + 1
                lengths = np.sqrt(np.sum(block * block, axis=1))
                block /= np.where(lengths == 0, 1.0, lengths)[:, None]
            names = ["review__" + token for token in vocabulary]
        else:
            strategy, encoding = kind.split("-")
            block, names = _bins(train.x, values.x, strategy, encoding)
        return np.column_stack([keep, block]), ["keep", *names]

    learned, names = raw(train)
    result, _ = raw(query)
    if scale:
        # Text and calendar parts precede numeric scaling. Ordinal bins follow
        # their numeric source; one-hot bins are created after scale selection.
        selected = (
            list(range(len(names)))
            if kind in {"date", "clock", "count", "tfidf", "quantile-ordinal", "uniform-ordinal"}
            else [0]
        )
        mean = learned[:, selected].mean(axis=0)
        sd = learned[:, selected].std(axis=0)
        result[:, selected] = (result[:, selected] - mean) / np.where(sd == 0, 1.0, sd)
    return result, names


def _ridge(train, target, query):
    mean = train.mean(axis=0)
    centered = train - mean
    coefficients = np.linalg.solve(
        centered.T @ centered + 0.6 * np.eye(train.shape[1]), centered.T @ (target - target.mean())
    )
    return (query - mean) @ coefficients + target.mean()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("duplicate", (False, True))
@pytest.mark.parametrize("scale", (False, True))
def test_generated_design_and_saved_ridge_match_independent_matrix(
    kind, duplicate, scale, tmp_path
):
    train, query = (
        _frame(kind, duplicate=duplicate).iloc[:32],
        _frame(kind, duplicate=duplicate).iloc[32:],
    )
    target = np.sin(np.arange(len(train)) / 4) + np.arange(len(train)) / 13
    before_train, before_query = train.copy(deep=True), query.copy(deep=True)
    recipe = _recipe(kind, scale)
    pipeline = Pipeline(
        [("features", FoldLocalPreprocessor(recipe)), ("model", Ridge(alpha=0.6))]
    ).fit(train, target)
    expected_train, names = _design(train, train, kind, scale)
    expected_query, _ = _design(train, query, kind, scale)
    design = pipeline.named_steps["features"].transform(query)
    assert list(design.columns) == names and design.index.equals(query.index)
    np.testing.assert_allclose(design.to_numpy(dtype=float), expected_query, atol=1e-12)
    expected_prediction = _ridge(expected_train, target, expected_query)
    np.testing.assert_allclose(pipeline.predict(query), expected_prediction, atol=1e-7)
    before = joblib.hash(pipeline)
    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(restored.predict(query), expected_prediction, atol=1e-7)
    assert joblib.hash(pipeline) == before == joblib.hash(restored)
    pd.testing.assert_frame_equal(train, before_train)
    pd.testing.assert_frame_equal(query, before_query)


def _session(frame, tmp_path, backing="pandas"):
    engine = backing.split("-")[0]
    if engine != "pandas":
        pytest.importorskip(engine)
    if backing.endswith(("parquet", "lazy")):
        path = tmp_path / "rows.parquet"
        frame.to_parquet(path, index=False)
        session = Session.ingest(
            path, engine=engine, mode="memory" if engine == "pandas" else "lazy"
        )
    else:
        session = Session.ingest(frame, engine=engine)
    session.set_roles(
        {column: "feature" for column in frame.columns if column != "target"} | {"target": "target"}
    )
    return session.inject_split(
        train_indices=list(range(32)),
        validation_indices=list(range(32, 40)),
        test_indices=list(range(40, 48)),
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("backing", BACKINGS)
def test_cv_refits_generated_design_on_fold_train_only(kind, backing, tmp_path):
    frame = _frame(kind)
    frame["target"] = np.sin(np.arange(48) / 4) + np.arange(48) / 13
    frame.loc[32:, "target"] = 1e12
    session = _session(frame, tmp_path, backing)
    before = session.dataset.to_pandas().copy(deep=True)
    split = joblib.hash(session.split_plan)
    cv = KFold(3, shuffle=True, random_state=47)
    result = session.cv_score(
        Ridge(alpha=0.6),
        task="regression",
        cv=cv,
        scoring_metric="rmse",
        preprocess=_recipe(kind, True),
    )
    errors = []
    population = frame.iloc[:32]
    for report, (fitting, validation) in zip(result.folds, cv.split(population), strict=True):
        train, query = population.iloc[fitting], population.iloc[validation]
        x_train, _ = _design(train, train, kind, True)
        x_query, _ = _design(train, query, kind, True)
        predictions = _ridge(x_train, train.target.to_numpy(), x_query)
        rmse = float(np.sqrt(np.mean((predictions - query.target.to_numpy()) ** 2)))
        assert report.metrics["rmse"] == pytest.approx(rmse, abs=1e-7)
        errors.append(rmse)
    assert result.mean_metrics["rmse"] == pytest.approx(np.mean(errors), abs=1e-7)
    assert result.std_metrics["rmse"] == pytest.approx(np.std(errors, ddof=1), abs=1e-7)
    pd.testing.assert_frame_equal(session.dataset.to_pandas(), before)
    assert joblib.hash(session.split_plan) == split


def _collision(kind):
    if kind in {"date", "clock"}:
        return "when_year"
    if kind in {"count", "tfidf"}:
        return "review__red"
    return "x_bin_0" if kind.endswith("onehot") else "x_bin"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("phase", ("fit", "replay"))
@pytest.mark.parametrize("dtype", ("numeric", "text"))
@pytest.mark.parametrize("duplicate", (False, True))
def test_generated_names_cannot_replace_retained_columns(kind, phase, dtype, duplicate):
    frame = _frame(kind, duplicate=duplicate)
    recipe = _recipe(kind)
    processor = build_fold_preprocessor(frame, recipe) if phase == "replay" else None
    frame[_collision(kind)] = np.arange(48) + 500 if dtype == "numeric" else ["original"] * 48
    before = frame.copy(deep=True)
    model_before = None if processor is None else joblib.hash(processor)
    with pytest.raises(ValidationError, match="ambiguous|colli|retained|duplicate"):
        if processor is None:
            build_fold_preprocessor(frame, recipe)
        else:
            processor.transform(frame)
    pd.testing.assert_frame_equal(frame, before)
    if processor is not None:
        assert joblib.hash(processor) == model_before


@pytest.mark.parametrize("kind", KINDS)
def test_duplicate_sources_cannot_create_duplicate_generated_names(kind):
    frame = _frame(kind)
    recipe = _recipe(kind)
    column = (
        "when" if kind in {"date", "clock"} else "review" if kind in {"count", "tfidf"} else "x"
    )
    field = (
        "date_columns"
        if column == "when"
        else "text_columns"
        if column == "review"
        else "binning_columns"
    )
    setattr(recipe, field, (column, column))
    with pytest.raises(ValidationError, match="ambiguous|duplicate"):
        build_fold_preprocessor(frame, recipe)


@pytest.mark.parametrize("strategy", ("uniform", "quantile"))
@pytest.mark.parametrize("encoding", ("ordinal", "onehot"))
@pytest.mark.parametrize("nullable", (False, True))
def test_missing_bins_remain_missing_after_saved_replay(strategy, encoding, nullable, tmp_path):
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.inf, -np.inf]})
    query = pd.DataFrame(
        {
            "x": pd.Series(
                [-np.inf, -9.0, 0.0, 2.5, 5.0, 99.0, np.inf, None],
                dtype="Float64" if nullable else "float64",
            )
        }
    )
    query.index = ["q", "q", "r", "s", "s", "t", "u", "v"]
    processor = build_fold_preprocessor(train, _recipe(strategy + "-" + encoding))
    path = tmp_path / "features.joblib"
    joblib.dump(processor, path)
    before = joblib.hash(processor)
    expected, names = _bins(train.x, query.x, strategy, encoding)
    for candidate in (processor, joblib.load(path)):
        result = candidate.transform(query)
        assert list(result.columns) == names and result.index.equals(query.index)
        np.testing.assert_allclose(result.to_numpy(dtype=float), expected, equal_nan=True)
        assert joblib.hash(candidate) == before


@pytest.mark.parametrize("strategy", ("uniform", "quantile"))
@pytest.mark.parametrize("encoding", ("ordinal", "onehot"))
@pytest.mark.parametrize("duplicate", (False, True))
@pytest.mark.parametrize("scale", (False, True))
def test_binning_sources_are_read_before_generated_names_are_written(
    strategy, encoding, duplicate, scale
):
    train = pd.DataFrame({"x": np.arange(12.0) ** 2, "x_bin": np.arange(12.0)[::-1] ** 3})
    query = pd.DataFrame(
        {"x": [-10.0, 22.0, 200.0], "x_bin": [0.0, 50.0, 3000.0]},
        index=["q", "q", "r"] if duplicate else ["q", "r", "s"],
    )
    recipe = _recipe(strategy + "-" + encoding, scale)
    recipe.binning_columns = ("x", "x_bin")
    processor = build_fold_preprocessor(train, recipe)
    blocks, names, learned = [], [], []
    for column in train:
        values, generated = _bins(train[column], query[column], strategy, encoding, column)
        fit_values, _ = _bins(train[column], train[column], strategy, encoding, column)
        if scale and encoding == "ordinal":
            values = (values - fit_values.mean(axis=0)) / fit_values.std(axis=0)
        blocks.append(values)
        names += generated
        learned.append(fit_values)
    result = processor.transform(query)
    assert list(result.columns) == names
    np.testing.assert_allclose(result.to_numpy(), np.column_stack(blocks), atol=1e-12)


BAD_COUNTS = (True, np.bool_(False), 2.5, np.nan, np.inf, "3", None, 3 + 0j)


@pytest.mark.parametrize("value", BAD_COUNTS)
@pytest.mark.parametrize("strategy", ("uniform", "quantile"))
@pytest.mark.parametrize("route", ("direct", "cv"))
def test_invalid_bin_counts_fail_before_replacing_cv_state(value, strategy, route, tmp_path):
    frame = _frame(strategy + "-ordinal")
    frame["target"] = np.sin(np.arange(48) / 4)
    recipe = _recipe(strategy + "-ordinal")
    recipe.n_bins = value
    if route == "direct":
        with pytest.raises(ValidationError, match="n_bins"):
            build_fold_preprocessor(frame[["x", "keep"]], recipe)
    else:
        session = _session(frame, tmp_path)
        prior = session.cv_score(Ridge(alpha=0.6), cv=3, task="regression", scoring_metric="rmse")
        history, split = joblib.hash(session.history), joblib.hash(session.split_plan)
        before = session.dataset.to_pandas().copy(deep=True)
        with pytest.raises(ValidationError, match="n_bins"):
            session.cv_score(
                Ridge(alpha=0.6), cv=3, task="regression", scoring_metric="rmse", preprocess=recipe
            )
        assert session.last_cv is prior and joblib.hash(session.history) == history
        assert joblib.hash(session.split_plan) == split
        pd.testing.assert_frame_equal(session.dataset.to_pandas(), before)


@pytest.mark.parametrize("kind", KINDS)
def test_ambiguous_cv_recipe_preserves_prior_result_and_session(kind, tmp_path):
    frame = _frame(kind)
    frame[_collision(kind)] = 17.0
    frame["target"] = np.sin(np.arange(48) / 4)
    session = _session(frame, tmp_path)
    prior = session.cv_score(
        Ridge(alpha=0.6),
        task="regression",
        cv=3,
        scoring_metric="rmse",
        preprocess=PreprocessRecipe(impute=None, scale="standard", scale_columns=("keep",)),
    )
    history, split = joblib.hash(session.history), joblib.hash(session.split_plan)
    before = session.dataset.to_pandas().copy(deep=True)
    with pytest.raises(ValidationError, match="ambiguous|colli|retained|duplicate"):
        session.cv_score(
            Ridge(alpha=0.6),
            task="regression",
            cv=3,
            scoring_metric="rmse",
            preprocess=_recipe(kind),
        )
    assert session.last_cv is prior and joblib.hash(session.history) == history
    assert joblib.hash(session.split_plan) == split
    pd.testing.assert_frame_equal(session.dataset.to_pandas(), before)
