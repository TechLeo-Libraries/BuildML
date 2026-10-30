"""Nullable numeric synthesis preserves missingness through real consumers."""

import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from buildml import Session
from buildml.core.errors import ValidationError

BACKINGS = ["pandas", "pandas-parquet", "polars-memory", "polars-lazy", "duckdb-memory", "duckdb-lazy"]
DTYPES = ["Int64", "Int32", "UInt32"]
COLS = ["signal", "count", "value", "group", "target", "constant"]
TRAIN = list(range(128))


def _frame(dtype="Int64", missing=True):
    i = np.arange(192)
    count = pd.Series(1000 + (i % 12)*5, dtype=dtype)
    value = pd.Series(np.sin(i/7)+i/100)
    group = pd.Series(np.where(i % 3, "common", "other"), dtype=object)
    if missing:
        count.iloc[::5] = pd.NA
        value.iloc[::7] = np.nan
        group.iloc[::9] = None
    frame = pd.DataFrame(dict(signal=i/20, count=count, value=value, group=group,
                              target=2+i/10+np.cos(i), constant=7.5))
    frame.loc[128:, "group"] = "holdout-only"
    frame.loc[128:, "signal"] += 1000
    frame.index = [f"row-{n//2}" for n in i]
    return frame[COLS]


def _session(tmp_path, dtype="Int64", missing=True, backing="pandas"):
    raw = _frame(dtype, missing)
    engine = backing.split("-")[0]
    if engine != "pandas":
        pytest.importorskip(engine)
    if backing.endswith(("lazy", "parquet")):
        path = tmp_path / "source.parquet"
        raw.to_parquet(path, index=False)
        session = Session.ingest(path, engine=engine, mode="lazy" if engine != "pandas" else "memory")
    else:
        session = Session.ingest(raw, engine=engine)
    session.set_roles({"signal": "feature", "count": "ignore", "value": "ignore",
                       "group": "ignore", "target": "target", "constant": "ignore"})
    session.inject_split(train_indices=TRAIN, validation_indices=list(range(128,160)), test_indices=list(range(160,192)))
    return session


def _numbers(series):
    return pd.to_numeric(series).to_numpy(dtype=float, na_value=np.nan)


def _bootstrap_reference(train, n, sigma, seed):
    """Independent row draw, sample-variance noise and integer rounding oracle."""
    rng = np.random.default_rng(seed)
    result = train.iloc[rng.integers(0, len(train), size=n)].reset_index(drop=True).copy()
    integer = [c for c in COLS if pd.api.types.is_integer_dtype(train[c].dtype)]
    continuous = [c for c in COLS if pd.api.types.is_numeric_dtype(train[c].dtype) and c not in integer]
    if sigma:
        for column in continuous + integer:
            observed = [float(v) for v in train[column] if not pd.isna(v)]
            mean = sum(observed)/len(observed)
            sd = (sum((v-mean)**2 for v in observed)/(len(observed)-1))**.5
            if sd:
                values = _numbers(result[column]) + rng.normal(0, sigma*sd, n)
                result[column] = np.rint(values) if column in integer else values
    return result


def _assert_sample(actual, expected):
    assert list(actual) == list(expected) and len(actual) == len(expected)
    np.testing.assert_array_equal(actual.isna(), expected.isna())
    for column in COLS:
        if column == "group":
            np.testing.assert_array_equal(actual[column].dropna().to_numpy(), expected[column].dropna().to_numpy())
        else:
            np.testing.assert_allclose(_numbers(actual[column]), _numbers(expected[column]), rtol=1e-12, atol=2e-12, equal_nan=True)


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("sigma", [0., .25])
def test_bootstrap_draws_noise_null_masks_and_saved_replay(backing, dtype, missing, sigma, tmp_path):
    session = _session(tmp_path, dtype, missing, backing)
    try:
        session.fit(Ridge(alpha=.6), task="regression")
        original, split, fitted = session.to_pandas().copy(), session.split_plan, session.fit_result
        train = original.iloc[TRAIN].reset_index(drop=True)
        expected = _bootstrap_reference(train, 256, sigma, 17)
        session.synthetic.fit(method="bootstrap", columns=COLS, smooth_sigma=sigma, random_state=3)
        path = session.synthetic.save_bundle(tmp_path / "generator")
        artifacts = {p.relative_to(path):p.read_bytes() for p in Path(path).rglob("*") if p.is_file()}
        for replay in [False, True]:
            if replay:
                session.synthetic.load_bundle(path, trusted=True)
            frozen = joblib.hash(session.synthesizer_plan)
            result = session.synthetic.sample(n=256, random_state=17)
            _assert_sample(result.frame, expected)
            if missing:
                assert result.frame["count"].isna().any()
            assert "holdout-only" not in set(result.frame.group.dropna())
            assert joblib.hash(session.synthesizer_plan) == frozen
            assert session.split_plan is split and session.fit_result is fitted
            pd.testing.assert_frame_equal(session.to_pandas(), original)
        assert artifacts == {p.relative_to(path):p.read_bytes() for p in Path(path).rglob("*") if p.is_file()}
    finally:
        session.close_native()


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("condition", [None, {"count": None}, {"count": 1000}])
def test_copula_numeric_marginals_missingness_and_exact_conditions(backing, dtype, condition, tmp_path):
    session = _session(tmp_path, dtype, backing=backing)
    try:
        before, split = session.to_pandas().copy(), session.split_plan
        train = before.iloc[TRAIN]
        session.synthetic.fit(method="gaussian_copula", columns=COLS, random_state=3)
        path = session.synthetic.save_bundle(tmp_path / "generator")
        first = session.synthetic.sample(n=4096 if condition is None else 48, random_state=23, condition=condition)
        if condition is None:
            for column in ["count", "value", "group"]:
                assert first.frame[column].isna().mean() == pytest.approx(train[column].isna().mean(), abs=.05)
            observed = set(train["count"].dropna().astype(int))
            assert set(first.frame["count"].dropna().astype(int)) == observed
            counts = first.frame["count"].dropna().astype(int)
            reference = train["count"].dropna().astype(int)
            for value in observed:
                assert counts.eq(value).mean() == pytest.approx(reference.eq(value).mean(), abs=.04)
            assert first.frame.value.dropna().isin(train.value.dropna()).all()
        elif condition["count"] is None:
            assert first.frame["count"].isna().all()
        else:
            assert first.frame["count"].eq(1000).all()
        assert first.frame.constant.eq(7.5).all()
        assert "holdout-only" not in set(first.frame.group.dropna())
        session.synthetic.load_bundle(path, trusted=True)
        frozen = joblib.hash(session.synthesizer_plan)
        second = session.synthetic.sample(n=first.n_rows, random_state=23, condition=condition)
        pd.testing.assert_frame_equal(first.frame, second.frame)
        assert joblib.hash(session.synthesizer_plan) == frozen and session.split_plan is split
        pd.testing.assert_frame_equal(session.to_pandas(), before)
    finally:
        session.close_native()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("method", ["bootstrap", "gaussian_copula"])
@pytest.mark.parametrize("replay", [False, True])
def test_nullable_generated_train_merge_imputation_and_real_ridge_consumer(dtype, method, replay, tmp_path):
    session = _session(tmp_path, dtype)
    session.fit(Ridge(alpha=.6), task="regression")
    before, split = session.to_pandas().copy(), session.split_plan
    session.synthetic.fit(method=method, columns=COLS, smooth_sigma=.25, random_state=3)
    if replay:
        path = session.synthetic.save_bundle(tmp_path / "generator")
        session.synthetic.load_bundle(path, trusted=True)
    frozen = copy.deepcopy(session.synthesizer_plan)
    expected = session.synthetic.sample(n=256, random_state=17).frame.copy()
    assert expected["count"].isna().any()
    generated = session.synthetic.sample(n=256, random_state=17, merge_mode="extend_train", provenance_column="generated")
    assert generated.merged and session.fit_result is None
    assert session.dataset.roles["generated"].value == "ignore"
    assert session.synthesizer_plan.to_dict() == frozen.to_dict()
    if method == "bootstrap":
        # Pandas can populate read-only dtype/index caches during concatenation.
        # Compare every learned value and parameter independently of those caches.
        generator, original = session.synthesizer_plan.generator_, frozen.generator_
        pd.testing.assert_frame_equal(generator.frame, original.frame)
        assert {k:v for k,v in vars(generator).items() if k != "frame"} == {
            k:v for k,v in vars(original).items() if k != "frame"}
    else:
        assert joblib.hash(session.synthesizer_plan) == joblib.hash(frozen)
    train = session.partition("train")
    assert train.generated.sum() == 256
    _assert_sample(train.iloc[-256:][COLS].reset_index(drop=True), expected)
    for partition in ["validation", "test"]:
        actual = session.partition(partition)
        assert not actual.generated.any()
        pd.testing.assert_frame_equal(actual[COLS].reset_index(drop=True), before.iloc[list(getattr(split, partition+"_indices"))].reset_index(drop=True))
    session.set_roles({"count":"feature"})
    median = float(train["count"].median())
    design = np.column_stack([train.signal, train["count"].fillna(median).astype(float)])
    outcomes = train.target.to_numpy()
    centered = design-design.mean(axis=0)
    beta = np.linalg.solve(centered.T@centered+.6*np.eye(2), centered.T@(outcomes-outcomes.mean()))
    session.impute(columns=["count"], strategy="median").fit(Ridge(alpha=.6), task="regression")
    test = session.partition("test")
    test_design = np.column_stack([test.signal, test["count"].astype(float)])
    reference = outcomes.mean()+(test_design-design.mean(axis=0))@beta
    np.testing.assert_allclose(session.predict(partition="test"), reference, atol=2e-9)
    assert session.fit_result.feature_columns == ("signal", "count")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("method", ["bootstrap", "gaussian_copula"])
@pytest.mark.parametrize("fractional", [True, False])
def test_loaded_generator_numeric_compatibility_and_atomic_refusal(dtype, method, fractional, tmp_path):
    frame = _frame(dtype)
    frame["count"] = frame["count"].astype("Float64") + (.25 if fractional else 2**40)
    donor = Session.ingest(frame).set_roles({"signal":"feature", "target":"target"})
    donor.inject_split(train_indices=TRAIN, validation_indices=list(range(128,160)), test_indices=list(range(160,192)))
    donor.synthetic.fit(method=method, columns=COLS, random_state=3)
    path = donor.synthetic.save_bundle(tmp_path / "generator")
    session = _session(tmp_path, dtype)
    session.fit(Ridge(alpha=.6), task="regression")
    session.synthetic.load_bundle(path, trusted=True)
    prior_sample = session.synthetic.sample(n=64, random_state=17)
    before, split, fitted = session.to_pandas().copy(), session.split_plan, session.fit_result
    history = list(session.history)
    frozen = joblib.hash(session.synthesizer_plan)
    if not fractional and dtype == "Int64":
        # These integral values fit Int64; a wider training range is valid.
        result = session.synthetic.sample(n=64, random_state=17, merge_mode="extend_train")
        assert result.merged and session.partition("train")["count"].max() > 2**40
    else:
        with pytest.raises(ValidationError):
            session.synthetic.sample(n=64, random_state=17, merge_mode="extend_train")
        assert session.synthetic_sample_result is prior_sample and session.split_plan is split
        assert session.fit_result is fitted and session.history == history
        assert joblib.hash(session.synthesizer_plan) == frozen
        pd.testing.assert_frame_equal(session.to_pandas(), before)
