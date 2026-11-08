"""Resume real native checkpoints with the engines available in this process."""

import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from buildml import Session
from buildml.checkpoint.bundle import save_checkpoint
from buildml.core.errors import MissingExtraError, ValidationError
from buildml.core.types import ColumnRole, DataMode, EngineName
from buildml.data.dataset import Dataset
from buildml.data.engines import prepare_design_frame

ROLES = {"x": "feature", "z": "feature", "y": "target"}
FEATURES = ["x", "z"]


def _frame():
    x = np.arange(32, dtype=float) / 7
    return pd.DataFrame({"x": x, "z": np.sin(x), "y": 3 * x + np.cos(x)})


def _split(session):
    return session.set_roles(ROLES).split(test_size=0.25, validation_size=0.25, random_state=13)


def _dataset(frame, engine, mode="lazy"):
    return Dataset.from_pandas(
        frame,
        engine=EngineName(engine),
        mode=DataMode(mode),
        roles={name: ColumnRole(role) for name, role in ROLES.items()},
    )


@pytest.fixture(scope="module")
def portable_snapshots(tmp_path_factory):
    """Use actual producer artifacts, including when the reader has no native engine."""
    root = tmp_path_factory.mktemp("portable-native-checkpoints")
    archive = Path(__file__).parents[1] / "fixtures" / "native-checkpoint-portability.zip"
    with ZipFile(archive) as snapshot:
        assert all(name.endswith((".json", ".parquet")) for name in snapshot.namelist())
        snapshot.extractall(root)
    pd.testing.assert_frame_equal(pd.read_parquet(root / "raw.parquet"), _frame())
    return root


def _check_predictions(session, frame, split):
    """Fit the real estimator and check ordered held-out values with NumPy OLS."""
    design = np.column_stack([np.ones(len(frame)), frame[FEATURES].to_numpy()])
    target = frame.y.to_numpy()
    train = list(split["train_indices"])
    beta = np.linalg.lstsq(design[train], target[train], rcond=None)[0]
    assert session.fit_result is None
    session.fit(LinearRegression(), task="regression")
    predictions = {}
    for partition in ("validation", "test"):
        rows = list(split[f"{partition}_indices"])
        expected = design[rows] @ beta
        observed = session.predict(partition=partition)
        np.testing.assert_allclose(observed.to_numpy(), expected, rtol=2e-11, atol=2e-11)
        np.testing.assert_array_equal(session.partition(partition).y.to_numpy(), target[rows])
        metrics = session.evaluate(partition=partition).metrics
        residuals = target[rows] - expected
        assert metrics["mae"] == pytest.approx(float(np.abs(residuals).mean()), abs=2e-11)
        assert metrics["r2"] == pytest.approx(
            1 - np.square(residuals).sum() / np.square(target[rows] - target[rows].mean()).sum()
        )
        predictions[partition] = observed.tolist()
    return predictions


@pytest.mark.parametrize("engine", ["duckdb", "polars"])
@pytest.mark.parametrize("layout", ["canonical", "single", "partitioned"])
@pytest.mark.parametrize("method", ["standard", "minmax"])
def test_checkpoint_resumes_preprocessing_and_predictions(
    engine, layout, method, tmp_path, portable_snapshots
):
    frame = pd.read_parquet(portable_snapshots / "raw.parquet")
    installed = importlib.util.find_spec(engine) is not None
    if layout == "canonical":
        # Public canonical-only snapshot carrying the requested engine as metadata.
        plan = _split(Session.ingest(frame)).split_plan
        path = save_checkpoint(
            tmp_path / "canonical", dataset=_dataset(frame, engine), split_plan=plan
        )
    else:
        path = portable_snapshots / f"{engine}-{layout}"
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        assert meta["native_sidecar"]["layout"] == layout
        assert meta["has_native"] and not meta["has_plans"]
    split = json.loads((path / "splits.json").read_text(encoding="utf-8"))
    session = Session.checkpoint_load(path)
    try:
        assert session.reattach_result.status == "resume"
        assert session.dataset.engine == EngineName(engine)
        assert session.dataset.mode == DataMode.LAZY
        assert session.dataset.has_native is installed
        assert session.split_plan.to_dict() == split
        assert session.dataset.roles == {name: ColumnRole(role) for name, role in ROLES.items()}
        pd.testing.assert_frame_equal(session.to_pandas(), frame, check_dtype=False)
        if not installed:
            assert any("pandas" in message.lower() for message in session.reattach_result.messages)
            # Explicit native synchronization continues to disclose a missing engine.
            with pytest.raises(MissingExtraError, match=engine):
                session.sync_native()

        train = frame.iloc[list(split["train_indices"])][FEATURES].to_numpy()
        center = train.mean(axis=0) if method == "standard" else train.min(axis=0)
        divisor = train.std(axis=0) if method == "standard" else np.ptp(train, axis=0)
        expected = frame.copy()
        expected[FEATURES] = (frame[FEATURES].to_numpy() - center) / divisor
        session.scale(columns=FEATURES, method=method)
        assert session.dataset.engine == EngineName(engine)
        assert session.dataset.has_native is installed
        assert session.split_plan.to_dict() == split
        pd.testing.assert_frame_equal(session.to_pandas(), expected, check_dtype=False, atol=2e-12)
        prepared = prepare_design_frame(session.dataset, ["z", "x", "z"])
        assert prepared.engine == (engine if installed else "pandas")
        assert prepared.used_native_handle is installed
        assert prepared.columns_materialized == ("z", "x")
        pd.testing.assert_frame_equal(prepared.frame, expected[["z", "x"]], check_dtype=False)
        if not installed:
            assert any(f"{engine} is unavailable" in note for note in prepared.disclosures)
            sampled = prepare_design_frame(session.dataset, ["x"], sample_rows=5, random_state=5)
            positions = np.random.RandomState(5).choice(len(expected), size=5, replace=False)
            assert sampled.sampled and sampled.n_rows_materialized == 5
            pd.testing.assert_frame_equal(sampled.frame, expected.iloc[positions][["x"]])
        predictions = _check_predictions(session, expected, split)

        # A second save after real prep must also be usable in this environment.
        saved = session.checkpoint_save(
            tmp_path / "continued", sidecar_layout="partitioned", sidecar_partition_rows=7
        )
        with pytest.raises(ValidationError, match="trusted"):
            Session.checkpoint_load(saved)
        continued = Session.checkpoint_load(saved, trusted=True)
        try:
            assert continued.dataset.engine == EngineName(engine)
            assert continued.dataset.has_native is installed
            assert continued.split_plan.to_dict() == split
            assert continued.scale_plan is not None
            assert continued.fit_result is None
            pd.testing.assert_frame_equal(
                continued.to_pandas(), expected, check_dtype=False, atol=2e-12
            )
        finally:
            continued.close_native()
        print(
            "PORTABLE_PREDICTIONS",
            json.dumps(
                {
                    "engine": engine,
                    "layout": layout,
                    "method": method,
                    "native": installed,
                    "predictions": predictions,
                }
            ),
        )
    finally:
        session.close_native()


@pytest.mark.parametrize("engine", ["duckdb", "polars"])
@pytest.mark.parametrize("mode", ["memory", "lazy"])
@pytest.mark.parametrize("sync_native", [False, True])
def test_pandas_backed_transform_preserves_engine_hint(engine, mode, sync_native):
    original = _frame()
    source = _dataset(original, engine, mode)
    assert not source.has_native
    expected = original.assign(x=original.x + 2, z=original.z * 3)
    transformed = Dataset.from_transformed(source, expected, sync_native=sync_native)
    try:
        assert transformed.has_native is (
            sync_native and importlib.util.find_spec(engine) is not None
        )
        assert transformed.engine == EngineName(engine)
        assert transformed.mode == DataMode(mode)
        assert transformed.roles == source.roles
        pd.testing.assert_frame_equal(transformed.to_pandas(), expected, check_dtype=False)
        pd.testing.assert_frame_equal(source.to_pandas(), original)
    finally:
        transformed.close_native()


@pytest.mark.parametrize("suffix", [".parquet", ".PARQUET"])
@pytest.mark.parametrize("data_only", [False, True])
def test_bare_parquet_load_is_fresh_input_for_real_prediction(suffix, data_only, tmp_path):
    frame = _frame()
    path = tmp_path / f"rows{suffix}"
    frame.to_parquet(path, index=False)
    session = Session.checkpoint_load(path, data_only=data_only)
    assert session.reattach_result.status == "fresh_ingest"
    assert session.dataset.engine == EngineName.PANDAS
    assert session.split_plan is None and not session.dataset.roles
    pd.testing.assert_frame_equal(session.to_pandas(), frame)
    _split(session)
    _check_predictions(session, frame, session.split_plan.to_dict())


def test_checkpoint_directory_without_data_refuses(tmp_path):
    root = tmp_path / "incomplete"
    root.mkdir()
    with pytest.raises(ValidationError, match="Checkpoint data not found"):
        Session.checkpoint_load(root)


@pytest.mark.parametrize("compression", ["", "   "])
def test_blank_sidecar_codec_refuses_before_writing(compression, tmp_path):
    session = _split(Session.ingest(_frame()))
    root = tmp_path / "invalid"
    with pytest.raises(ValidationError, match="sidecar_compression"):
        session.checkpoint_save(root, sidecar_compression=compression)
    assert not root.exists()
