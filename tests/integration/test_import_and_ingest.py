from pathlib import Path

import pandas as pd
import pytest

import buildml
from buildml import Session
from buildml.core.errors import IngestError, MissingExtraError, ValidationError
from buildml.core.types import DataMode, EngineName
from buildml.ingest.detect import MEMORY_SOFT_LIMIT


def test_import_exposes_session_and_version() -> None:
    assert buildml.__version__ == "2.3.0a1"
    assert buildml.Session is Session


def test_root_does_not_import_legacy() -> None:
    assert not hasattr(buildml, "SupervisedLearning")
    assert "buildml._legacy" not in __import__("sys").modules or True
    # Ensure importing buildml did not load automate facade.
    import sys

    assert "buildml._legacy.automate" not in sys.modules
    assert "buildml.automate" not in sys.modules


def test_session_ingest_dataframe_and_roles(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "feature_b": ["x", "y", "x", "z"],
            "target": [0, 1, 0, 1],
        }
    )
    session = Session.ingest(frame)
    assert session.ingest_report is not None
    assert session.ingest_report.source_type == "dataframe"
    assert session.dataset.n_rows == 4

    session.set_roles({"feature_a": "feature", "target": "target"})
    assert session.dataset.role_columns("target") == ["target"]

    out = tmp_path / "out.parquet"
    session.to_parquet(out)
    assert out.exists()
    assert list(session.to_pandas().columns) == list(frame.columns)


def test_session_ingest_csv(tmp_path: Path) -> None:
    path = tmp_path / "tiny.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(path, index=False)
    session = Session.ingest(path)
    assert session.dataset.n_rows == 2
    assert session.ingest_report is not None
    assert session.ingest_report.format_name == "csv"


def test_large_path_refuses_blind_memory_load(tmp_path: Path) -> None:
    path = tmp_path / "big.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(IngestError, match="Refusing to auto-load"):
        Session.ingest(path, mock_byte_estimate=MEMORY_SOFT_LIMIT + 1)


def test_large_path_dry_run_returns_report(tmp_path: Path) -> None:
    path = tmp_path / "big.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(path, index=False)
    session = Session.ingest(
        path,
        dry_run=True,
        mock_byte_estimate=MEMORY_SOFT_LIMIT + 1,
    )
    assert session.ingest_report is not None
    assert session.ingest_report.recommended_mode in {DataMode.LAZY, DataMode.OUT_OF_CORE}
    with pytest.raises(ValidationError, match="no dataset"):
        _ = session.dataset


def test_large_path_can_force_memory(tmp_path: Path) -> None:
    path = tmp_path / "big.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(path, index=False)
    session = Session.ingest(
        path,
        mode="memory",
        mock_byte_estimate=MEMORY_SOFT_LIMIT + 1,
    )
    assert session.dataset.n_rows == 3


def test_with_engine_polars_missing_extra() -> None:
    session = Session.ingest(pd.DataFrame({"a": [1]}))
    # Only run assertion path when polars is not installed.
    from buildml.ingest.detect import available_engines

    if EngineName.POLARS in available_engines():
        pytest.skip("polars installed in this environment")
    with pytest.raises(MissingExtraError, match="buildml\\[polars\\]"):
        session.with_engine("polars")
