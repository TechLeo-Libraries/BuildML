"""Excel extra: Session.ingest reads .xlsx when openpyxl is installed."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import IngestError, MissingExtraError
from buildml.ingest.detect import detect_path_format


def test_detect_xlsx_as_excel() -> None:
    assert detect_path_format(Path("loan.xlsx")) == "excel"
    assert detect_path_format(Path("loan.xlsm")) == "excel"
    assert detect_path_format(Path("loan.xls")) == "unknown"


def test_session_ingest_xlsx(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    path = tmp_path / "tiny.xlsx"
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    frame.to_excel(path, index=False)
    session = Session.ingest(path)
    assert session.ingest_report is not None
    assert session.ingest_report.format_name == "excel"
    assert session.dataset.n_rows == 3
    assert list(session.to_pandas().columns) == ["a", "b"]


def test_excel_ingest_refuses_polars_engine(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    path = tmp_path / "tiny.xlsx"
    pd.DataFrame({"a": [1, 2]}).to_excel(path, index=False)
    with pytest.raises(IngestError, match="openpyxl"):
        Session.ingest(path, engine="polars")


def test_excel_missing_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "tiny.xlsx"
    path.write_bytes(b"not-a-real-workbook")
    import builtins

    import buildml.ingest.loaders as loaders

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "openpyxl":
            raise ImportError("no openpyxl")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(MissingExtraError, match="buildml\\[excel\\]"):
        loaders.load_excel(path)
