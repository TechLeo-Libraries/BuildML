"""Public-table loaders for classical / AutoML proofs.

Imports ``proofs/_lib/datasets.py`` by path so ``proofs._lib.env`` (Torch)
never loads in this process.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _datasets():
    path = ROOT / "proofs" / "_lib" / "datasets.py"
    spec = importlib.util.spec_from_file_location("proofs_lib_datasets_isolated", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_breast_cancer_is_offline_public() -> None:
    ds = _datasets()
    frame, meta = ds.load_sklearn_breast_cancer()
    assert meta["evidence_tier"] == "REAL_PUBLIC_DATASET"
    assert meta["offline_safe"] is True
    assert meta["target"] in frame.columns
    assert meta["feature_columns"]
    roles = ds.supervised_roles(meta)
    assert roles[meta["target"]] == "target"
    assert all(roles[col] == "feature" for col in meta["feature_columns"])


def test_classical_credit_table_records_which_loader_ran() -> None:
    ds = _datasets()
    frame, meta = ds.load_classical_credit_table(seed=0)
    assert meta["loader_selected"] in {
        "openml_credit_g",
        "credit_approval_synthetic",
    }
    assert meta["target"] in frame.columns
    assert meta["feature_columns"]
    roles = ds.supervised_roles(meta)
    assert roles[str(meta["target"])] == "target"
    for col in meta.get("ignore_columns") or ():
        assert roles[str(col)] == "ignore"


def test_credit_fallback_is_disclosed(monkeypatch) -> None:
    ds = _datasets()

    def _boom() -> tuple[object, dict]:
        raise RuntimeError("offline")

    monkeypatch.setattr(ds, "load_openml_credit_g", _boom)
    frame, meta = ds.load_classical_credit_table(seed=1)
    assert meta["loader_selected"] == "credit_approval_synthetic"
    assert meta["real_public_dataset"] is False
    assert "fallback_reason" in meta
    assert len(frame) == 1200


def test_infer_feature_kinds_splits_mixed_credit_draw() -> None:
    ds = _datasets()
    frame, meta = ds.load_credit_approval_synthetic(n=40, seed=0)
    numeric, categorical = ds.infer_feature_kinds(frame, list(meta["feature_columns"]))
    assert "age" in numeric
    assert "region" in categorical


def test_classical_and_automl_proofs_call_public_loaders() -> None:
    """Parse scripts as text so proofs._lib.env (Torch) never loads."""
    loan = (ROOT / "proofs" / "loan-approval-classical" / "script.py").read_text(
        encoding="utf-8"
    )
    mortgage = (ROOT / "proofs" / "mortgage-default-classical" / "script.py").read_text(
        encoding="utf-8"
    )
    churn = (ROOT / "proofs" / "churn-automl-search" / "script.py").read_text(
        encoding="utf-8"
    )
    assert "load_classical_credit_table" in loan
    assert "load_credit_approval_synthetic(" not in loan
    assert "load_classical_credit_table" in mortgage
    assert "load_mortgage_default_synthetic" not in mortgage
    assert "load_sklearn_breast_cancer" in churn
    assert "load_telco_churn_synthetic" not in churn
