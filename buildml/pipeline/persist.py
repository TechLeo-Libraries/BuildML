"""Persist and reload fitted modeling artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.core.errors import ValidationError
from buildml.model.supervised import FitResult


def save_fit_result(path: str | Path, fit_result: FitResult) -> Path:
    """Save a fitted estimator bundle to a directory.

    Layout
    ------
    ``model.joblib``, ``meta.json``
    """
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    joblib.dump(fit_result.estimator, root / "model.joblib")
    meta = {
        "buildml_version": __version__,
        **fit_result.to_dict(),
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return root


def load_fit_result(path: str | Path) -> FitResult:
    """Load a fitted estimator bundle saved by :func:`save_fit_result`."""
    root = Path(path)
    model_path = root / "model.joblib"
    meta_path = root / "meta.json"
    if not model_path.exists() or not meta_path.exists():
        raise ValidationError(f"Fit artifact incomplete at '{root}'")
    estimator = joblib.load(model_path)
    meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    return FitResult(
        estimator=estimator,
        task=meta["task"],
        feature_columns=tuple(meta["feature_columns"]),
        target_column=meta["target_column"],
        n_train_rows=int(meta["n_train_rows"]),
    )
