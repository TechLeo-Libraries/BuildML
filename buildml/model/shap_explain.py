"""Optional SHAP attribution helpers for fitted classical estimators."""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from buildml.core.errors import MissingExtraError, ValidationError


def shap_spec_present() -> bool:
    """Whether the shap package is installed (spec only)."""
    return importlib.util.find_spec("shap") is not None


def shap_available() -> bool:
    """Whether shap imports cleanly."""
    if not shap_spec_present():
        return False
    try:
        import shap  # noqa: F401
    except Exception:
        return False
    return True


def require_shap(*, feature: str = "SHAP attribution") -> Any:
    """Import shap or raise MissingExtraError."""
    try:
        import shap
    except Exception as exc:  # noqa: BLE001
        raise MissingExtraError("shap", feature) from exc
    return shap


@dataclass(slots=True)
class ShapExplainResult:
    """Mean |SHAP| importances plus honesty disclosures."""

    backend: str
    n_rows: int
    n_features: int
    feature_names: tuple[str, ...]
    mean_abs_shap: dict[str, float]
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def explain_with_shap(
    estimator: Any,
    frame: pd.DataFrame,
    *,
    max_samples: int = 100,
    random_state: int | None = 0,
) -> ShapExplainResult:
    """Compute mean absolute SHAP values for a fitted estimator.

    Prefers TreeExplainer; falls back to generic Explainer. Caps rows for cost.
    """
    shap = require_shap(feature="explain_with_shap")
    if frame.empty:
        raise ValidationError("explain_with_shap requires a non-empty feature frame.")
    n = int(len(frame))
    take = min(int(max_samples), n)
    if take < n:
        sampled = frame.sample(n=take, random_state=random_state)
    else:
        sampled = frame
    x = sampled.to_numpy(dtype=float)
    warnings: list[str] = []
    backend = "explainer"
    try:
        explainer = shap.TreeExplainer(estimator)
        backend = "tree_explainer"
    except Exception:
        explainer = shap.Explainer(estimator, sampled)
        warnings.append("TreeExplainer unavailable; used generic Explainer.")
    values = explainer(x)
    raw = getattr(values, "values", values)
    arr = np.asarray(raw)
    if arr.ndim == 3:
        # (rows, features, classes) → mean over classes
        arr = np.mean(np.abs(arr), axis=2)
    else:
        arr = np.abs(arr)
    mean_abs = arr.mean(axis=0)
    names = tuple(str(c) for c in sampled.columns)
    mapping = {
        name: float(mean_abs[i]) if i < len(mean_abs) else float("nan")
        for i, name in enumerate(names)
    }
    return ShapExplainResult(
        backend=backend,
        n_rows=int(take),
        n_features=len(names),
        feature_names=names,
        mean_abs_shap=mapping,
        disclosures=(
            "SHAP values are model attributions on the requested partition sample, "
            "not causal effects.",
            f"Computed on at most {max_samples} rows for cost control.",
            "Optional dependency: pip install 'buildml[shap]'.",
        ),
        warnings=tuple(warnings),
    )
