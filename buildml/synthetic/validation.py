"""Post-sample validation for synthetic frames (built-in + optional GE)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.synthetic.extras import great_expectations_available
from buildml.synthetic.results import SynthesizerPlan


@dataclass(slots=True)
class SyntheticValidationResult:
    """Outcome of validate_synthetic on a sampled frame."""

    passed: bool
    n_rows: int
    n_checks: int
    n_failed: int
    checks: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def validate_synthetic(
    plan: SynthesizerPlan,
    frame: pd.DataFrame,
    *,
    null_rate_tolerance: float = 0.15,
    numeric_range_slack: float = 0.25,
    categorical_oov_tolerance: float = 0.05,
    run_great_expectations: bool = True,
) -> SyntheticValidationResult:
    """Validate a synthetic sample against the fitted plan schema.

Always runs built-in checks. When ``great_expectations`` is importable and
``run_great_expectations=True``, adds lite GE column-presence expectations.

Parameters
----------
plan:
    Fitted plan object carrying model state and feature contract.
frame:
    Partition or full DataFrame slice used for this operation.
null_rate_tolerance:
    null rate tolerance (float).
numeric_range_slack:
    numeric range slack (float).
categorical_oov_tolerance:
    categorical oov tolerance (float).
run_great_expectations:
    run great expectations (bool).

Returns
-------
SyntheticValidationResult
    Serializable result summary (SyntheticValidationResult) for history recording.

Raises
------
ValidationError
    When preconditions for this operation are not met.
    """
    if plan is None:
        raise ValidationError("No SynthesizerPlan for validate_synthetic.")
    if frame is None or frame.empty:
        raise ValidationError("Cannot validate an empty synthetic frame.")

    cols = [c for c in plan.columns if c in frame.columns]
    missing = [c for c in plan.columns if c not in frame.columns]
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    warnings: list[str] = []

    checks["columns_present"] = len(missing) == 0
    if missing:
        warnings.append(f"Missing modeled columns in sample: {missing[:8]}")

    checks["row_count_positive"] = len(frame) >= 1
    metrics["n_rows"] = float(len(frame))
    metrics["n_columns_modeled"] = float(len(cols))

    kind_map = {s.name: s.kind for s in plan.column_specs}
    failed = 0
    for col in cols:
        kind = kind_map.get(col, "continuous")
        series = frame[col]
        spec = next((s for s in plan.column_specs if s.name == col), None)
        train_null_rate = 0.0 if spec is None else spec.n_null / max(plan.n_rows_fitted, 1)
        syn_null_rate = float(series.isna().mean())
        null_ok = abs(syn_null_rate - train_null_rate) <= null_rate_tolerance
        checks[f"null_rate::{col}"] = null_ok
        metrics[f"null_rate::{col}"] = syn_null_rate
        if not null_ok:
            failed += 1
            warnings.append(
                f"Column {col!r} null rate {syn_null_rate:.3f} diverges from "
                f"train {train_null_rate:.3f} beyond tolerance {null_rate_tolerance}."
            )

        if kind == "categorical" and spec is not None and spec.categories:
            vocab = set(spec.categories)
            oov = series.astype("string").fillna("__NA__").map(lambda v: v not in vocab)
            oov_rate = float(oov.mean())
            metrics[f"oov_rate::{col}"] = oov_rate
            oov_ok = oov_rate <= categorical_oov_tolerance
            checks[f"categorical_vocab::{col}"] = oov_ok
            if not oov_ok:
                failed += 1
                warnings.append(
                    f"Column {col!r} has OOV rate {oov_rate:.3f} "
                    f"> tolerance {categorical_oov_tolerance}."
                )
        elif kind in {"continuous", "integer"}:
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().any():
                metrics[f"mean::{col}"] = float(numeric.mean())
                if spec is not None and spec.extras.get("train_min") is not None:
                    t_min = float(spec.extras["train_min"])
                    t_max = float(spec.extras["train_max"])
                    span = max(t_max - t_min, 1e-9)
                    slack = numeric_range_slack * span
                    out_of_range = (numeric < t_min - slack) | (numeric > t_max + slack)
                    oor_rate = float(out_of_range.mean())
                    metrics[f"oor_rate::{col}"] = oor_rate
                    checks[f"numeric_range::{col}"] = oor_rate <= categorical_oov_tolerance
                    if oor_rate > categorical_oov_tolerance:
                        failed += 1

    ge_checks = 0
    if run_great_expectations and great_expectations_available():
        ge_checks, ge_failed, ge_warn = _run_ge_lite(frame, plan.columns)
        checks.update(ge_checks)
        failed += ge_failed
        warnings.extend(ge_warn)

    n_failed = sum(0 if ok else 1 for ok in checks.values())
    passed = n_failed == 0
    disclosures = [
        f"Validated n={len(frame)} synthetic rows against plan method={plan.method!r}.",
        "Built-in checks: column presence, null rates, categorical vocabulary, "
        "numeric range slack: not a privacy or leakage audit.",
    ]
    if ge_checks:
        disclosures.append(
            "Great Expectations lite column-presence expectations appended when installed."
        )

    return SyntheticValidationResult(
        passed=passed,
        n_rows=int(len(frame)),
        n_checks=len(checks),
        n_failed=n_failed,
        checks=checks,
        metrics=metrics,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def _run_ge_lite(
    frame: pd.DataFrame,
    columns: tuple[str, ...] | list[str],
) -> tuple[dict[str, bool], int, list[str]]:
    """Lite GE checks via legacy PandasDataset when available."""
    checks: dict[str, bool] = {}
    warnings: list[str] = []
    failed = 0
    try:
        from great_expectations.dataset import PandasDataset
    except ImportError:
        return checks, failed, warnings

    ds = PandasDataset(frame)
    for col in columns:
        key = f"ge_column_present::{col}"
        try:
            result = ds.expect_column_to_exist(col)
            ok = bool(result.get("success", False))
        except Exception as exc:  # noqa: BLE001
            ok = col in frame.columns
            warnings.append(f"GE expectation failed for {col!r}: {exc}")
        checks[key] = ok
        if not ok:
            failed += 1
    return checks, failed, warnings


def enrich_specs_with_train_stats(train: pd.DataFrame, specs: tuple[Any, ...]) -> tuple[Any, ...]:
    """Attach train min/max to numeric specs for validation range checks.

Called from the Session-facing workflow after splits and roles are set. Validation and test partitions are evaluation-only unless explicitly documented.

Parameters
----------
train:
    train (pd.DataFrame).
specs:
    specs (tuple[Any, ...]).

Returns
-------
tuple[Any, ...]
    Tuple of results (tuple[Any, ...]) for downstream Session steps.
    """
    from buildml.synthetic.types import ColumnSchemaSpec

    enriched: list[ColumnSchemaSpec] = []
    for spec in specs:
        extras = dict(spec.extras)
        if spec.kind in {"continuous", "integer"} and spec.name in train.columns:
            numeric = pd.to_numeric(train[spec.name], errors="coerce").dropna()
            if len(numeric):
                extras["train_min"] = float(np.min(numeric))
                extras["train_max"] = float(np.max(numeric))
        enriched.append(
            ColumnSchemaSpec(
                name=spec.name,
                kind=spec.kind,
                n_unique=spec.n_unique,
                n_null=spec.n_null,
                categories=spec.categories,
                extras=extras,
            )
        )
    return tuple(enriched)
