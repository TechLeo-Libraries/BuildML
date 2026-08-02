"""Shared helpers for Tier C industry comparison twins."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from proofs._lib.harness import ProofContext, metrics_round, write_results


DISCLOSURE = (
    "Deltas are descriptive on one synthetic draw; not a claim of universal "
    "superiority. Workflow parity and leakage discipline matter more than tiny "
    "metric gaps. Success bar is competitive qualitative parity (5-B)."
)


def load_buildml_results(project_dir: Path) -> dict[str, Any]:
    """Load ``results/results.json`` written by the Tier A script."""
    path = project_dir / "results" / "results.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing BuildML results at {path}; run script.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def extract_buildml_test_metrics(
    results: Mapping[str, Any],
    *,
    keys: Sequence[str] = (),
    prefer: Sequence[str] = (
        "test_metrics",
        "test_labeled_metrics",
        "retrieval_metrics",
    ),
) -> dict[str, Any]:
    """Pull a flat metric dict from a Tier A results envelope."""
    blob: dict[str, Any] = {}
    for name in prefer:
        raw = results.get(name)
        if isinstance(raw, Mapping) and raw:
            blob = dict(raw)
            break
    if keys:
        return metrics_round({k: blob[k] for k in keys if k in blob})
    # Drop nested non-scalars.
    out: dict[str, Any] = {}
    for k, v in blob.items():
        if isinstance(v, (int, float, bool, str)) or v is None:
            out[k] = v
    return metrics_round(out)


def compute_deltas(
    buildml: Mapping[str, Any],
    industry: Mapping[str, Any],
    *,
    keys: Sequence[str] | None = None,
) -> dict[str, float]:
    """``buildml - industry`` for overlapping numeric keys."""
    use_keys = list(keys) if keys is not None else [
        k
        for k in buildml
        if k in industry and isinstance(buildml[k], (int, float)) and isinstance(industry[k], (int, float))
    ]
    deltas: dict[str, float] = {}
    for key in use_keys:
        if key in buildml and key in industry:
            try:
                deltas[key] = round(float(buildml[key]) - float(industry[key]), 6)
            except (TypeError, ValueError):
                continue
    return deltas


def write_comparison(
    ctx: ProofContext,
    *,
    buildml: Mapping[str, Any],
    industry: Mapping[str, Any],
    same_split: bool = True,
    split_counts: Mapping[str, int] | None = None,
    delta_keys: Sequence[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``results/comparison.json`` with the Tier C envelope."""
    bml_metrics = dict(buildml.get("test_metrics", buildml))
    ind_metrics = dict(industry.get("test_metrics", {}))
    payload: dict[str, Any] = {
        "same_split": same_split,
        "split_counts": dict(split_counts or {}),
        "buildml": dict(buildml),
        "industry": dict(industry),
        "deltas": compute_deltas(bml_metrics, ind_metrics, keys=delta_keys),
        "disclosure": DISCLOSURE,
        "status": "filled",
    }
    if extra:
        payload.update(dict(extra))
    return write_results(ctx, payload, filename="comparison.json")
