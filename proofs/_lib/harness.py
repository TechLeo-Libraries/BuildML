"""Result writers, seeds, and leakage assertion helpers for proof projects."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from proofs._lib.env import TORCH_STATUS


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ProofContext:
    """Filesystem layout and metadata for one proof project run."""

    slug: str
    project_dir: Path
    results_dir: Path
    artifacts_dir: Path
    seed: int = 42
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def result_path(self, name: str = "results.json") -> Path:
        return self.results_dir / name


def new_proof_context(slug: str, *, seed: int = 42) -> ProofContext:
    """Create ``proofs/<slug>/{results,artifacts}`` and return a context."""
    project_dir = REPO_ROOT / "proofs" / slug
    results_dir = project_dir / "results"
    artifacts_dir = project_dir / "artifacts"
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)
    return ProofContext(
        slug=slug,
        project_dir=project_dir,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        seed=seed,
    )


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy (and Torch when healthy) for reproducible proofs."""
    random.seed(seed)
    np.random.seed(seed)
    if not TORCH_STATUS.get("skip_torch_paths"):
        try:
            import torch

            torch.manual_seed(seed)
        except Exception:  # noqa: BLE001
            pass


def json_safe(value: Any) -> Any:
    """Convert common ML objects into JSON-serializable structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return json_safe(value.to_dict())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "__dict__"):
        try:
            return json_safe(
                {
                    k: v
                    for k, v in vars(value).items()
                    if not k.startswith("_") and not callable(v)
                }
            )
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def metrics_round(metrics: Mapping[str, Any], *, ndigits: int = 6) -> dict[str, Any]:
    """Round float metrics for stable JSON diffs."""
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            out[key] = round(value, ndigits)
        elif isinstance(value, Mapping):
            out[key] = metrics_round(value, ndigits=ndigits)
        else:
            out[key] = json_safe(value)
    return out


def write_results(
    ctx: ProofContext,
    payload: Mapping[str, Any],
    *,
    filename: str = "results.json",
) -> Path:
    """Write a proof result document with standard envelope fields."""
    envelope = {
        "project": ctx.slug,
        "seed": ctx.seed,
        "started_at": ctx.started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "torch": TORCH_STATUS,
        "repo_root": str(REPO_ROOT),
        **dict(payload),
    }
    path = ctx.result_path(filename)
    path.write_text(json.dumps(json_safe(envelope), indent=2, sort_keys=True) + "\n")
    return path


def assert_disjoint_partitions(
    membership: Sequence[str] | np.ndarray | Iterable[str],
    *,
    expected: Sequence[str] = ("train", "validation", "test"),
) -> dict[str, int]:
    """Assert partition labels are known and all expected labels appear."""
    labels = [str(x) for x in membership]
    counts: dict[str, int] = {}
    for label in labels:
        if label not in expected and label != "unused":
            raise AssertionError(f"Unexpected partition label: {label!r}")
        counts[label] = counts.get(label, 0) + 1
    for need in expected:
        if counts.get(need, 0) <= 0:
            raise AssertionError(f"Missing required partition {need!r}; counts={counts}")
    return counts


def assert_no_test_in_selection(
    *,
    selection_partition: str,
    evaluation_partition: str = "test",
) -> None:
    """Document and enforce that selection did not use the final holdout."""
    if selection_partition == evaluation_partition:
        raise AssertionError(
            "Selection/tuning used the evaluation holdout "
            f"({selection_partition!r}); this violates industry leakage policy."
        )
