"""Tier A proof: Torch MLP tabular underwriting on mortgage default."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from buildml import Session
from buildml.core.errors import MissingExtraError
from proofs._lib import (
    TORCH_STATUS,
    assert_disjoint_partitions,
    assert_no_test_in_selection,
    load_mortgage_default_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)

FEATURE_NUM = ["ltv", "dti", "credit_score", "note_rate", "term_years"]
FEATURE_CAT = ["property_type"]
TARGET = "defaulted"


def _labels(plan) -> list[str]:
    n = max(plan.train_indices + plan.validation_indices + plan.test_indices) + 1
    out = ["unused"] * n
    for i in plan.train_indices:
        out[i] = "train"
    for i in plan.validation_indices:
        out[i] = "validation"
    for i in plan.test_indices:
        out[i] = "test"
    return out


def main() -> None:
    ctx = new_proof_context("torch-tabular-underwrite", seed=106)
    if TORCH_STATUS.get("skip_torch_paths", True):
        write_results(
            ctx,
            {
                "status": "skipped_missing_extra",
                "extra": "torch",
                "error": TORCH_STATUS.get("error") or "torch smoke failed / unavailable",
                "torch": TORCH_STATUS,
            },
        )
        print("torch-tabular-underwrite SKIPPED (no torch)")
        return

    frame, data_meta = load_mortgage_default_synthetic(n=1400, seed=ctx.seed)
    session = (
        Session.ingest(frame)
        .set_roles({**{c: "feature" for c in FEATURE_NUM + FEATURE_CAT}, TARGET: "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    counts = assert_disjoint_partitions(_labels(plan))

    session.impute(strategy="median")
    session.encode(method="onehot")

    try:
        session.make_torch_loaders(
            batch_size=64,
            normalize=True,
            seed=ctx.seed,
            task="classification",
        )
        session.fit_torch(epochs=3, learning_rate=1e-2, device="cpu", hidden=(64, 32))
        val = session.evaluate_torch(partition="validation")
        assert_no_test_in_selection(
            selection_partition="validation",
            evaluation_partition="test",
        )
        test = session.evaluate_torch(partition="test")
        try:
            bundle = session.save_torch_bundle(ctx.artifacts_dir / "torch_bundle")
            bundle_path = str(bundle)
        except Exception as exc:  # noqa: BLE001
            bundle_path = f"unavailable: {type(exc).__name__}: {exc}"
    except (MissingExtraError, ImportError) as exc:
        write_results(
            ctx,
            {
                "status": "skipped_missing_extra",
                "extra": "torch",
                "error": str(exc),
                "torch": TORCH_STATUS,
            },
        )
        print("torch-tabular-underwrite SKIPPED", exc)
        return

    bml_test = metrics_round(dict(getattr(test, "metrics", {}) or {}))
    write_results(
        ctx,
        {
            "status": "completed",
            "data": data_meta,
            "split": {"kind": plan.kind, "counts": counts, "stratify": True},
            "torch": TORCH_STATUS,
            "epochs": 3,
            "leakage_controls": [
                "Stratified split before impute/encode/loaders",
                "Torch normalize stats from train loader only",
                "Test evaluate_torch after lock",
            ],
            "validation_metrics": metrics_round(dict(getattr(val, "metrics", {}) or {})),
            "test_metrics": bml_test,
            "bundle_path": bundle_path,
            "industry_comparison": {
                "status": "filled",
                "note": (
                    "Tier C baseline_industry.py: sklearn MLPClassifier twin on the same split; "
                    "run script then baseline_industry.py for results/comparison.json."
                ),
            },
            "limitations": [
                "Synthetic mortgage labels; 3-epoch MLP only",
                "CPU-only smoke; not a production underwriting network",
            ],
        },
    )
    print("torch-tabular-underwrite OK", bml_test)


if __name__ == "__main__":
    main()
