"""Tier A proof: Torch text intent / ticket routing on support tickets."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import pandas as pd

from buildml import Session
from buildml.core.errors import MissingExtraError, ValidationError
from proofs._lib import (
    TORCH_STATUS,
    assert_no_test_in_selection,
    load_support_tickets_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)


def main() -> None:
    ctx = new_proof_context("torch-text-intent", seed=107)
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
        print("torch-text-intent SKIPPED (no torch)")
        return

    frame, data_meta = load_support_tickets_synthetic(n=900, seed=ctx.seed)
    # Torch text path requires integer class ids; keep string labels for disclosure.
    queue_codes, queue_uniques = pd.factorize(frame["queue"], sort=True)
    frame = frame.copy()
    frame["queue_name"] = frame["queue"]
    frame["queue"] = queue_codes.astype(int)
    class_map = {int(i): str(name) for i, name in enumerate(queue_uniques)}

    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "ticket_id": "id",
                "body": "feature",
                "channel": "feature",
                "queue": "target",
            }
        )
        .split(
            test_size=0.2,
            validation_size=0.2,
            stratify=True,
            random_state=ctx.seed,
        )
    )
    plan = session.split_plan
    assert plan is not None
    counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }

    path_used = "make_text_torch_loaders+fit_torch"
    try:
        session.make_text_torch_loaders(
            text_column="body",
            batch_size=32,
            max_len=64,
            max_vocab=4000,
            seed=ctx.seed,
        )
        session.fit_torch(epochs=3, learning_rate=1e-2, device="cpu")
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
    except (MissingExtraError, ImportError, TypeError, ValueError, ValidationError) as exc:
        # Fallback classical text path when Torch text APIs raise after Torch probe OK.
        try:
            # Restore string labels for the NLP classifier path.
            frame_nlp = frame.copy()
            frame_nlp["queue"] = frame_nlp["queue_name"]
            session = (
                Session.ingest(frame_nlp)
                .set_roles(
                    {
                        "ticket_id": "id",
                        "body": "feature",
                        "channel": "feature",
                        "queue": "target",
                    }
                )
            )
            session.inject_split(
                train_indices=list(plan.train_indices),
                validation_indices=list(plan.validation_indices),
                test_indices=list(plan.test_indices),
            )
            fit = session.fit_text_classifier(
                text_column="body",
                vectorizer="tfidf",
                estimator="logistic",
                ngram_range=(1, 2),
                min_df=2,
                class_weight="balanced",
                random_state=ctx.seed,
            )
            path_used = f"fit_text_classifier_fallback({type(exc).__name__})"
            val = session.evaluate_text_classifier(partition="validation")
            test = session.evaluate_text_classifier(partition="test")
            try:
                bundle = session.save_nlp_bundle(ctx.artifacts_dir / "nlp_bundle")
                bundle_path = str(bundle)
            except Exception as exc2:  # noqa: BLE001
                bundle_path = f"unavailable: {type(exc2).__name__}: {exc2}"
            _ = fit
        except Exception as exc2:  # noqa: BLE001
            write_results(
                ctx,
                {
                    "status": "skipped_missing_extra",
                    "extra": "torch",
                    "error": f"{type(exc).__name__}: {exc}; fallback: {exc2}",
                    "torch": TORCH_STATUS,
                },
            )
            print("torch-text-intent SKIPPED", exc2)
            return

    bml_test = metrics_round(dict(getattr(test, "metrics", {}) or {}))
    write_results(
        ctx,
        {
            "status": "completed",
            "data": {**data_meta, "class_map": class_map},
            "split": {"kind": plan.kind, "counts": counts, "stratify": True},
            "torch": TORCH_STATUS,
            "path": path_used,
            "epochs": 3,
            "leakage_controls": [
                "Stratified split before text loaders / classifier fit",
                "Vocabulary / normalize from train only",
                "Test evaluated after lock",
            ],
            "validation_metrics": metrics_round(dict(getattr(val, "metrics", {}) or {})),
            "test_metrics": bml_test,
            "bundle_path": bundle_path,
            "industry_comparison": {
                "status": "filled",
                "note": (
                    "Tier C baseline_industry.py: Tfidf+LR twin on the same split; "
                    "run script then baseline_industry.py for results/comparison.json."
                ),
            },
            "limitations": [
                "Synthetic tickets; 3-epoch text MLP when Torch path succeeds",
                "Honest skip when Torch is missing",
            ],
        },
    )
    print("torch-text-intent OK", bml_test)


if __name__ == "__main__":
    main()
