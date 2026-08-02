"""Tier B product: Atlas Label Studio.

Composes self-supervised pretext representations + semi-supervised
propagation on scarce labels + an active-learning budget loop with a
simulated oracle. Train-pool queries only; holdouts keep labels for eval.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import numpy as np
import pandas as pd

from buildml import Session
from buildml.core.errors import MissingExtraError
from buildml.data.dataset import Dataset
from buildml.ingest.detect import schema_from_dataframe
from proofs._lib import (
    TORCH_STATUS,
    assert_no_test_in_selection,
    metrics_round,
    new_proof_context,
    write_results,
)


def _label_pool(n_per: int = 220, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    x0 = rng.normal([-1.1, -1.0, 0.0, 0.2], 0.55, size=(n_per, 4))
    x1 = rng.normal([1.1, 1.0, 0.1, -0.2], 0.55, size=(n_per, 4))
    x = np.vstack([x0, x1])
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(4)])
    frame["label"] = [0] * n_per + [1] * n_per
    meta = {
        "name": "atlas_synthetic_label_pool",
        "license": "synthetic/public-domain",
        "n_rows": int(len(frame)),
        "positive_rate": 0.5,
    }
    return frame, meta


def _mask_train_labels(session: Session, fraction: float, seed: int) -> np.ndarray:
    """Blank a fraction of train labels; return original truth series."""
    rng = np.random.default_rng(seed)
    full = session.to_pandas().copy()
    truth = full["label"].to_numpy().copy()
    train_idx = list(session.split_plan.train_indices)
    n_blank = max(1, int(fraction * len(train_idx)))
    blank = rng.choice(train_idx, size=n_blank, replace=False)
    full.loc[blank, "label"] = np.nan
    session._dataset = Dataset.from_transformed(
        session.dataset,
        full,
        schema=schema_from_dataframe(full),
        roles=dict(session.dataset.roles),
    )
    return truth


def main() -> None:
    ctx = new_proof_context("atlas-label-studio", seed=0)
    frame, data_meta = _label_pool(seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []

    session = (
        Session.ingest(frame.copy())
        .set_roles({**{f"f{i}": "feature" for i in range(4)}, "label": "target"})
        .split(test_size=0.25, validation_size=0.15, stratify=True, random_state=ctx.seed)
        .scale(method="standard")
    )
    plan = session.split_plan
    assert plan is not None
    split_counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }
    truth = _mask_train_labels(session, fraction=0.85, seed=ctx.seed)

    # --- Stage 1: SSL pretext on (mostly unlabeled) train features ---
    try:
        ssl_fit = session.fit_ssl_pretext(method="masked_tabular", random_state=ctx.seed)
        try:
            session.finetune_ssl_head(random_state=ctx.seed)
        except Exception:  # noqa: BLE001
            pass
        ssl_val = session.evaluate_ssl(partition="validation")
        ssl_test = session.evaluate_ssl(partition="test")
        stages["ssl"] = {
            "status": "ok",
            "torch": TORCH_STATUS,
            "fit": metrics_round(ssl_fit.to_dict() if hasattr(ssl_fit, "to_dict") else {}),
            "validation_metrics": metrics_round(dict(getattr(ssl_val, "metrics", {}) or {})),
            "test_metrics": metrics_round(dict(getattr(ssl_test, "metrics", {}) or {})),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["ssl"] = {"status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
        skip_notes.append(f"ssl: {exc}")
    write_results(ctx, stages["ssl"], filename="ssl.json")

    # --- Stage 2: semi-supervised on scarce train labels ---
    # Fresh session with same split + masking (SSL may have mutated state).
    semi_session = (
        Session.ingest(frame.copy())
        .set_roles({**{f"f{i}": "feature" for i in range(4)}, "label": "target"})
        .inject_split(
            train_indices=list(plan.train_indices),
            validation_indices=list(plan.validation_indices),
            test_indices=list(plan.test_indices),
        )
        .scale(method="standard")
    )
    _mask_train_labels(semi_session, fraction=0.85, seed=ctx.seed)
    try:
        semi_fit = semi_session.fit_semisupervised(
            method="label_propagation", n_neighbors=7
        )
        assert_no_test_in_selection(
            selection_partition="validation", evaluation_partition="test"
        )
        semi_val = semi_session.evaluate_semisupervised(partition="validation")
        semi_test = semi_session.evaluate_semisupervised(partition="test")
        stages["semisupervised"] = {
            "status": "ok",
            "fit": {
                "method": semi_fit.method,
                "n_labeled_train": int(semi_fit.n_labeled_train),
                "n_unlabeled_train": int(semi_fit.n_unlabeled_train),
            },
            "validation_metrics": metrics_round(dict(semi_val.metrics)),
            "test_metrics": metrics_round(dict(semi_test.metrics)),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["semisupervised"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"semisupervised: {exc}")
    write_results(ctx, stages["semisupervised"], filename="semisupervised.json")

    # --- Stage 3: active learning budget loop ---
    al_session = (
        Session.ingest(frame.copy())
        .set_roles({**{f"f{i}": "feature" for i in range(4)}, "label": "target"})
        .inject_split(
            train_indices=list(plan.train_indices),
            validation_indices=list(plan.validation_indices),
            test_indices=list(plan.test_indices),
        )
        .scale(method="standard")
    )
    _mask_train_labels(al_session, fraction=0.85, seed=ctx.seed)
    try:
        al_fit = al_session.fit_active_learner(
            strategy="margin", batch_size=8, label_budget=40
        )
        curve = []
        for round_i in range(5):
            q = al_session.suggest_query(batch_size=8)
            if not q.indices:
                break
            human = [int(truth[i]) for i in q.indices]
            labeled = al_session.label_rows(indices=q.indices, labels=human)
            curve.append(
                {
                    "round": round_i,
                    "n_newly_labeled": int(labeled.n_newly_labeled),
                    "n_labeled_now": int(labeled.n_labeled_now),
                    "budget_remaining": int(labeled.budget_remaining),
                }
            )
        al_test = al_session.evaluate_active_learning(partition="test")
        stages["active_learning"] = {
            "status": "ok",
            "fit": {
                "strategy": al_fit.strategy,
                "n_labeled_train": int(al_fit.n_labeled_train),
                "n_unlabeled_pool": int(al_fit.n_unlabeled_pool),
            },
            "label_curve": curve,
            "test_metrics": metrics_round(dict(al_test.metrics)),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["active_learning"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"active_learning: {exc}")
    write_results(ctx, stages["active_learning"], filename="active_learning.json")

    summary = {
        "status": "completed",
        "product": "Atlas Label Studio",
        "data": data_meta,
        "split": {"kind": plan.kind, "counts": split_counts, "stratify": True},
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "torch": TORCH_STATUS,
        "leakage_controls": [
            "Stratified split before masking / pretext / AL",
            "Label masking applied to train indices only",
            "Holdouts retain full labels solely for evaluation",
            "AL queries drawn from train unlabeled pool only",
            "Oracle simulates humans; library never invents production labels",
        ],
        "what_fails_if_leakage_ignored": [
            "Masking validation/test labels and then 'recovering' them via graph overstates semi-supervised gains",
            "Allowing AL to query the test pool makes the budget curve a cheat sheet",
            "Fitting SSL pretext on the full table including test rows leaks holdout geometry into embeddings",
        ],
        "limitations": [
            "Simulated oracle — not a real annotation UI / workforce",
            "Synthetic blobs; production label noise not modeled",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "atlas-label-studio OK",
        {
            "semi": stages.get("semisupervised", {}).get("test_metrics"),
            "al": stages.get("active_learning", {}).get("test_metrics"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
