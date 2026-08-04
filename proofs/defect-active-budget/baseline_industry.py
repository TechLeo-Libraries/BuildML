"""Tier C: margin-sampling active learning twin for defect-active-budget."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from buildml import Session
from proofs._lib import (
    extract_buildml_test_metrics,
    load_buildml_results,
    metrics_round,
    new_proof_context,
    write_comparison,
)


FEATS = ["surface_roughness", "pixel_anomaly", "acoustic_z"]


def main() -> None:
    ctx = new_proof_context("defect-active-budget", seed=23)
    rng = np.random.default_rng(ctx.seed)
    ok = rng.normal([-1.1, -0.9, 0.2], 0.5, size=(170, 3))
    defect = rng.normal([1.0, 1.1, -0.3], 0.5, size=(170, 3))
    frame = pd.DataFrame(np.vstack([ok, defect]), columns=FEATS)
    frame["is_defect"] = [0] * 170 + [1] * 170
    truth = frame["is_defect"].to_numpy()

    session = (
        Session.ingest(frame.copy())
        .set_roles({**{c: "feature" for c in FEATS}, "is_defect": "target"})
        .split(test_size=0.25, validation_size=0.15, stratify=True, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    train_idx = np.asarray(list(plan.train_indices), dtype=int)
    test_idx = np.asarray(list(plan.test_indices), dtype=int)
    val_idx = list(plan.validation_indices)

    scaler = StandardScaler()
    x_train_full = scaler.fit_transform(frame.loc[train_idx, FEATS])
    x_test = scaler.transform(frame.loc[test_idx, FEATS])

    labeled_mask = np.zeros(len(train_idx), dtype=bool)
    n_seed = max(2, int(0.15 * len(train_idx)))
    seed_pick = rng.choice(len(train_idx), size=n_seed, replace=False)
    labeled_mask[seed_pick] = True

    batch_size = 8
    budget = 32
    used = 0
    curve = []
    while used < budget and (~labeled_mask).any():
        clf = LogisticRegression(max_iter=1000, random_state=ctx.seed)
        clf.fit(x_train_full[labeled_mask], truth[train_idx][labeled_mask])
        unlabeled = np.where(~labeled_mask)[0]
        if len(unlabeled) == 0:
            break
        proba = clf.predict_proba(x_train_full[unlabeled])
        margin = np.abs(proba[:, 1] - proba[:, 0])
        order = np.argsort(margin)
        take = order[: min(batch_size, budget - used, len(order))]
        labeled_mask[unlabeled[take]] = True
        used += len(take)
        curve.append({"n_labeled": int(labeled_mask.sum()), "budget_used": used})

    clf = LogisticRegression(max_iter=1000, random_state=ctx.seed)
    clf.fit(x_train_full[labeled_mask], truth[train_idx][labeled_mask])
    pred = clf.predict(x_test)
    y_test = truth[test_idx]
    industry_metrics = metrics_round(
        {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "n_labeled_final": int(labeled_mask.sum()),
            "budget_used": int(used),
        }
    )

    bml_raw = load_buildml_results(ctx.project_dir)
    bml_metrics = extract_buildml_test_metrics(
        bml_raw, prefer=("test_metrics",), keys=("accuracy", "f1", "f1_weighted", "macro_f1")
    )
    if "f1" not in bml_metrics:
        for alt in ("f1_weighted", "macro_f1"):
            if alt in bml_metrics:
                bml_metrics["f1"] = bml_metrics[alt]
                break

    write_comparison(
        ctx,
        buildml={
            "backend": "buildml.session.active_learning.fit",
            "strategy": bml_raw.get("fit", {}).get("strategy", "margin"),
            "test_metrics": bml_metrics,
        },
        industry={
            "backend": "sklearn LogisticRegression + margin sampling",
            "test_metrics": industry_metrics,
            "label_curve": curve,
            "leakage_controls": [
                "Queries drawn from train unlabeled pool only",
                "Oracle labels simulated from held-out truth for train indices",
                "Scaler fit on train only",
                "Test evaluated after budget loop",
            ],
        },
        split_counts={
            "train": int(len(train_idx)),
            "validation": len(val_idx),
            "test": int(len(test_idx)),
        },
        delta_keys=("accuracy", "f1"),
    )
    print("defect-active-budget Tier C OK", industry_metrics)


if __name__ == "__main__":
    main()
