"""Tier C: sklearn Pipeline twin for mortgage-default-classical."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

from buildml import Session
from proofs._lib import (
    extract_buildml_test_metrics,
    load_buildml_results,
    load_classical_credit_table,
    new_proof_context,
    sklearn_logreg_twin,
    supervised_roles,
    write_comparison,
)


def main() -> None:
    ctx = new_proof_context("mortgage-default-classical", seed=101)
    frame, data_meta = load_classical_credit_table(seed=ctx.seed)
    session = (
        Session.ingest(frame.copy())
        .set_roles(supervised_roles(data_meta))
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=ctx.seed)
    )
    plan = session.split_plan
    assert plan is not None
    industry = sklearn_logreg_twin(
        frame,
        plan,
        feature_columns=list(data_meta["feature_columns"]),
        target=str(data_meta["target"]),
        seed=ctx.seed,
    )

    bml_raw = load_buildml_results(ctx.project_dir)
    bml_metrics = extract_buildml_test_metrics(
        bml_raw,
        prefer=("test_metrics",),
        keys=("accuracy", "f1", "roc_auc", "f1_weighted"),
    )
    if "f1" not in bml_metrics and "f1_weighted" in bml_metrics:
        bml_metrics["f1"] = bml_metrics["f1_weighted"]

    write_comparison(
        ctx,
        buildml={
            "backend": "buildml.Session",
            "estimator": "LogisticRegression",
            "test_metrics": bml_metrics,
        },
        industry=industry,
        split_counts={
            "train": len(plan.train_indices),
            "validation": len(plan.validation_indices),
            "test": len(plan.test_indices),
        },
        delta_keys=("accuracy", "f1", "roc_auc"),
        extra={"loader_selected": data_meta.get("loader_selected")},
    )
    print(
        "mortgage-default-classical Tier C OK",
        data_meta.get("loader_selected"),
        industry["test_metrics"],
    )


if __name__ == "__main__":
    main()
