"""Tier A proof: semi-label-efficiency."""

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
from buildml.data.dataset import Dataset
from buildml.ingest.detect import schema_from_dataframe
from proofs._lib import metrics_round, new_proof_context, write_results


def _mask(session: Session, fraction: float, seed: int) -> Session:
    rng = np.random.default_rng(seed)
    full = session.to_pandas().copy()
    train_idx = list(session.split_plan.train_indices)
    n_blank = max(1, int(fraction * len(train_idx)))
    blank = rng.choice(train_idx, size=n_blank, replace=False)
    full.loc[blank, "label"] = np.nan
    session._dataset = Dataset.from_transformed(
        session.dataset, full,
        schema=schema_from_dataframe(full),
        roles=dict(session.dataset.roles),
    )
    return session


def main() -> None:
    ctx = new_proof_context("semi-label-efficiency", seed=0)
    rng = np.random.default_rng(ctx.seed)
    x0 = rng.normal([-1.0, -1.0], 0.6, size=(180, 2))
    x1 = rng.normal([1.2, 1.0], 0.6, size=(180, 2))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=["x", "y"])
    frame["label"] = [0] * 180 + [1] * 180
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "y": "feature", "label": "target"})
        .split(test_size=0.25, validation_size=0.15, stratify=True, random_state=ctx.seed)
        .scale(method="standard")
    )
    session = _mask(session, fraction=0.75, seed=ctx.seed)
    fit = session.semisupervised.fit(method="label_propagation", n_neighbors=7)
    val = session.semisupervised.evaluate(partition="validation")
    test = session.semisupervised.evaluate(partition="test")
    bundle = session.semisupervised.save_bundle(ctx.artifacts_dir / "semi_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {"name": "synthetic_scarce_labels", "license": "synthetic/public-domain", "n_rows": int(len(frame))},
        "fit": {
            "method": fit.method,
            "n_labeled_train": int(fit.n_labeled_train),
            "n_unlabeled_train": int(fit.n_unlabeled_train),
        },
        "validation_metrics": metrics_round(dict(val.metrics)),
        "test_metrics": metrics_round(dict(test.metrics)),
        "bundle_path": str(bundle),
        "leakage_controls": [
            "Holdouts keep full labels for eval only",
            "Masking applied to train indices only",
            "Test never used for label propagation graph fitting claims beyond eval",
        ],
        "industry_comparison": {
            "status": "filled",
            "note": (
                "Tier C baseline_industry.py: sklearn LabelPropagation twin on the same split; "
                "run script then baseline_industry.py for results/comparison.json."
            ),
        },
        "limitations": ["Synthetic blobs; masking is proof harness, not production labeling"],
    })
    print("semi-label-efficiency OK", dict(test.metrics))


if __name__ == "__main__":
    main()
