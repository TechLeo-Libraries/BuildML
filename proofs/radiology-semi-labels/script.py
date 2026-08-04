"""Tier A proof: radiology-semi-labels — scarce labels on imaging proxy features."""

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
    full.loc[blank, "lesion_present"] = np.nan
    session._dataset = Dataset.from_transformed(
        session.dataset, full,
        schema=schema_from_dataframe(full),
        roles=dict(session.dataset.roles),
    )
    return session


def main() -> None:
    ctx = new_proof_context("radiology-semi-labels", seed=22)
    rng = np.random.default_rng(ctx.seed)
    # Tabular proxy for radiology patch features (not pixels): intensity / texture / edges.
    neg = rng.normal([-0.9, -0.8, -0.5, 0.2], [0.55, 0.5, 0.45, 0.4], size=(190, 4))
    pos = rng.normal([1.1, 0.9, 0.7, -0.3], [0.55, 0.5, 0.45, 0.4], size=(190, 4))
    frame = pd.DataFrame(
        np.vstack([neg, pos]),
        columns=["hu_mean", "texture_entropy", "edge_density", "asymmetry"],
    )
    frame["lesion_present"] = [0] * 190 + [1] * 190
    session = (
        Session.ingest(frame)
        .set_roles({
            "hu_mean": "feature",
            "texture_entropy": "feature",
            "edge_density": "feature",
            "asymmetry": "feature",
            "lesion_present": "target",
        })
        .split(test_size=0.25, validation_size=0.15, stratify=True, random_state=ctx.seed)
        .scale(method="standard")
    )
    session = _mask(session, fraction=0.78, seed=ctx.seed)
    fit = session.semisupervised.fit(method="label_propagation", n_neighbors=7)
    val = session.semisupervised.evaluate(partition="validation")
    test = session.semisupervised.evaluate(partition="test")
    bundle = session.semisupervised.save_bundle(ctx.artifacts_dir / "semi_bundle")
    write_results(ctx, {
        "status": "completed",
        "data": {
            "name": "synthetic_radiology_proxy_features",
            "license": "synthetic/public-domain",
            "n_rows": int(len(frame)),
        },
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
            "Test never used for label propagation graph fitting beyond eval",
        ],
        "industry_comparison": {
            "status": "filled",
            "note": (
                "Tier C baseline_industry.py: sklearn LabelPropagation twin on the same "
                "split; run script then baseline_industry.py for results/comparison.json."
            ),
        },
        "limitations": [
            "Tabular imaging proxy — not DICOM / CNN training",
            "Masking is proof harness, not a PACS labeling workflow",
        ],
    })
    print("radiology-semi-labels OK", dict(test.metrics))


if __name__ == "__main__":
    main()
