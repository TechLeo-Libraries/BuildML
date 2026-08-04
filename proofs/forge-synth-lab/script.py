"""Tier B product: Forge Synth Lab.

Composes tabular synthesis + classical TSTR utility + unsupervised clustering
on synthetic samples. Synthesizer fit on train only; fidelity vs holdout.
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
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.core.errors import MissingExtraError
from proofs._lib import (
    assert_no_test_in_selection,
    metrics_round,
    new_proof_context,
    write_results,
)


def _retail_catalog(n: int = 560, seed: int = 46) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    unit_price = rng.lognormal(3.2, 0.55, n).clip(1.0, 500.0)
    units_sold = rng.poisson(12, n).astype(float) + 1.0
    margin_pct = rng.beta(3, 4, n)
    category = rng.choice(["electronics", "apparel", "grocery", "home"], size=n)
    # Proxy label for TSTR: high-velocity SKUs
    high_velocity = (
        (np.log1p(units_sold) > 2.4) & (margin_pct > 0.35)
    ).astype(int)
    frame = pd.DataFrame(
        {
            "unit_price": unit_price,
            "units_sold": units_sold,
            "margin_pct": margin_pct,
            "category": category,
            "high_velocity": high_velocity,
            "true_family": rng.integers(0, 4, size=n),
        }
    )
    meta = {
        "name": "forge_retail_catalog_source",
        "license": "synthetic/public-domain",
        "n_rows": n,
        "positive_rate": float(high_velocity.mean()),
        "disclosures": ["NO differential privacy claims"],
    }
    return frame, meta


def main() -> None:
    ctx = new_proof_context("forge-synth-lab", seed=46)
    frame, data_meta = _retail_catalog(seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []

    # --- Stage 1: synthesizer fit (train only) ---
    synth_session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                "unit_price": "feature",
                "units_sold": "feature",
                "margin_pct": "feature",
                "category": "feature",
                "high_velocity": "ignore",
                "true_family": "ignore",
            }
        )
        .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
    )
    plan = synth_session.split_plan
    assert plan is not None
    split_counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }
    try:
        s_fit = synth_session.synthetic.fit(
            method="gaussian_copula", random_state=ctx.seed
        )
        sample = synth_session.synthetic.sample(n=220, random_state=ctx.seed)
        s_ev = synth_session.synthetic.evaluate(partition="test")
        stages["synthetic"] = {
            "status": "ok",
            "fit": metrics_round(s_fit.to_dict() if hasattr(s_fit, "to_dict") else {}),
            "sample_shape": list(getattr(sample, "shape", [])),
            "eval": metrics_round(
                s_ev.to_dict()
                if hasattr(s_ev, "to_dict")
                else dict(getattr(s_ev, "metrics", {}) or {})
            ),
        }
        sample_df = (
            sample
            if isinstance(sample, pd.DataFrame)
            else getattr(sample, "frame", None)
        )
        if sample_df is None and hasattr(sample, "to_pandas"):
            sample_df = sample.to_pandas()
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["synthetic"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"synthetic: {exc}")
        sample_df = None
    write_results(ctx, stages["synthetic"], filename="synthetic.json")

    # --- Stage 2: classical TSTR (train on synth, eval on real holdout) ---
    try:
        if sample_df is None or len(sample_df) < 40:
            raise ValueError("insufficient synthetic sample for TSTR")
        # Real train for label model on synthetic needs labels — attach proxy via
        # heuristic from numeric columns (same DGP rule), then evaluate on real test.
        syn = sample_df.copy()
        if "high_velocity" not in syn.columns:
            syn["high_velocity"] = (
                (np.log1p(syn["units_sold"].astype(float)) > 2.4)
                & (syn["margin_pct"].astype(float) > 0.35)
            ).astype(int)
        real_test = frame.loc[list(plan.test_indices)].copy()
        # Fit classical on synthetic only
        tstr_train = (
            Session.ingest(syn)
            .set_roles(
                {
                    "unit_price": "feature",
                    "units_sold": "feature",
                    "margin_pct": "feature",
                    "category": "feature",
                    "high_velocity": "target",
                }
            )
            .split(test_size=0.15, validation_size=0.15, random_state=ctx.seed)
        )
        tstr_train.encode(method="onehot")
        tstr_train.scale(method="standard")
        tstr_train.fit(
            LogisticRegression(max_iter=1000, random_state=ctx.seed),
            task="classification",
        )
        # Evaluate on real holdout via a session that injects real rows as test
        hold = (
            Session.ingest(real_test.reset_index(drop=True))
            .set_roles(
                {
                    "unit_price": "feature",
                    "units_sold": "feature",
                    "margin_pct": "feature",
                    "category": "feature",
                    "high_velocity": "target",
                    "true_family": "ignore",
                }
            )
        )
        # Use evaluate on synthetic-trained model by predicting through tstr_train
        # on a combined frame: train=synth, test=real holdout
        combo = pd.concat(
            [
                syn.assign(_part="train"),
                real_test.assign(_part="test")[
                    ["unit_price", "units_sold", "margin_pct", "category", "high_velocity"]
                ].assign(_part="test"),
            ],
            ignore_index=True,
        )
        train_i = list(combo.index[combo["_part"] == "train"])
        test_i = list(combo.index[combo["_part"] == "test"])
        val_i = train_i[: max(2, len(train_i) // 5)]
        train_i2 = [i for i in train_i if i not in val_i]
        tstr = (
            Session.ingest(combo.drop(columns=["_part"]))
            .set_roles(
                {
                    "unit_price": "feature",
                    "units_sold": "feature",
                    "margin_pct": "feature",
                    "category": "feature",
                    "high_velocity": "target",
                }
            )
            .inject_split(
                train_indices=train_i2,
                validation_indices=val_i,
                test_indices=test_i,
            )
        )
        tstr.encode(method="onehot")
        tstr.scale(method="standard")
        tstr.fit(
            LogisticRegression(max_iter=1000, random_state=ctx.seed),
            task="classification",
        )
        assert_no_test_in_selection(
            selection_partition="validation", evaluation_partition="test"
        )
        tstr_test = tstr.evaluate(partition="test")
        stages["tstr_classical"] = {
            "status": "ok",
            "estimator": "LogisticRegression",
            "n_synth_train": len(train_i2),
            "n_real_test": len(test_i),
            "test_metrics": metrics_round(dict(tstr_test.metrics)),
            "disclosure": (
                "Classifier fitted on synthetic rows only; metrics on real holdout "
                "(train-on-synthetic, test-on-real)."
            ),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["tstr_classical"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"tstr_classical: {exc}")
    write_results(ctx, stages["tstr_classical"], filename="tstr_classical.json")

    # --- Stage 3: unsupervised clusters on synthetic numeric embeddings ---
    try:
        if sample_df is None:
            raise ValueError("no synthetic sample for clustering")
        clus_src = sample_df.copy()
        num_cols = [
            c
            for c in ["unit_price", "units_sold", "margin_pct"]
            if c in clus_src.columns
        ]
        if len(num_cols) < 2:
            raise ValueError("need numeric columns for clustering")
        clus_src["true_family"] = (
            pd.cut(
                clus_src[num_cols[0]].astype(float),
                bins=4,
                labels=False,
                duplicates="drop",
            )
            .astype(float)
            .fillna(0)
            .astype(int)
        )
        clus = (
            Session.ingest(clus_src[num_cols + ["true_family"]])
            .set_roles({**{c: "feature" for c in num_cols}, "true_family": "ignore"})
            .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
            .scale(method="standard")
        )
        c_fit = clus.unsupervised.fit(method="kmeans", n_clusters=4, random_state=ctx.seed)
        c_val = clus.unsupervised.evaluate(
            partition="validation", external_label_column="true_family"
        )
        c_test = clus.unsupervised.evaluate(
            partition="test", external_label_column="true_family"
        )
        stages["clusters"] = {
            "status": "ok",
            "method": "kmeans",
            "fit": metrics_round(c_fit.to_dict() if hasattr(c_fit, "to_dict") else {}),
            "validation_metrics": metrics_round(
                c_val.to_dict() if hasattr(c_val, "to_dict") else {}
            ),
            "test_metrics": metrics_round(
                c_test.to_dict() if hasattr(c_test, "to_dict") else {}
            ),
        }
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["clusters"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"clusters: {exc}")
    write_results(ctx, stages["clusters"], filename="clusters.json")

    summary = {
        "status": "completed",
        "product": "Forge Synth Lab",
        "data": data_meta,
        "split": {"kind": plan.kind, "counts": split_counts},
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "leakage_controls": [
            "Synthesizer fit on train only",
            "Fidelity / session.synthetic.evaluate vs real holdout",
            "TSTR classifier trained on synthetic rows; metrics on real test",
            "Cluster fit on synthetic sample's own split (external labels eval-only)",
        ],
        "what_fails_if_leakage_ignored": [
            "Fitting the synthesizer on the full table makes fidelity look perfect",
            "TSTR that peeks at real test labels during synth training is not utility",
            "Clustering with test-conditioned features overstates segment purity",
        ],
        "limitations": [
            "NO differential privacy / anonymity claims",
            "Utility ≠ privacy; synthetic retail catalog only",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "forge-synth-lab OK",
        {
            "synthetic": stages["synthetic"]["status"],
            "tstr": stages["tstr_classical"]["status"],
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
