"""Tier B product: Relay Edge Federated.

Composes multi-site FedAvg simulation + probabilistic uncertainty intervals +
centralized classical baseline disclosure for edge device risk scores.
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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from buildml import Session
from buildml.core.errors import MissingExtraError
from proofs._lib import (
    TORCH_STATUS,
    metrics_round,
    new_proof_context,
    write_results,
)


def _edge_fleet(n_sites: int = 6, n_per: int = 95, seed: int = 17):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sites):
        shift = s * 0.16
        for _ in range(n_per):
            sensors = rng.normal(size=5) + shift
            logit = (
                -0.25
                + 0.85 * sensors[0]
                + 0.5 * sensors[1]
                - 0.3 * sensors[2]
                + rng.normal(0, 0.3)
            )
            y = int(1 / (1 + np.exp(-logit)) > 0.5)
            rows.append(
                {
                    **{f"s{i}": float(sensors[i]) for i in range(5)},
                    "fault": y,
                    "site": f"edge-{s}",
                }
            )
    frame = pd.DataFrame(rows)
    meta = {
        "name": "relay_synthetic_edge_fleet",
        "license": "synthetic/public-domain",
        "n_rows": int(len(frame)),
        "n_sites": n_sites,
        "positive_rate": float(frame["fault"].mean()),
    }
    return frame, meta


def main() -> None:
    ctx = new_proof_context("relay-edge-federated", seed=17)
    frame, data_meta = _edge_fleet(seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []
    feat_cols = [f"s{i}" for i in range(5)]

    session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                **{c: "feature" for c in feat_cols},
                "fault": "target",
                "site": "group",
            }
        )
        .group_split(
            test_size=0.2,
            validation_size=0.15,
            random_state=ctx.seed,
            group_column="site",
        )
    )
    plan = session.split_plan
    assert plan is not None
    split_counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }
    train_sites = sorted(frame.loc[list(plan.train_indices), "site"].unique().tolist())
    test_sites = sorted(frame.loc[list(plan.test_indices), "site"].unique().tolist())

    # --- Stage 1: federated FedAvg ---
    try:
        fit = session.fit_federated(
            method="fedavg",
            client_column="site",
            n_rounds=6,
            random_state=ctx.seed,
        )
        ev = session.evaluate_federated(partition="test")
        try:
            bundle = session.save_federated_bundle(ctx.artifacts_dir / "fed_bundle")
            bundle_path = str(bundle)
        except Exception as exc:  # noqa: BLE001
            bundle_path = f"unavailable: {exc}"
        stages["federated"] = {
            "status": "ok",
            "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
            "test_metrics": metrics_round(dict(getattr(ev, "metrics", {}) or {})),
            "bundle_path": bundle_path,
            "train_sites": train_sites,
            "test_sites": test_sites,
        }
        write_results(ctx, stages["federated"], filename="federated.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["federated"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"federated: {exc}")

    # --- Stage 2: probabilistic intervals on continuous risk proxy ---
    try:
        risk_frame = frame.copy()
        risk_frame["risk_score"] = (
            0.85 * risk_frame["s0"]
            + 0.5 * risk_frame["s1"]
            - 0.3 * risk_frame["s2"]
            + np.random.default_rng(ctx.seed).normal(0, 0.35, size=len(risk_frame))
        )
        prob_session = (
            Session.ingest(risk_frame)
            .set_roles(
                {
                    **{c: "feature" for c in feat_cols},
                    "risk_score": "target",
                    "site": "group",
                    "fault": "ignore",
                }
            )
            .inject_split(
                train_indices=list(plan.train_indices),
                validation_indices=list(plan.validation_indices),
                test_indices=list(plan.test_indices),
            )
            .scale(method="standard")
        )
        p_fit = prob_session.fit_probabilistic(
            estimator="bayesian_ridge",
            alpha=0.1,
            conformal=True,
            interval_method="both",
            random_state=ctx.seed,
        )
        try:
            intervals = prob_session.predict_interval(partition="test", alpha=0.1)
            interval_payload = metrics_round(
                intervals.to_dict() if hasattr(intervals, "to_dict") else {}
            )
        except Exception as exc:  # noqa: BLE001
            interval_payload = {"error": f"{type(exc).__name__}: {exc}"}
        p_ev = prob_session.evaluate_probabilistic(partition="test")
        stages["probabilistic"] = {
            "status": "ok",
            "fit": metrics_round(p_fit.to_dict() if hasattr(p_fit, "to_dict") else {}),
            "intervals": interval_payload,
            "test_metrics": metrics_round(dict(getattr(p_ev, "metrics", {}) or {})),
        }
        write_results(ctx, stages["probabilistic"], filename="probabilistic.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["probabilistic"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"probabilistic: {exc}")

    # --- Stage 3: classical centralized baseline (disclosure contrast) ---
    try:
        classical_session = (
            Session.ingest(frame.copy())
            .set_roles({**{c: "feature" for c in feat_cols}, "fault": "target", "site": "ignore"})
            .inject_split(
                train_indices=list(plan.train_indices),
                validation_indices=list(plan.validation_indices),
                test_indices=list(plan.test_indices),
            )
            .scale(method="standard")
        )
        classical_session.fit(
            LogisticRegression(max_iter=1000, random_state=ctx.seed),
            task="classification",
        )
        c_test = classical_session.evaluate(partition="test")
        # sklearn twin on same split for disclosure
        tr = list(plan.train_indices)
        te = list(plan.test_indices)
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(frame.loc[tr, feat_cols])
        x_te = scaler.transform(frame.loc[te, feat_cols])
        y_tr = frame.loc[tr, "fault"].to_numpy()
        y_te = frame.loc[te, "fault"].to_numpy()
        clf = SGDClassifier(loss="log_loss", random_state=ctx.seed, max_iter=800)
        clf.fit(x_tr, y_tr)
        proba = clf.predict_proba(x_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        twin = {
            "accuracy": float(accuracy_score(y_te, pred)),
            "f1": float(f1_score(y_te, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_te, proba)),
        }
        stages["classical"] = {
            "status": "ok",
            "estimator": "LogisticRegression",
            "test_metrics": metrics_round(dict(c_test.metrics)),
            "sklearn_twin_test": metrics_round(twin),
            "note": "Centralized pooled baseline — disclosure contrast, not leakage",
        }
        write_results(ctx, stages["classical"], filename="classical.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["classical"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"classical: {exc}")

    ok_stages = sum(1 for v in stages.values() if v.get("status") == "ok")
    summary = {
        "status": "completed" if ok_stages >= 3 else "partial",
        "product": "Relay Edge Federated",
        "data": data_meta,
        "split": {
            "kind": plan.kind,
            "counts": split_counts,
            "group_column": "site",
            "train_sites": train_sites,
            "test_sites": test_sites,
        },
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "torch": TORCH_STATUS,
        "leakage_controls": [
            "group_split by site so held-out edges never train FedAvg clients",
            "Probabilistic fit uses the same inject_split indices",
            "Classical pooled baseline is a disclosure contrast on the same split",
            "Test evaluate_federated / evaluate after locks",
        ],
        "what_fails_if_leakage_ignored": [
            "Including test sites as FL clients invents cross-silo generalization",
            "Fitting probabilistic intervals on the full fleet hides miscalibration",
            "Pooling then splitting after feature stats overstates classical ROC",
        ],
        "limitations": [
            "Local FedAvg simulation — not a deployed cross-silo network",
            "Synthetic edge sensors; not a real IoT fleet",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "relay-edge-federated OK",
        {
            "federated": (stages.get("federated") or {}).get("status"),
            "probabilistic": (stages.get("probabilistic") or {}).get("status"),
            "classical": (stages.get("classical") or {}).get("status"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
