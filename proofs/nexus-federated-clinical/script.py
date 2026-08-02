"""Tier B product: Nexus Federated Clinical.

Composes a multi-hospital FedAvg simulation + probabilistic uncertainty
intervals + honest evaluation disclosures. Local FL only — not a deployed
cross-silo network.
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
from proofs._lib import (
    TORCH_STATUS,
    metrics_round,
    new_proof_context,
    write_results,
)


def _hospital_cohort(n_hospitals: int = 6, n_per: int = 90, seed: int = 17):
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(n_hospitals):
        shift = h * 0.18
        for _ in range(n_per):
            labs = rng.normal(size=5) + shift
            # Site-shift risk score → binary readmit-ish label
            logit = (
                -0.3
                + 0.9 * labs[0]
                + 0.55 * labs[1]
                - 0.35 * labs[2]
                + rng.normal(0, 0.3)
            )
            y = int(1 / (1 + np.exp(-logit)) > 0.5)
            rows.append(
                {
                    **{f"lab{i}": float(labs[i]) for i in range(5)},
                    "readmit": y,
                    "hospital": f"h{h}",
                }
            )
    frame = pd.DataFrame(rows)
    meta = {
        "name": "nexus_synthetic_hospital_cohort",
        "license": "synthetic/public-domain",
        "n_rows": int(len(frame)),
        "n_hospitals": n_hospitals,
        "positive_rate": float(frame["readmit"].mean()),
        "notes": "Synthetic labs + site shift; not PHI / not a licensed clinical dataset.",
    }
    return frame, meta


def main() -> None:
    ctx = new_proof_context("nexus-federated-clinical", seed=17)
    frame, data_meta = _hospital_cohort(seed=ctx.seed)
    stages: dict = {}
    skip_notes: list[str] = []

    session = (
        Session.ingest(frame.copy())
        .set_roles(
            {
                **{f"lab{i}": "feature" for i in range(5)},
                "readmit": "target",
                "hospital": "group",
            }
        )
        .group_split(
            test_size=0.2,
            validation_size=0.15,
            random_state=ctx.seed,
            group_column="hospital",
        )
    )
    plan = session.split_plan
    assert plan is not None
    split_counts = {
        "train": len(plan.train_indices),
        "validation": len(plan.validation_indices),
        "test": len(plan.test_indices),
    }
    train_hospitals = sorted(
        frame.loc[list(plan.train_indices), "hospital"].unique().tolist()
    )
    test_hospitals = sorted(
        frame.loc[list(plan.test_indices), "hospital"].unique().tolist()
    )

    # --- Stage 1: federated FedAvg simulation ---
    try:
        fit = session.fit_federated(
            method="fedavg",
            client_column="hospital",
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
            "train_hospitals": train_hospitals,
            "test_hospitals": test_hospitals,
        }
        write_results(ctx, stages["federated"], filename="federated.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["federated"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"federated: {exc}")

    # --- Stage 2: probabilistic uncertainty on a pooled regression proxy ---
    # Use a continuous risk score target derived from labs (train-scope fit).
    try:
        risk_frame = frame.copy()
        risk_frame["risk_score"] = (
            0.9 * risk_frame["lab0"]
            + 0.55 * risk_frame["lab1"]
            - 0.35 * risk_frame["lab2"]
            + np.random.default_rng(ctx.seed).normal(0, 0.35, size=len(risk_frame))
        )
        prob_session = (
            Session.ingest(risk_frame)
            .set_roles(
                {
                    **{f"lab{i}": "feature" for i in range(5)},
                    "risk_score": "target",
                    "hospital": "group",
                    "readmit": "ignore",
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

    # --- Stage 3: centralized pooled baseline (disclosure contrast, not leakage) ---
    try:
        from sklearn.linear_model import SGDClassifier
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.preprocessing import StandardScaler

        feat_cols = [f"lab{i}" for i in range(5)]
        tr = list(plan.train_indices)
        te = list(plan.test_indices)
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(frame.loc[tr, feat_cols])
        x_te = scaler.transform(frame.loc[te, feat_cols])
        y_tr = frame.loc[tr, "readmit"].to_numpy()
        y_te = frame.loc[te, "readmit"].to_numpy()
        clf = SGDClassifier(
            loss="log_loss", max_iter=800, random_state=ctx.seed, tol=1e-3
        )
        clf.fit(x_tr, y_tr)
        pred = clf.predict(x_te)
        stages["pooled_centralized_contrast"] = {
            "status": "ok",
            "note": (
                "Pooled centralized SGD on the same train rows — contrast only. "
                "Not used to tune federated hyperparameters."
            ),
            "test_metrics": metrics_round(
                {
                    "accuracy": float(accuracy_score(y_te, pred)),
                    "f1_macro": float(f1_score(y_te, pred, average="macro", zero_division=0)),
                }
            ),
        }
    except Exception as exc:  # noqa: BLE001
        stages["pooled_centralized_contrast"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"pooled_contrast: {exc}")

    summary = {
        "status": "completed",
        "product": "Nexus Federated Clinical",
        "data": data_meta,
        "split": {
            "kind": getattr(plan, "kind", "group"),
            "counts": split_counts,
            "group_column": "hospital",
            "train_hospitals": train_hospitals,
            "test_hospitals": test_hospitals,
        },
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "torch": TORCH_STATUS,
        "disclosures": [
            "Local FedAvg simulation — raw rows stay in-process; not a deployed FL network",
            "Aggregation is in-process weighted coefficient averaging — not cryptographic secure aggregation",
            "No PHI; synthetic labs with site shift only",
            "Probabilistic intervals are empirical coverage tools, not clinical guarantees",
            "Pooled centralized contrast is disclosure-only; federated HPs not tuned on it or on test",
        ],
        "leakage_controls": [
            "group_split by hospital before any federated / probabilistic fit",
            "Federated local updates use train-client rows only",
            "Holdout hospitals/rows reserved for evaluate_federated",
            "Probabilistic model fit on train; intervals evaluated on test after lock",
        ],
        "what_fails_if_leakage_ignored": [
            "Including holdout hospitals in FedAvg rounds invents generalization",
            "Tuning rounds/client fraction on test overstates multi-site accuracy",
            "Fitting probabilistic intervals on the evaluation set understates coverage error",
        ],
        "limitations": [
            "Simulation honesty: not production cross-silo FL",
            "Not a clinical decision support device; no regulatory claim",
            "Site shift is synthetic and mild",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "nexus-federated-clinical OK",
        {
            "fed_metrics": (stages.get("federated") or {}).get("test_metrics"),
            "prob_metrics": (stages.get("probabilistic") or {}).get("test_metrics"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
