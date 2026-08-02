"""Decision policy value benchmark: cost-optimal vs baseline threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.optimize.catalog import decision_capability_matrix
from buildml.optimize.extras import optimize_industry_available, xgboost_available


def _reference_session(seed: int = 7) -> Session:
    x, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=6,
        weights=[0.75, 0.25],
        random_state=seed,
    )
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    frame["y"] = y
    return (
        Session.ingest(frame)
        .set_roles({**{c: "feature" for c in frame.columns if c.startswith("f")}, "y": "target"})
        .split(test_size=0.25, validation_size=0.25, random_state=seed)
        .fit(LogisticRegression(max_iter=600), task="classification")
    )


def _eval_expected_cost(session: Session, *, fp_cost: float, fn_cost: float) -> dict[str, object]:
    ev = session.evaluate_decisions(partition="test")
    metrics = dict(ev.metrics)
    return {
        "threshold": session.decision_plan.threshold if session.decision_plan else None,
        "backend": (
            session.decision_fit_result.backend if session.decision_fit_result else None
        ),
        "expected_cost_total": metrics.get("expected_cost_total"),
        "realized_cost": ev.realized_cost,
        "f1": metrics.get("f1"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BuildML decision policy value benchmark (cost vs baseline)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/optimize/results/policy_value.json"),
    )
    args = parser.parse_args(argv)

    fp_cost, fn_cost = 1.0, 8.0
    runs: list[dict[str, object]] = []

    baseline = _reference_session()
    baseline.fit_decision_policy(
        method="threshold",
        backend="native",
        partition="validation",
        fp_cost=fp_cost,
        fn_cost=fn_cost,
    )
    runs.append(
        {
            "label": "cost_optimal_native",
            **_eval_expected_cost(baseline, fp_cost=fp_cost, fn_cost=fn_cost),
        }
    )

    fixed = _reference_session()
    fixed.fit_decision_policy(
        method="threshold",
        backend="native",
        partition="validation",
        fp_cost=fp_cost,
        fn_cost=fn_cost,
    )
    if fixed.decision_plan is not None:
        fixed.decision_plan.threshold = 0.5
    runs.append(
        {
            "label": "baseline_threshold_0.5",
            **_eval_expected_cost(fixed, fp_cost=fp_cost, fn_cost=fn_cost),
        }
    )

    if xgboost_available():
        xgb_session = _reference_session()
        xgb_session.fit_decision_policy(
            method="threshold",
            backend="xgb",
            partition="validation",
            fp_cost=fp_cost,
            fn_cost=fn_cost,
        )
        runs.append(
            {
                "label": "cost_optimal_xgb",
                **_eval_expected_cost(xgb_session, fp_cost=fp_cost, fn_cost=fn_cost),
            }
        )

    cal_session = _reference_session()
    cal_session.fit_decision_policy(
        method="threshold",
        backend="calibrated",
        partition="validation",
        fp_cost=fp_cost,
        fn_cost=fn_cost,
    )
    runs.append(
        {
            "label": "cost_optimal_calibrated",
            **_eval_expected_cost(cal_session, fp_cost=fp_cost, fn_cost=fn_cost),
        }
    )

    native_cost = next(
        (r.get("realized_cost") for r in runs if r.get("label") == "cost_optimal_native"),
        None,
    )
    baseline_cost = next(
        (r.get("realized_cost") for r in runs if r.get("label") == "baseline_threshold_0.5"),
        None,
    )
    lift = None
    if native_cost is not None and baseline_cost is not None and baseline_cost > 0:
        lift = float(baseline_cost - native_cost) / float(baseline_cost)

    payload = {
        "capability_matrix": decision_capability_matrix(),
        "fp_cost": fp_cost,
        "fn_cost": fn_cost,
        "runs": runs,
        "cost_lift_vs_baseline_0.5": lift,
        "optimize_industry_available": optimize_industry_available(),
        "xgboost_available": xgboost_available(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "n_runs": len(runs), "lift": lift}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
