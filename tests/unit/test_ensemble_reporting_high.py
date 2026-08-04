"""HIGH-depth ensemble reporting: contributions, diversity, leakage-safe evaluate."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from buildml import Session
from buildml.ensemble import build_ensemble_eval_report, ensemble_capability_matrix


def _frame(n: int = 120, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + 0.4 * x2 + rng.normal(scale=0.45, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _bases():
    return {
        "lr": LogisticRegression(max_iter=500),
        "rf": RandomForestClassifier(n_estimators=20, random_state=0),
    }


def test_evaluate_ensemble_attaches_contribution_and_diversity() -> None:
    session = (
        Session.ingest(_frame())
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.25, validation_size=0.2, stratify=True, random_state=0)
        .scale(method="standard")
    )
    session.ensemble.fit_voting(_bases(), voting="soft")
    result = session.ensemble.evaluate(partition="test")
    assert "base_contributions" in result.diagnostics
    assert "diversity" in result.diagnostics
    assert "ensemble_report" in result.diagnostics
    contribs = result.diagnostics["base_contributions"]
    assert len(contribs) == 2
    names = {c["name"] for c in contribs}
    assert names == {"lr", "rf"}
    for c in contribs:
        assert "accuracy" in c["metrics"]
        assert c["agree_with_ensemble"] is not None
        assert 0.0 <= float(c["agree_with_ensemble"]) <= 1.0
    diversity = result.diagnostics["diversity"]
    assert diversity["n_bases"] == 2
    assert diversity["mean_pairwise_disagreement"] is not None
    assert 0.0 <= float(diversity["mean_pairwise_disagreement"]) <= 1.0
    joined = " ".join(result.recommendations).lower()
    assert "predict-only" in joined or "diversity" in joined


def test_stacking_contribution_predict_only_no_refit() -> None:
    session = (
        Session.ingest(_frame(140))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=1)
        .scale(method="standard")
    )
    session.ensemble.fit_stacking(_bases(), cv=3)
    plan = session.ensemble.plan
    assert plan is not None
    # Capture base params fingerprint before evaluate.
    est = session._fit_result.estimator
    before = {
        name: tuple(np.round(getattr(model, "coef_", np.array([])).ravel(), 6))
        if hasattr(model, "coef_")
        else id(model)
        for name, model in est.named_estimators_.items()
    }
    report = build_ensemble_eval_report(
        session.dataset,
        session.split_plan,
        session._fit_result,
        plan,
        partition="test",
        ensemble_metrics={},
    )
    after = {
        name: tuple(np.round(getattr(model, "coef_", np.array([])).ravel(), 6))
        if hasattr(model, "coef_")
        else id(model)
        for name, model in est.named_estimators_.items()
    }
    assert before == after
    assert any("predict-only" in d.lower() for d in report.disclosures)
    assert any("test never re-enters" in d.lower() for d in report.disclosures)
    assert report.diversity is not None
    assert report.diversity.n_bases >= 2


def test_blending_evaluate_report_leakage_disclosure() -> None:
    session = (
        Session.ingest(_frame(160))
        .set_roles({"x1": "feature", "x2": "feature", "y": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=2)
        .scale(method="standard")
    )
    fit = session.ensemble.fit_blending(_bases(), holdout_fraction=0.2, random_state=0)
    assert any("holdout" in d.lower() for d in fit.disclosures)
    ev = session.ensemble.evaluate(partition="validation")
    report = ev.diagnostics["ensemble_report"]
    joined = " ".join(report["disclosures"]).lower()
    assert "leakage-safe" in joined or "train" in joined
    assert len(report["base_contributions"]) == 2


def test_catalog_discloses_reporting_enrichment() -> None:
    matrix = ensemble_capability_matrix()
    assert "reporting" in matrix
    assert "base_contributions" in matrix["reporting"]["evaluate_enrichment"]
    assert "leakage_safe_defaults" in matrix["reporting"]
