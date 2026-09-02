"""Documentation and user-copy contract tests."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from buildml import Session
from scripts.lint_user_copy import (
    COPY_RULES,
    EM_DASH,
    MOJIBAKE_MARKERS,
    SOFT_LEAKAGE_FALSE_CLAIM,
    STALE_API,
    lint_paths,
)

ROOT = Path(__file__).resolve().parents[2]


def test_user_copy_lint_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/lint_user_copy.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_documented_v2_methods_exist() -> None:
    documented_methods = {
        "calibration",
        "checkpoint_load",
        "checkpoint_save",
        "eda",
        "evaluate",
        "explain",
        "feature_importance",
        "fit",
        "impute",
        "inject_split",
        "save_model",
        "scale",
        "set_roles",
        "split",
        "tune_threshold",
        "walkthrough",
        "workflow",
    }
    missing = sorted(name for name in documented_methods if not hasattr(Session, name))
    assert not missing


def test_readme_states_session_entry_and_legacy_boundary() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "The public entry point is `buildml.Session`" in readme
    assert "pip install buildml" in readme
    assert "BuildML 1.x legacy boundary" in readme
    assert "There is no compatibility shim" in readme


def test_sphinx_current_path_uses_session() -> None:
    current_pages = ("readme.rst", "usage.rst", "features.rst", "package.rst")
    text = "\n".join(
        (ROOT / "docs" / page).read_text(encoding="utf-8") for page in current_pages
    )
    assert "buildml.Session" in text
    assert "buildml.automate" not in text


def test_copy_lint_rejects_approved_banned_filler_and_stale_apis() -> None:
    banned = (
        "Executive narrative",
        "Actionable recommendations",
        "This is research-grade output",
        "Unlock seamless workflows",
    )
    for text in banned:
        assert any(pattern.search(text) for _, pattern in COPY_RULES), text
    assert STALE_API.search("Call buildml.preprocessing before fitting")
    assert EM_DASH.search("plain clause\u2014then more text")


def test_copy_lint_catches_multiline_soft_leakage_and_mojibake(tmp_path: Path) -> None:
    soft = tmp_path / "soft.py"
    soft.write_text(
        '"""Docs.\n'
        "allow_session_global_preprocess:\n"
        "    Explicit opt-in when Session-global preprocess already ran and no\n"
        "    fold-local recipe is provided (default False; refuses otherwise).\n"
        '"""\n',
        encoding="utf-8",
    )
    soft_hits = [item for item in lint_paths([soft]) if item.rule == "soft-leakage-false-claim"]
    assert soft_hits, "wrapped soft-leakage false claim must fail copy lint"

    assert SOFT_LEAKAGE_FALSE_CLAIM.search(
        "before CV without a fold recipe, treat preprocess honesty as limited"
    )
    assert MOJIBAKE_MARKERS.search("report mean\u00c2\u00b1std across folds")

    baked = tmp_path / "baked.py"
    baked.write_text('msg = "mean\u00c2\u00b1std"\n', encoding="utf-8")
    bake_hits = [item for item in lint_paths([baked]) if item.rule == "mojibake-text"]
    assert bake_hits


def test_generated_prose_matches_human_tone_fixture_without_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "age": [20, 22, None, 35, 35, 42] * 8,
            "constant": ["same"] * 48,
            "customer_id": [f"customer-{index}" for index in range(48)],
            "target": [0, 1, 0, 1, 0, 1] * 8,
        }
    )
    report = Session.ingest(frame).set_roles(
        {
            "age": "feature",
            "constant": "feature",
            "customer_id": "id",
            "target": "target",
        }
    ).eda(max_plots=0)
    fixture = json.loads(
        (ROOT / "tests" / "fixtures" / "human_tone.json").read_text(encoding="utf-8")
    )
    details = [finding.detail for finding in report.findings]
    titles = [recommendation.title for recommendation in report.recommendation_details]
    assert all(expected in details for expected in fixture["findings"])
    assert all(expected in titles for expected in fixture["recommendation_titles"])

    def normalize(value: str) -> str:
        return re.sub(r"\W+", " ", value.casefold()).strip()

    narrative = {normalize(value) for value in report.narrative}
    recommendations = {normalize(value) for value in report.recommendations}
    assert narrative.isdisjoint(recommendations)
    assert len(narrative) == len(report.narrative)
    assert len(recommendations) == len(report.recommendations)
    assert all(recommendation.based_on for recommendation in report.recommendation_details)


def test_domain_quickstart_contracts_match_resolvers() -> None:
    """Bind the 11 domain openings to the catalogs, not to mixin wishful thinking.

    Mixin defaults are the caller-facing knobs. Runtime backend/method can still
    change when ``backend=None`` / ``method=None`` and an extra is installed.
    These pages must state that resolution, not a constant that is only true on
    a bare core install.
    """
    import inspect

    from buildml.cbr.catalog import cbr_industry_available, resolve_backend_metric
    from buildml.federated.catalog import flwr_runtime_available
    from buildml.federated.catalog import resolve_backend as resolve_federated_backend
    from buildml.ranking.catalog import ranking_capability_matrix, resolve_backend_method
    from buildml.ranking.extras import ranking_industry_available
    from buildml.session.mixins.cbr import CbrSessionMixin
    from buildml.session.mixins.decision import DecisionSessionMixin
    from buildml.session.mixins.federated import FederatedSessionMixin
    from buildml.session.mixins.kg import KgSessionMixin
    from buildml.session.mixins.probabilistic import ProbabilisticSessionMixin
    from buildml.session.mixins.ranking import RankingSessionMixin
    from buildml.session.mixins.rl import RlSessionMixin
    from buildml.session.mixins.symbolic import SymbolicSessionMixin
    from buildml.session.mixins.synthetic import SyntheticSessionMixin
    from buildml.session.mixins.tda import TdaSessionMixin
    from buildml.tda.catalog import resolve_backend_vectorization
    from buildml.tda.extras import giotto_available, tda_available

    fed = inspect.signature(FederatedSessionMixin.fit_federated)
    assert fed.parameters["method"].default == "fedavg"
    assert fed.parameters["estimator"].default == "sgd_classifier"
    resolved_fed = resolve_federated_backend(None, method="fedavg")
    assert resolved_fed == ("flower" if flwr_runtime_available() else "native")

    prob = inspect.signature(ProbabilisticSessionMixin.fit_probabilistic)
    assert prob.parameters["estimator"].default == "bayesian_ridge"
    assert prob.parameters["conformal"].default is True
    assert prob.parameters["conformal_calibration_fraction"].default == 0.2

    sym = inspect.signature(SymbolicSessionMixin.fit_symbolic)
    assert sym.parameters["source"].default == "decision_tree"
    assert "fit_neuro" in SymbolicSessionMixin.__dict__ or hasattr(
        SymbolicSessionMixin, "fit_neuro_symbolic"
    )

    cbr = inspect.signature(CbrSessionMixin.fit_cbr)
    assert cbr.parameters["k"].default == 5
    assert cbr.parameters["metric"].default == "euclidean"
    resolved_cbr, _ = resolve_backend_metric(backend=None, metric="euclidean")
    assert resolved_cbr == ("industry" if cbr_industry_available() else "sklearn")

    rl = inspect.signature(RlSessionMixin.fit_rl)
    assert rl.parameters["algorithm"].default == "linucb"

    tda = inspect.signature(TdaSessionMixin.fit_tda)
    assert tda.parameters["vectorization"].default == "persistence_image"
    if tda_available():
        resolved_tda, vec = resolve_backend_vectorization(
            backend=None, vectorization="persistence_image"
        )
        assert vec == "persistence_image"
        assert resolved_tda == ("giotto" if giotto_available() else "native")

    rank = inspect.signature(RankingSessionMixin.fit_ranker)
    assert rank.parameters["method"].default is None
    default_rank_method = ranking_capability_matrix()["default_method_when_installed"]
    resolved_rank_backend, resolved_rank_method = resolve_backend_method(
        backend=None, method=str(default_rank_method)
    )
    if ranking_industry_available():
        assert resolved_rank_method != "pointwise" or resolved_rank_backend != "sklearn"
    else:
        assert resolved_rank_backend == "sklearn"
        assert resolved_rank_method == "pointwise"

    kg = inspect.signature(KgSessionMixin.fit_kg)
    assert kg.parameters["method"].default == "transe"

    decision = inspect.signature(DecisionSessionMixin.fit_decision_policy)
    assert decision.parameters["partition"].default == "validation"
    assert decision.parameters["allow_test_tuning"].default is False
    assert decision.parameters["score_source"].default == "model_proba"

    synth = inspect.signature(SyntheticSessionMixin.fit_synthesizer)
    assert synth.parameters["method"].default == "gaussian_copula"

    guides = ROOT / "guides"
    federated = (guides / "quickstart-federated.md").read_text(encoding="utf-8")
    ranking = (guides / "quickstart-ranking.md").read_text(encoding="utf-8")
    cbr_doc = (guides / "quickstart-cbr.md").read_text(encoding="utf-8")
    rl_doc = (guides / "quickstart-imitation-rl.md").read_text(encoding="utf-8")
    tda_doc = (guides / "quickstart-tda.md").read_text(encoding="utf-8")
    kg_doc = (guides / "quickstart-kg.md").read_text(encoding="utf-8")
    decision_doc = (guides / "quickstart-optimize.md").read_text(encoding="utf-8")
    synth_doc = (guides / "quickstart-synthetic.md").read_text(encoding="utf-8")
    ts_doc = (guides / "quickstart-timeseries-analysis.md").read_text(encoding="utf-8")
    prob_doc = (guides / "quickstart-probabilistic.md").read_text(encoding="utf-8")
    symbolic_doc = (guides / "quickstart-symbolic.md").read_text(encoding="utf-8")

    assert "Default is native FedAvg with SGD" not in federated
    assert "sgd_classifier" in federated
    assert "picks Flower" in federated
    assert "at least\ntwo eligible clients" in federated or "at least two eligible clients" in federated

    assert "and `relevance_column` are required" not in ranking
    assert "`relevance_column` defaults" in ranking
    assert "`method=None` picks LightGBM LambdaRank" in ranking
    assert "Default is sklearn pointwise." not in ranking

    assert "Default is exact sklearn kNN" not in cbr_doc
    assert "industry ANN" in cbr_doc
    assert "k=5" in cbr_doc

    assert "needs `reward_column`" not in rl_doc
    assert "name a column `reward`" in rl_doc
    assert "LinUCB" in rl_doc

    assert "persistence image" in tda_doc
    assert "buildml[tda]" in tda_doc
    assert "picks giotto" in tda_doc

    assert "head, relation, tail" in kg_doc or "`(head, relation, tail)`" in kg_doc
    assert "native TransE" in kg_doc

    assert "prior `session.fit`" in decision_doc
    assert "allow_test_tuning=True" in decision_doc

    assert "Gaussian copula" in synth_doc
    assert "extend_train" in synth_doc
    assert "FitResult" in synth_doc

    assert "a target" in ts_doc
    assert "time_split" in ts_doc

    assert "BayesianRidge" in prob_doc
    assert "20% of train" in prob_doc

    assert 'source="decision_tree"' in symbolic_doc or "decision tree" in symbolic_doc
    assert "fit_neuro" in symbolic_doc
