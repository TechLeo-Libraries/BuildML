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
    """Bind quickstart and deep-guide copy to the catalogs, not mixin wishful thinking.

    Mixin defaults are the caller-facing knobs. Runtime backend/method can still
    change when ``backend=None`` / ``method=None`` and an extra is installed.
    Both the short on-ramp and the long guide must state that resolution, not a
    constant that is only true on a bare core install.
    """
    import inspect

    from buildml.cbr.catalog import cbr_industry_available, resolve_backend_metric
    from buildml.federated.catalog import flwr_runtime_available
    from buildml.federated.catalog import resolve_backend as resolve_federated_backend
    from buildml.ranking.catalog import ranking_capability_matrix, resolve_backend_method
    from buildml.ranking.extras import ranking_industry_available
    from buildml.recommenders.catalog import default_method_for_feedback
    from buildml.recommenders.extras import implicit_available
    from buildml.session.mixins.cbr import CbrSessionMixin
    from buildml.session.mixins.decision import DecisionSessionMixin
    from buildml.session.mixins.federated import FederatedSessionMixin
    from buildml.session.mixins.kg import KgSessionMixin
    from buildml.session.mixins.probabilistic import ProbabilisticSessionMixin
    from buildml.session.mixins.ranking import RankingSessionMixin
    from buildml.session.mixins.recommender import RecommenderSessionMixin
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

    rec = inspect.signature(RecommenderSessionMixin.fit_recommender)
    assert rec.parameters["method"].default is None
    assert rec.parameters["feedback"].default == "explicit"
    assert default_method_for_feedback("explicit") == "item_knn"
    assert default_method_for_feedback("implicit") == (
        "als" if implicit_available() else "nmf"
    )

    forbidden = (
        "Default is native FedAvg with SGD",
        "Default is sklearn pointwise.",
        "Default is exact sklearn kNN",
        "and `relevance_column` are required",
        "Default without extras is item kNN",
    )
    pages: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "quickstart-federated.md",
            ("sgd_classifier", "picks Flower"),
        ),
        (
            "federated-deep.md",
            ("sgd_classifier", "picks Flower"),
        ),
        (
            "quickstart-ranking.md",
            ("`relevance_column` defaults", "LightGBM LambdaRank"),
        ),
        (
            "ranking-deep.md",
            ("`relevance_column` defaults", "LightGBM LambdaRank"),
        ),
        (
            "quickstart-cbr.md",
            ("industry ANN", "k=5"),
        ),
        (
            "cbr-deep.md",
            ("industry ANN", "k` is 5"),
        ),
        (
            "quickstart-imitation-rl.md",
            ("LinUCB", "name a column `reward`"),
        ),
        (
            "imitation-rl-deep.md",
            ("linucb", "column literally named"),
        ),
        (
            "quickstart-tda.md",
            ("persistence image", "picks giotto"),
        ),
        (
            "tda-deep.md",
            ("persistence_image", "giotto"),
        ),
        (
            "quickstart-kg.md",
            ("native TransE",),
        ),
        (
            "kg-deep.md",
            ("native TransE", "stays native"),
        ),
        (
            "quickstart-optimize.md",
            ("prior `session.fit`", "allow_test_tuning=True"),
        ),
        (
            "optimize-deep.md",
            ("prior `session.fit`", "allow_test_tuning=True"),
        ),
        (
            "quickstart-synthetic.md",
            ("Gaussian copula", "extend_train", "FitResult"),
        ),
        (
            "synthetic-deep.md",
            ("gaussian_copula", "extend_train", "FitResult"),
        ),
        (
            "quickstart-timeseries-analysis.md",
            ("a target", "time_split"),
        ),
        (
            "timeseries-analysis-deep.md",
            ("a target", "time_split"),
        ),
        (
            "quickstart-probabilistic.md",
            ("BayesianRidge", "20% of train"),
        ),
        (
            "probabilistic-deep.md",
            ("bayesian_ridge", "0.2"),
        ),
        (
            "quickstart-symbolic.md",
            ("decision tree", "fit_neuro"),
        ),
        (
            "symbolic-deep.md",
            ('source="decision_tree"', "fit_neuro"),
        ),
        (
            "quickstart-recommenders.md",
            ("item kNN", "picks ALS"),
        ),
        (
            "recommenders-deep.md",
            ("item kNN", "picks ALS"),
        ),
    )
    guides = ROOT / "guides"
    for name, required in pages:
        text = (guides / name).read_text(encoding="utf-8")
        flat = re.sub(r"\s+", " ", text)
        for phrase in forbidden:
            assert phrase not in text and phrase not in flat, (
                f"{name} still states {phrase!r}"
            )
        for phrase in required:
            assert phrase in text or phrase in flat, f"{name} is missing {phrase!r}"


def test_sphinx_contributor_notes_are_in_toctree() -> None:
    """Contributor notes stay reachable without a visible nav entry."""
    index = (ROOT / "docs" / "index.rst").read_text(encoding="utf-8")
    assert "pypi-2x-publish" in index
    assert "session-facade-migration" in index
    hidden = index.split(".. toctree::")[-1]
    assert ":hidden:" in hidden
    assert "pypi-2x-publish" in hidden
    assert "session-facade-migration" in hidden


def test_sphinx_guide_wrappers_include_markdown() -> None:
    """Read the Docs must render guides/*.md, not a stale RST copy."""
    docs = ROOT / "docs"
    wrappers = sorted(docs.glob("quickstart-*.rst"))
    wrappers.extend(sorted(docs.glob("*-deep.rst")))
    assert wrappers, "expected Sphinx wrappers under docs/"
    missing: list[str] = []
    for path in wrappers:
        text = path.read_text(encoding="utf-8")
        if ".. include:: ../guides/" not in text:
            missing.append(path.name)
    assert not missing, (
        "Sphinx wrappers still duplicate Markdown instead of including it: "
        + ", ".join(missing)
    )
