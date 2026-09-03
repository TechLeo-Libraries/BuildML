"""Lock the examples / guides / proofs triangle as a user-facing contract."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUIDES = ROOT / "guides"
EXAMPLES = ROOT / "examples"
PROOFS = ROOT / "proofs"

# Quickstart (or adjacent guide) → paste script. AI has no paste script.
QUICKSTART_EXAMPLE: dict[str, str] = {
    "quickstart-classical.md": "classical_loan_loop.py",
    "classical-end-to-end.md": "classical_loan_loop.py",
    "leakage-cv-recipes.md": "leakage_cv_recipe.py",
    "classical-diagnostics-search.md": "evolutionary_search_loop.py",
    "quickstart-unsupervised.md": "unsupervised_cluster_loop.py",
    "quickstart-ensemble.md": "ensemble_vote_stack_loop.py",
    "quickstart-automl.md": "automl_search_loop.py",
    "quickstart-forecasting.md": "forecast_lag_loop.py",
    "quickstart-timeseries-analysis.md": "timeseries_analyze_loop.py",
    "quickstart-anomaly.md": "anomaly_iforest_loop.py",
    "quickstart-semisupervised.md": "semisupervised_label_propagation_loop.py",
    "quickstart-selfsupervised.md": "selfsupervised_masked_tabular_loop.py",
    "quickstart-active-learning.md": "activelearning_margin_loop.py",
    "quickstart-online-learning.md": "online_partial_fit_loop.py",
    "quickstart-multi-task.md": "multitask_multioutput_loop.py",
    "quickstart-meta-learning.md": "metalearning_prototypical_loop.py",
    "quickstart-federated.md": "federated_fedavg_loop.py",
    "quickstart-probabilistic.md": "probabilistic_bayesian_ridge.py",
    "quickstart-causal.md": "causal_aipw_ate.py",
    "quickstart-graph.md": "graph_node_classification.py",
    "quickstart-symbolic.md": "symbolic_rules_loop.py",
    "quickstart-cbr.md": "cbr_knn_loop.py",
    "quickstart-imitation-rl.md": "imitation_rl_loop.py",
    "quickstart-tda.md": "tda_loop.py",
    "quickstart-recommenders.md": "recommender_item_knn_loop.py",
    "quickstart-ranking.md": "ranking_pointwise_loop.py",
    "quickstart-kg.md": "kg_transe_loop.py",
    "quickstart-optimize.md": "decision_threshold_loop.py",
    "quickstart-synthetic.md": "synthetic_copula_loop.py",
    "quickstart-nlp.md": "nlp_text_classifier_loop.py",
    "quickstart-rag.md": "rag_hashing_loop.py",
    "quickstart-fairness.md": "fairness_observational_loop.py",
    "quickstart-torch.md": "torch_tabular_mlp_loop.py",
}

SCOREBOARD = re.compile(r"\b(?:63/63|36/36|58/62|62/62|R1.R6)\b")
API_STEPS = "BuildML API steps"
PRODUCT_NARRATIVE = "Product narrative"
WIN_PATH = re.compile(r"python proofs\\")


def _readme_table_scripts() -> set[str]:
    text = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    return set(re.findall(r"`([a-z0-9_]+\.py)`", text))


def test_proof_index_run_commands_point_at_real_files() -> None:
    text = (PROOFS / "README.md").read_text(encoding="utf-8")
    missing = []
    for slug, name in re.findall(r"python proofs/([a-z0-9-]+)/([A-Za-z0-9_.]+)", text):
        if not (PROOFS / slug / name).is_file():
            missing.append(f"proofs/{slug}/{name}")
    assert not missing, f"proofs/README.md runs missing files: {missing}"
    assert "loan-approval-classical/baseline_industry.py" not in text
    assert "checkout-only" in text
    assert "examples/breast_cancer_classical_loop.py" in text


def test_every_example_script_is_listed_in_examples_readme() -> None:
    on_disk = {path.name for path in EXAMPLES.glob("*.py") if not path.name.startswith("_")}
    listed = _readme_table_scripts()
    missing = sorted(on_disk - listed)
    extra = sorted(listed - on_disk)
    assert not missing, f"examples/README.md missing scripts: {missing}"
    assert not extra, f"examples/README.md lists missing files: {extra}"


def test_quickstarts_link_their_paste_script() -> None:
    missing: list[str] = []
    for page, script in QUICKSTART_EXAMPLE.items():
        text = (GUIDES / page).read_text(encoding="utf-8")
        if f"examples/{script}" not in text:
            missing.append(f"{page} -> {script}")
    assert not missing, "guides missing paste links:\n" + "\n".join(missing)


def test_guides_and_proof_index_have_no_suite_scoreboard() -> None:
    pages = [
        GUIDES / "README.md",
        PROOFS / "README.md",
        EXAMPLES / "README.md",
    ]
    hits: list[str] = []
    for path in pages:
        text = path.read_text(encoding="utf-8")
        if SCOREBOARD.search(text):
            hits.append(path.as_posix())
    assert not hits, f"scoreboard counts still on {hits}"


def test_proof_readmes_are_user_evidence_not_checklists() -> None:
    banned: list[str] = []
    for path in PROOFS.rglob("README.md"):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT).as_posix()
        if API_STEPS in text:
            banned.append(f"{rel}: API steps")
        if PRODUCT_NARRATIVE in text:
            banned.append(f"{rel}: Product narrative")
        if WIN_PATH.search(text):
            banned.append(f"{rel}: Windows proof path")
        if SCOREBOARD.search(text) and path.name == "README.md" and path.parent == PROOFS:
            banned.append(f"{rel}: scoreboard")
    assert not banned, "proof README regressions:\n" + "\n".join(banned)


def _tier_b_slugs() -> list[str]:
    """Parse TIER_B from run_all.py without importing the harness (Torch-safe)."""
    source = (PROOFS / "_lib" / "run_all.py").read_text(encoding="utf-8")
    match = re.search(r"TIER_B = \[(.*?)\n\]", source, re.S)
    assert match is not None, "TIER_B list not found in proofs/_lib/run_all.py"
    return re.findall(r'"([a-z0-9-]+)"', match.group(1))


def test_composition_proofs_say_they_are_not_shipped_products() -> None:
    missing = []
    for slug in _tier_b_slugs():
        text = re.sub(
            r"\s+",
            " ",
            (PROOFS / slug / "README.md").read_text(encoding="utf-8"),
        )
        if "not a product BuildML ships" not in text:
            missing.append(slug)
    assert not missing, f"composition READMEs missing honesty line: {missing}"


def test_recommender_example_uses_resolver_defaults() -> None:
    tree = ast.parse((EXAMPLES / "recommender_item_knn_loop.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("recommender.fit")
    ]
    assert calls, "expected session.recommender.fit in the example"
    keywords = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords if kw.arg}
    assert keywords.get("method") == "None"
    assert keywords.get("feedback") == "'explicit'"


def test_federated_example_matches_quickstart_draw() -> None:
    text = (EXAMPLES / "federated_fedavg_loop.py").read_text(encoding="utf-8")
    assert "default_rng(0)" in text
    assert "for i in range(40)" in text
    assert "n_rounds=5" in text
    assert "fit_federated" not in text


def test_torch_example_skips_without_the_extra() -> None:
    text = (EXAMPLES / "torch_tabular_mlp_loop.py").read_text(encoding="utf-8")
    assert "pip install 'buildml[torch]'" in text
    assert "except ImportError" in text


def test_cbr_example_forces_sklearn_backend() -> None:
    text = (EXAMPLES / "cbr_knn_loop.py").read_text(encoding="utf-8")
    assert 'backend="sklearn"' in text


def test_breast_cancer_example_is_pasteable() -> None:
    text = (EXAMPLES / "breast_cancer_classical_loop.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(name == "proofs" or name.startswith("proofs.") for name in imported)
    assert "load_breast_cancer" in text
    assert "session.calibration()" in text
    assert "session.tune_threshold" in text


def test_fairness_example_shows_group_split() -> None:
    text = (EXAMPLES / "fairness_observational_loop.py").read_text(encoding="utf-8")
    assert "group_split" in text
    assert '"household": "group"' in text
    guide = (GUIDES / "quickstart-fairness.md").read_text(encoding="utf-8")
    assert "group_split" in guide


def test_example_bundles_stay_beside_the_script() -> None:
    offenders = []
    for path in EXAMPLES.glob("*.py"):
        if path.name.startswith("_"):
            continue
        text = path.read_text(encoding="utf-8")
        if 'Path("artifacts"' in text or "Path('artifacts'" in text:
            offenders.append(path.name)
        if 'save_bundle("artifacts/' in text:
            offenders.append(path.name)
    assert not offenders, f"examples still write cwd artifacts/: {offenders}"


def test_composition_scripts_open_with_harness_banner() -> None:
    slugs = _tier_b_slugs()
    missing = []
    for slug in slugs:
        head = (PROOFS / slug / "script.py").read_text(encoding="utf-8")[:400]
        if "Not a product BuildML ships" not in head:
            missing.append(slug)
    assert not missing, f"composition scripts missing banner: {missing}"
