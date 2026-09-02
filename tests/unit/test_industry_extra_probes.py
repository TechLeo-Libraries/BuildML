"""Industry extra probe rows must match installed package extras."""

from __future__ import annotations

import re
from importlib.metadata import distribution
from importlib.util import find_spec

import pytest

from scripts.probe_industry_extras import PROBES

# Import names that do not match the distribution name on PyPI.
_MODULE_TO_DIST = {
    "imblearn": "imbalanced-learn",
    "sentence_transformers": "sentence-transformers",
    "gtda": "giotto-tda",
    "skactiveml": "scikit-activeml",
    "stable_baselines3": "stable-baselines3",
    "torch_geometric": "torch-geometric",
    "skrules": "skope-rules",
    "z3": "z3-solver",
    "autogluon.tabular": "autogluon.tabular",
    "umap": "umap-learn",
    "faiss": "faiss-cpu",
}


def _installed_extra_dists() -> dict[str, set[str]]:
    dist = distribution("buildml")
    extras: dict[str, set[str]] = {
        extra: set() for extra in (dist.metadata.get_all("Provides-Extra") or [])
    }
    for req in dist.requires or []:
        match = re.search(r"""extra\s*==\s*['"]([^'"]+)['"]""", req)
        if match is None:
            continue
        extra = match.group(1)
        name = req.split(";")[0].split("[")[0].strip()
        name = re.split(r"[<>=!~]", name, maxsplit=1)[0].strip().lower()
        extras.setdefault(extra, set()).add(name)
    return extras


def _dist_for_module(module: str) -> str:
    return _MODULE_TO_DIST.get(module, module.replace("_", "-")).lower()


def test_probe_keys_exist_on_the_installed_package() -> None:
    extras = _installed_extra_dists()
    missing = sorted(name for name in PROBES if name not in extras)
    assert not missing, f"probe extras are not package extras: {missing}"


def test_every_industry_extra_is_probed() -> None:
    extras = _installed_extra_dists()
    industry = sorted(name for name in extras if name.endswith("-industry"))
    missing = [name for name in industry if name not in PROBES]
    assert not missing, f"industry extras have no probe row: {missing}"


def _declared_dists(extra: str, extras: dict[str, set[str]]) -> set[str]:
    declared = set(extras.get(extra, set()))
    # Nested extras are recorded as a bare "buildml" requirement. Union the
    # extras that this extra is documented to wrap (same prefix family).
    if extra.endswith("-industry"):
        base = extra.removesuffix("-industry")
        declared.update(extras.get(base, set()))
    if extra == "graph-pyg":
        declared.update(extras.get("graph", set()))
    if extra == "timeseries-ml" or extra == "timeseries-prophet":
        declared.update(extras.get("timeseries", set()))
    if extra == "nlp-industry":
        declared.update(extras.get("nlp", set()))
    if extra == "rl-industry":
        declared.update(extras.get("rl", set()))
    if extra == "tda-industry":
        declared.update(extras.get("tda", set()))
    if extra == "speech":
        declared.update(extras.get("torch", set()))
    return declared


def test_probe_modules_belong_to_the_named_extra() -> None:
    extras = _installed_extra_dists()
    mismatches: list[str] = []
    for extra, modules in PROBES.items():
        declared = _declared_dists(extra, extras)
        for module in modules:
            dist_name = _dist_for_module(module)
            if dist_name not in declared:
                mismatches.append(
                    f"{extra}:{module}->{dist_name} not in {sorted(declared)}"
                )
    assert not mismatches, "\n".join(mismatches)


def test_cbr_industry_probe_is_hnswlib_only() -> None:
    assert PROBES["cbr-industry"] == ("hnswlib",)
    assert PROBES["cbr-faiss"] == ("faiss",)


def test_optimize_industry_probe_does_not_claim_optuna() -> None:
    assert "optuna" not in PROBES["optimize-industry"]
    assert PROBES["optuna"] == ("optuna",)


@pytest.mark.skipif(find_spec("buildml") is None, reason="buildml must be importable")
def test_session_extras_are_probed() -> None:
    for extra in ("shap", "imbalanced", "polars", "duckdb", "speech", "serve", "ai"):
        assert extra in PROBES
