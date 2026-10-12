"""Advertised domain APIs must remain importable through their public package."""
import importlib

import pytest


@pytest.mark.parametrize("domain", [
    "ai", "dl", "rag", "symbolic", "causal", "probabilistic", "cbr",
    "graph", "online", "ranking", "recommenders", "federated", "metalearning",
    "multitask", "selfsupervised", "semisupervised", "synthetic", "tda",
    "rl", "optimize", "anomaly", "forecasting", "unsupervised", "nlp",
])
def test_advertised_public_symbols_resolve_and_unknown_names_fail(domain):
    package = importlib.import_module(f"buildml.{domain}")
    exports = getattr(package, "__all__", ())
    assert exports, f"Public domain {domain} has no advertised API"
    for name in exports:
        value = getattr(package, name)
        assert value is not None, f"Public API {domain}.{name} resolves to None"
    with pytest.raises(AttributeError):
        _ = package.buildml_nonexistent_contract_symbol
