"""Probe honesty: industry ``available`` must not trust find_spec alone.

These tests monkeypatch spec-present True while forcing subprocess import
failure, then assert capability matrices / industry_available stay False.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_import_cache() -> None:
    from buildml.dl.extras import clear_subprocess_import_cache

    clear_subprocess_import_cache()
    yield
    clear_subprocess_import_cache()


def _patch_subprocess_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "buildml.dl.extras._subprocess_import_ok",
        lambda module, timeout=12.0: False,
    )


def test_ranking_industry_false_when_import_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from buildml.ranking import catalog as ranking_catalog
    from buildml.ranking import extras as ranking_extras

    monkeypatch.setattr(ranking_extras, "lightgbm_spec_present", lambda: True)
    monkeypatch.setattr(ranking_extras, "xgboost_spec_present", lambda: False)
    monkeypatch.setattr(ranking_extras, "catboost_spec_present", lambda: False)
    monkeypatch.setattr(ranking_catalog, "lightgbm_spec_present", lambda: True)
    monkeypatch.setattr(ranking_catalog, "xgboost_spec_present", lambda: False)
    monkeypatch.setattr(ranking_catalog, "catboost_spec_present", lambda: False)
    monkeypatch.setattr(ranking_catalog, "ranking_industry_available", lambda: False)
    monkeypatch.setattr(ranking_catalog, "lightgbm_available", lambda: False)
    monkeypatch.setattr(ranking_catalog, "xgboost_available", lambda: False)
    monkeypatch.setattr(ranking_catalog, "catboost_available", lambda: False)
    _patch_subprocess_false(monkeypatch)
    assert ranking_extras.ranking_industry_available() is False
    matrix = ranking_catalog.ranking_capability_matrix()
    assert matrix["backends"]["industry"]["available"] is False
    assert matrix["industry_extra_present"] is True
    assert matrix["industry_runtime_present"] is False


def test_causal_industry_false_when_import_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from buildml.causal import catalog as causal_catalog
    from buildml.causal import extras as causal_extras

    monkeypatch.setattr(causal_extras, "dowhy_spec_present", lambda: True)
    monkeypatch.setattr(causal_extras, "econml_spec_present", lambda: True)
    monkeypatch.setattr(causal_catalog, "dowhy_spec_present", lambda: True)
    monkeypatch.setattr(causal_catalog, "econml_spec_present", lambda: True)
    monkeypatch.setattr(causal_catalog, "dowhy_available", lambda: False)
    monkeypatch.setattr(causal_catalog, "econml_available", lambda: False)
    monkeypatch.setattr(causal_catalog, "causal_industry_available", lambda: False)
    _patch_subprocess_false(monkeypatch)
    assert causal_extras.causal_industry_available() is False
    matrix = causal_catalog.causal_capability_matrix()
    assert matrix["backends"]["dowhy"]["available"] is False
    assert matrix["backends"]["econml"]["available"] is False
    assert matrix["industry_extra_present"] is True
    assert matrix["industry_runtime_present"] is False


def test_kg_industry_false_when_pykeen_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from buildml.kg import catalog as kg_catalog
    from buildml.kg import extras as kg_extras

    monkeypatch.setattr(kg_extras, "pykeen_spec_present", lambda: True)
    monkeypatch.setattr(kg_catalog, "pykeen_spec_present", lambda: True)
    monkeypatch.setattr(kg_catalog, "pykeen_available", lambda: True)
    monkeypatch.setattr(kg_catalog, "pykeen_runtime_available", lambda: False)
    monkeypatch.setattr(kg_catalog, "kg_industry_available", lambda: False)
    monkeypatch.setattr("buildml.dl.extras.torch_available", lambda: True)
    _patch_subprocess_false(monkeypatch)
    assert kg_extras.kg_industry_available() is False
    assert kg_extras.pykeen_runtime_available() is False
    matrix = kg_catalog.kg_capability_matrix()
    assert matrix["backends"]["pykeen"]["available"] is False
    assert matrix["pykeen_spec_present"] is True
    assert matrix["industry_runtime_present"] is False


def test_graph_pyg_runtime_false_when_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from buildml.graph import extras as graph_extras

    monkeypatch.setattr(graph_extras, "pyg_spec_present", lambda: True)
    monkeypatch.setattr("buildml.dl.extras.torch_available", lambda: True)
    _patch_subprocess_false(monkeypatch)
    assert graph_extras.pyg_runtime_available() is False


def test_federated_industry_false_when_flwr_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from buildml.federated import catalog as fl_catalog
    from buildml.federated import extras as fl_extras

    monkeypatch.setattr(fl_extras, "flwr_spec_present", lambda: True)
    monkeypatch.setattr(fl_catalog, "flwr_spec_present", lambda: True)
    monkeypatch.setattr(fl_catalog, "flwr_runtime_available", lambda: False)
    monkeypatch.setattr(fl_catalog, "federated_industry_available", lambda: False)
    _patch_subprocess_false(monkeypatch)
    assert fl_extras.federated_industry_available() is False
    matrix = fl_catalog.federated_capability_matrix()
    assert matrix["backends"]["flower"]["available"] is False
    assert matrix["industry_extra_present"] is True
    assert matrix["industry_runtime_present"] is False


def test_rl_industry_false_when_import_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from buildml.rl import catalog as rl_catalog
    from buildml.rl import extras as rl_extras

    monkeypatch.setattr(rl_extras, "stable_baselines3_spec_present", lambda: True)
    monkeypatch.setattr(rl_extras, "imitation_spec_present", lambda: True)
    monkeypatch.setattr(rl_extras, "gymnasium_available", lambda: True)
    monkeypatch.setattr(rl_catalog, "stable_baselines3_spec_present", lambda: True)
    monkeypatch.setattr(rl_catalog, "imitation_spec_present", lambda: True)
    monkeypatch.setattr(rl_catalog, "gymnasium_available", lambda: True)
    monkeypatch.setattr(rl_catalog, "stable_baselines3_available", lambda: False)
    monkeypatch.setattr(rl_catalog, "imitation_available", lambda: False)
    monkeypatch.setattr(rl_catalog, "rl_industry_available", lambda: False)
    _patch_subprocess_false(monkeypatch)
    assert rl_extras.rl_industry_available() is False
    matrix = rl_catalog.rl_capability_matrix()
    assert matrix["rl_backends"]["industry"]["available"] is False
    assert matrix["industry_extra_present"] is True
    assert matrix["industry_runtime_present"] is False
    assert matrix["rl_industry_extra_present"] is True


def test_optimize_industry_false_when_import_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from buildml.optimize import extras as opt_extras

    monkeypatch.setattr(opt_extras, "pulp_spec_present", lambda: True)
    monkeypatch.setattr(opt_extras, "ortools_spec_present", lambda: False)
    monkeypatch.setattr(opt_extras, "cvxpy_spec_present", lambda: False)
    monkeypatch.setattr(opt_extras, "xgboost_spec_present", lambda: False)
    _patch_subprocess_false(monkeypatch)
    assert opt_extras.optimize_industry_available() is False


def test_online_industry_false_when_river_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from buildml.online import extras as online_extras

    monkeypatch.setattr(online_extras, "river_spec_present", lambda: True)
    _patch_subprocess_false(monkeypatch)
    assert online_extras.online_industry_available() is False


def test_multitask_industry_false_when_gbdt_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from buildml.multitask import extras as mt_extras

    monkeypatch.setattr(mt_extras, "lightgbm_spec_present", lambda: True)
    monkeypatch.setattr(mt_extras, "xgboost_spec_present", lambda: True)
    monkeypatch.setattr(mt_extras, "catboost_spec_present", lambda: True)
    _patch_subprocess_false(monkeypatch)
    assert mt_extras.multitask_industry_available() is False


def test_semisupervised_st_false_when_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from buildml.semisupervised import extras as ss_extras

    monkeypatch.setattr(ss_extras, "sentence_transformers_spec_present", lambda: True)
    monkeypatch.setattr("buildml.dl.extras.torch_available", lambda: True)
    _patch_subprocess_false(monkeypatch)
    assert ss_extras.sentence_transformers_available() is False


def test_forecasting_industry_false_when_statsmodels_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from buildml.forecasting import extras as fc_extras

    monkeypatch.setattr(fc_extras, "statsmodels_spec_present", lambda: True)
    _patch_subprocess_false(monkeypatch)
    assert fc_extras.industry_forecast_available() is False


def test_symbolic_industry_false_when_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from buildml.symbolic import extras as sym_extras

    monkeypatch.setattr(sym_extras, "skope_rules_spec_present", lambda: True)
    monkeypatch.setattr(sym_extras, "imodels_spec_present", lambda: False)
    _patch_subprocess_false(monkeypatch)
    assert sym_extras.symbolic_industry_available() is False


def test_tda_industry_false_when_giotto_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from buildml.tda import catalog as tda_catalog
    from buildml.tda import extras as tda_extras

    monkeypatch.setattr(tda_extras, "giotto_spec_present", lambda: True)
    monkeypatch.setattr(tda_extras, "ripser_spec_present", lambda: True)
    monkeypatch.setattr(tda_extras, "persim_spec_present", lambda: True)
    monkeypatch.setattr(tda_catalog, "giotto_spec_present", lambda: True)
    monkeypatch.setattr(tda_catalog, "ripser_spec_present", lambda: True)
    monkeypatch.setattr(tda_catalog, "persim_spec_present", lambda: True)
    _patch_subprocess_false(monkeypatch)
    assert tda_extras.tda_industry_available() is False
    matrix = tda_catalog.tda_capability_matrix()
    assert matrix["tda_industry_extra_present"] is True
    assert matrix["tda_industry_runtime_present"] is False


def test_unsupervised_extra_false_when_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from buildml.unsupervised import extras as uns_extras

    monkeypatch.setattr(uns_extras, "hdbscan_spec_present", lambda: True)
    monkeypatch.setattr(uns_extras, "umap_spec_present", lambda: True)
    _patch_subprocess_false(monkeypatch)
    assert uns_extras.unsupervised_extra_available() is False


def test_recommenders_industry_false_when_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from buildml.recommenders import catalog as rec_catalog
    from buildml.recommenders import extras as rec_extras

    monkeypatch.setattr(rec_extras, "implicit_spec_present", lambda: True)
    monkeypatch.setattr(rec_extras, "lightfm_spec_present", lambda: False)
    monkeypatch.setattr(rec_catalog, "implicit_spec_present", lambda: True)
    monkeypatch.setattr(rec_catalog, "lightfm_spec_present", lambda: False)
    _patch_subprocess_false(monkeypatch)
    assert rec_extras.recommenders_industry_available() is False
    matrix = rec_catalog.recommender_capability_matrix()
    assert matrix["industry_extra_present"] is True
    assert matrix["industry_runtime_present"] is False


def test_online_matrix_separates_spec_from_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from buildml.online import catalog as online_catalog
    from buildml.online import extras as online_extras

    monkeypatch.setattr(online_extras, "river_spec_present", lambda: True)
    monkeypatch.setattr(online_catalog, "river_spec_present", lambda: True)
    _patch_subprocess_false(monkeypatch)
    assert online_extras.online_industry_available() is False
    matrix = online_catalog.online_capability_matrix()
    assert matrix["industry_extra_present"] is True
    assert matrix["industry_runtime_present"] is False


def test_anomaly_matrix_separates_spec_from_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from buildml.anomaly import catalog as anomaly_catalog
    from buildml.anomaly import extras as anomaly_extras

    monkeypatch.setattr(anomaly_extras, "pyod_spec_present", lambda: True)
    monkeypatch.setattr(anomaly_extras, "lightgbm_spec_present", lambda: False)
    monkeypatch.setattr(anomaly_extras, "xgboost_spec_present", lambda: False)
    monkeypatch.setattr(anomaly_catalog, "pyod_spec_present", lambda: True)
    monkeypatch.setattr(anomaly_catalog, "lightgbm_spec_present", lambda: False)
    monkeypatch.setattr(anomaly_catalog, "xgboost_spec_present", lambda: False)
    _patch_subprocess_false(monkeypatch)
    assert anomaly_extras.anomaly_industry_available() is False
    matrix = anomaly_catalog.anomaly_capability_matrix()
    assert matrix["industry_extra_present"] is True
    assert matrix["industry_runtime_present"] is False


def test_activelearning_industry_extra_is_skactiveml_spec_not_always_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from buildml.activelearning import catalog as al_catalog
    from buildml.activelearning import extras as al_extras

    monkeypatch.setattr(al_extras, "scikit_activeml_spec_present", lambda: False)
    monkeypatch.setattr(al_catalog, "scikit_activeml_spec_present", lambda: False)
    matrix = al_catalog.activelearning_capability_matrix()
    assert matrix["backends"]["industry"]["available"] is True
    assert matrix["industry_extra_present"] is False
    assert matrix["scikit_activeml_present"] is False
