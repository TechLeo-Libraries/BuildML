"""Mapper summaries must report the real graph, not an empty placeholder."""
import numpy as np
import pytest

from buildml.tda.adapters.giotto import _optional_mapper_summary


@pytest.mark.parametrize("n", [80, 600])
def test_mapper_counts_match_real_igraph_and_bounded_training_sample(n):
    pytest.importorskip("gtda")
    from gtda.mapper import make_mapper_pipeline
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(49)
    x = rng.normal(scale=.2, size=(n, 3))
    x[n // 2:] += 3
    y = np.arange(n) % 2
    positions = np.random.default_rng(7).choice(n, size=500, replace=False) if n > 500 else np.arange(n)
    graph = make_mapper_pipeline(filter_func=PCA(n_components=3), cover=None,
                                 clusterer=DBSCAN(eps=.8, min_samples=3)).fit_transform(x[positions])
    assert graph.vcount() > 0
    summary = _optional_mapper_summary(x, y, random_state=7)
    assert summary["n_mapper_nodes"] == graph.vcount()
    assert summary["n_mapper_edges"] == graph.ecount()
    assert summary["n_train_mapper_points"] == min(n, 500)
    assert summary["has_labels"] is True


def test_mapper_unlabeled_input_and_small_population_disclosures():
    pytest.importorskip("gtda")
    rng = np.random.default_rng(61)
    assert _optional_mapper_summary(rng.normal(size=(10, 3)), None, random_state=0) is None
    summary = _optional_mapper_summary(rng.normal(size=(60, 3)), None, random_state=0)
    assert summary["n_train_mapper_points"] == 60
    assert "has_labels" not in summary
