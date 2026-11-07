"""Real ANN subprocess regression tests, including fault containment and persistence."""
import pickle

import numpy as np
import pytest

from buildml.cbr.adapters.industry_ann import (
    add_vectors_to_ann_index,
    build_ann_index,
    query_ann_index,
)
from buildml.cbr.adapters.isolated_ann import IsolatedAnnIndex
from buildml.cbr.extras import hnswlib_spec_present
from buildml.core.errors import ValidationError


@pytest.fixture
def worker():
    if not hnswlib_spec_present():
        pytest.skip("real hnswlib backend required")
    index = IsolatedAnnIndex(np.array([[0., 0.], [1., 1.], [2., 2.]]),
                             metric="euclidean", ef_construction=50, m=8)
    yield index
    index.close()


def test_worker_reuses_process_and_returns_neighbours(worker):
    pid = worker._process.pid
    for _ in range(3):
        distances, labels = query_ann_index(worker, "isolated", np.array([[1., 1.]]), k=20)
        assert labels.shape == (1, 3)
        assert labels[0, 0] == 1
        assert distances[0, 0] == 0
        assert worker._process.pid == pid


def test_worker_retention_and_pickle_rebuild(worker):
    add_vectors_to_ann_index(worker, "isolated", np.array([[5., 5.]]), start_id=3)
    restored = pickle.loads(pickle.dumps(worker))
    try:
        assert restored._process is None
        assert restored.query(np.array([[5., 5.]]), 1)[1][0, 0] == 3
        assert worker.ntotal == restored.ntotal == 4
    finally:
        restored.close()


def test_worker_death_is_an_actionable_error_not_parent_crash(worker):
    worker._process.kill()
    worker._process.wait(timeout=5)
    with pytest.raises(ValidationError, match="parent process is intact"):
        worker.query(np.array([[0., 0.]]), 1)
    assert worker._process is None


def test_retention_refuses_misaligned_case_ids(worker):
    with pytest.raises(ValidationError, match="case-memory positions"):
        worker.add(np.array([[4., 4.]]), 99)


def test_nonfinite_vectors_rejected_before_native_entry():
    with pytest.raises(ValidationError, match="finite"):
        build_ann_index(np.array([[np.inf, 0.]]))


def test_windows_guard_routes_to_worker(monkeypatch, worker):
    monkeypatch.setattr("buildml.cbr.adapters.industry_ann.windows_industry_ann_refused", lambda: True)
    index, library = build_ann_index(np.array([[0., 0.], [1., 1.]]))
    try:
        assert library == "isolated"
        assert index.query(np.array([[0., 0.]]), 1)[1][0, 0] == 0
    finally:
        index.close()
