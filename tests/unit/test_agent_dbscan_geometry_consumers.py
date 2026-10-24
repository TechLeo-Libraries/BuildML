"""Agent DBSCAN labels match an independent density graph and frozen core geometry."""

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry


def _run(session, name, arguments, success=True):
    registry = build_default_registry()
    result = execute_tool(session, propose_tool_execution(name, arguments, registry),
                          confirmed=True, registry=registry)
    assert result.executed is success, result.error
    return result.result if success else result.error


def _density_graph(points, eps, minimum):
    neighbors = [[j for j, other in enumerate(points)
                  if sum((float(a) - float(b)) ** 2 for a, b in zip(point, other)) <= eps**2]
                 for point in points]
    cores = [i for i, group in enumerate(neighbors) if len(group) >= minimum]
    labels = [-1] * len(points)
    cluster = 0
    for seed in cores:
        if labels[seed] >= 0:
            continue
        frontier = [seed]
        while frontier:
            i = frontier.pop()
            if labels[i] >= 0:
                continue
            labels[i] = cluster
            if i in cores:
                frontier.extend(j for j in neighbors[i] if labels[j] < 0)
        cluster += 1
    return np.array(labels), cores


def _assign(points, train, labels, cores, eps):
    result = []
    for point in points:
        if not cores:
            result.append(-1)
            continue
        nearest = min(cores, key=lambda i: sum((float(a) - float(b))**2 for a, b in zip(point, train[i])))
        distance = sum((float(a) - float(b))**2 for a, b in zip(point, train[nearest]))**.5
        result.append(int(labels[nearest]) if distance <= eps else -1)
    return result


@pytest.mark.parametrize('eps', [.25, .5])
@pytest.mark.parametrize('minimum', [2, 3, 99])
@pytest.mark.parametrize('partition', ['validation', 'test'])
def test_agent_dbscan_frozen_density_geometry_and_attachment(eps, minimum, partition):
    train = np.array([[0, 0], [0, .125], [.125, 0], [.125, .125],
                      [4, 0], [4, .125], [4.125, 0], [4.125, .125], [20, 20]], dtype=float)
    query = np.array([[-eps, 0], [-np.nextafter(eps, np.inf), 0], [4, 0], [2, 0], [20, 20]])
    x = np.vstack([train, query, query[::-1]])
    frame = pd.DataFrame(x, columns=['x', 'z'])
    session = Session.ingest(frame).set_roles({'x': 'feature', 'z': 'feature'})
    session.inject_split(train_indices=list(range(9)), validation_indices=list(range(9, 14)),
                         test_indices=list(range(14, 19)))
    before = session.to_pandas().copy(deep=True)
    expected_labels, cores = _density_graph(train, eps, minimum)
    fit = _run(session, 'fit_clusters', {'method': 'dbscan', 'eps': eps,
                'min_samples': minimum, 'prefer_reduce_components': False})
    plan = session.cluster_plan
    np.testing.assert_array_equal(plan.train_labels_, expected_labels)
    assert tuple(plan.core_sample_indices_) == tuple(cores)
    assert fit.n_train_rows == 9
    fingerprint = joblib.hash(plan)
    points = query if partition == 'validation' else query[::-1]
    assigned = _run(session, 'assign_clusters', {'partition': partition})
    expected = _assign(points, train, expected_labels, cores, eps)
    np.testing.assert_array_equal(assigned.labels, expected)
    assert assigned.n_noise == expected.count(-1)
    assert assigned.assign_strategy == 'nearest_core'
    assert joblib.hash(plan) == fingerprint
    history = list(session._history)
    error = _run(session, 'assign_clusters', {'partition': partition, 'attach': True}, False)
    assert "partition='all'" in error
    assert session._history == history
    pd.testing.assert_frame_equal(session.to_pandas(), before)
    result = _run(session, 'assign_clusters', {'partition': 'all', 'attach': True})
    assert result.attached
    np.testing.assert_array_equal(session.to_pandas()[plan.label_column],
                                  _assign(x, train, expected_labels, cores, eps))
    pd.testing.assert_frame_equal(session.to_pandas()[before.columns], before)
    assert joblib.hash(plan) == fingerprint
    assert tuple(session.split_plan.train_indices) == tuple(range(9))
    history = list(session._history)
    error = _run(session, 'assign_clusters', {'partition': 'all', 'attach': True}, False)
    assert 'already exists' in error
    assert session._history == history
