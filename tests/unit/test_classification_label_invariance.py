"""Categorical label renaming cannot change classification behavior."""
import numpy as np
import pytest

from buildml.data.dataset import Dataset
from buildml.ingest.detect import schema_from_dataframe
from tests.unit import test_executor_domain_contracts as helpers

CASES = [case for case in helpers.WORKFLOWS if case[0] in {
    'ensemble', 'online', 'probabilistic', 'symbolic', 'cbr', 'multitask',
    'federated', 'metalearning', 'nlp', 'imitation', 'semisupervised',
}]


@pytest.mark.parametrize('labels', [(3, 7), ('negative', 'positive')])
@pytest.mark.parametrize('domain,fit_tool,args,evaluate_tool', CASES)
def test_classification_metrics_are_invariant_to_label_names(labels, domain, fit_tool, args, evaluate_tool):
    baseline = helpers._session(domain=domain)
    renamed = helpers._session(domain=domain)
    frame = renamed.to_pandas().copy()
    for column, role in renamed.dataset.roles.items():
        if str(getattr(role, 'value', role)) == 'target':
            frame[column] = frame[column].map({0: labels[0], 1: labels[1]})
    renamed._dataset = Dataset.from_transformed(renamed.dataset, frame,
        schema=schema_from_dataframe(frame), roles=dict(renamed.dataset.roles))
    helpers._run(baseline, fit_tool, args)
    helpers._run(renamed, fit_tool, args)
    expected = helpers._run(baseline, evaluate_tool, {'partition': 'test'})
    actual = helpers._run(renamed, evaluate_tool, {'partition': 'test'})
    assert actual.metrics and expected.metrics
    assert actual.metrics.keys() == expected.metrics.keys()
    for metric, value in expected.metrics.items():
        if isinstance(value, (int, float)) and np.isfinite(value):
            assert actual.metrics[metric] == pytest.approx(value), f'{domain}: {metric} changed under categorical renaming'
