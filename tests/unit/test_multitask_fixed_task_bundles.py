"""Contradictory fixed-task head metadata refuses before save or attachment."""

import copy
import json

import joblib
import pytest
from test_multitask_task_consumers import SEED, options, population

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.multitask.checkpoint import save_multitask_bundle


@pytest.mark.parametrize('task', ['classification', 'regression'])
@pytest.mark.parametrize('learner', ['multi_output', 'torch'])
@pytest.mark.parametrize('fault', ['all-opposite', 'one-opposite'])
@pytest.mark.parametrize('operation', ['save', 'load-direct', 'load-payload'])
def test_fixed_task_metadata_refuses_without_writing_or_replacing(task, learner, fault, operation, tmp_path):
    if learner == 'torch':
        torch = pytest.importorskip('torch')
        torch.manual_seed(SEED)
    source, _, _ = population(task + '-ordinary')
    source.multitask.fit(**options(learner, task, ['t0', 't1', 't2']))
    bad = copy.deepcopy(source.multitask_plan)
    opposite = 'classification' if task == 'regression' else 'regression'
    bad.task_kinds_ = dict.fromkeys(bad.target_columns, opposite) if fault == 'all-opposite' else (
        dict.fromkeys(bad.target_columns, task) | {'t1': opposite})
    if operation == 'save':
        root = tmp_path / 'bad'
        with pytest.raises(ValidationError):
            save_multitask_bundle(root, bad)
        assert not root.exists()
        return
    root = source.multitask.save_bundle(tmp_path / 'bad')
    joblib.dump(bad if operation == 'load-direct' else {'plan': bad}, root / 'multitask_plan.joblib')
    document = json.loads((root / 'meta.json').read_text(encoding='utf-8'))
    # Match the payload, as actual older producers did; a stale summary must
    # not hide acceptance of an internally contradictory contract.
    document['plan']['task_kinds'] = bad.task_kinds_
    (root / 'meta.json').write_text(json.dumps(document), encoding='utf-8')
    destination = Session(dataset=source.dataset, split_plan=source.split_plan)
    destination._multitask_plan = source.multitask_plan
    original = destination.multitask_plan
    fingerprint = joblib.hash(original)
    history = len(destination.history)
    with pytest.raises(ValidationError):
        destination.multitask.load_bundle(root, trusted=True)
    assert destination.multitask_plan is original
    assert joblib.hash(original) == fingerprint
    assert len(destination.history) == history
