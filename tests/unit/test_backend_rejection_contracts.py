"""Unknown backend names cannot silently select an available implementation."""
import importlib

import pytest

from buildml.core.errors import ValidationError

RESOLVERS = [
    ('activelearning', 'resolve_backend_strategy', {'strategy': 'margin'}),
    ('anomaly', 'resolve_backend_method', {'method': 'isolation_forest', 'mode': 'point'}),
    ('causal', 'resolve_backend_method', {'method': 't_learner'}),
    ('cbr', 'resolve_backend_metric', {'metric': 'euclidean'}),
    ('federated', 'resolve_backend', {'method': 'fedavg'}),
    ('kg', 'resolve_backend_method', {'method': 'transe'}),
    ('metalearning', 'resolve_backend_method', {'method': 'prototypical'}),
    ('multitask', 'resolve_backend_method', {'method': 'independent'}),
    ('nlp', 'resolve_backend_estimator', {'estimator': 'logistic_regression'}),
    ('nlp', 'resolve_entity_backend', {}),
    ('nlp', 'resolve_language_backend', {}),
    ('online', 'resolve_backend_estimator', {'estimator': 'sgd'}),
    ('optimize', 'resolve_backend', {'method': 'threshold'}),
    ('probabilistic', 'resolve_backend_estimator', {'estimator': 'split', 'task': 'regression'}),
    ('ranking', 'resolve_backend_method', {'method': 'pointwise'}),
    ('recommenders', 'resolve_backend_method', {'method': 'popularity', 'feedback': 'implicit'}),
    ('rl', 'resolve_imitation_backend_method', {'estimator': None, 'method': None, 'task': 'classification'}),
    ('rl', 'resolve_rl_backend_mode_algorithm', {'mode': None, 'algorithm': None}),
    ('semisupervised', 'resolve_backend_method', {'method': 'self_training'}),
    ('symbolic', 'resolve_symbolic_backend_method', {'source': 'decision_tree', 'method': None}),
    ('symbolic', 'resolve_neuro_symbolic_backend', {'base_estimator': 'logistic_regression'}),
    ('synthetic', 'resolve_backend_method', {'method': 'bootstrap'}),
    ('tda', 'resolve_backend_vectorization', {'vectorization': 'persistence_image'}),
]


@pytest.mark.parametrize('package,function,kwargs', RESOLVERS)
def test_unknown_backend_fails_explicitly(package, function, kwargs):
    resolver = getattr(importlib.import_module(f'buildml.{package}.catalog'), function)
    with pytest.raises(ValidationError, match='(?i)backend'):
        resolver(backend='unsupported_backend', **kwargs)
