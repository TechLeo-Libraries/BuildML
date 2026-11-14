"""AutoGluon metrics retain BuildML meaning and selection uses held-out scores."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from buildml.automl.adapters.autogluon import _autogluon_metric, _leaderboard_trials
from buildml.core.errors import ValidationError


@pytest.mark.parametrize('metric', ['rmse','mae','log_loss'])
def test_signed_loss_leaderboards_are_normalized_before_ranking(metric):
    table = pd.DataFrame({'model':['worse','best'], 'score_val':[-4.,-1.]})
    trials = _leaderboard_trials(table, metric=metric, higher_is_better=False)
    assert [t.params['model'] for t in trials] == ['best','worse']
    assert [t.mean_score for t in trials] == [1.,4.]
    assert trials[0].mean_metrics == {metric:1.}


def test_weighted_f1_keeps_its_averaging_definition():
    assert _autogluon_metric('f1_weighted', task='classification') == 'f1_weighted'


def test_validation_scores_are_used_even_when_other_score_columns_come_first():
    table = pd.DataFrame({'model':['best','worse'], 'score_test':[-99.,-.1], 'score_val':[-1.,-4.]})
    trials = _leaderboard_trials(table, metric='rmse', higher_is_better=False)
    assert [t.params['model'] for t in trials] == ['best','worse']
    assert [t.mean_score for t in trials] == [1.,4.]


def test_negative_higher_is_better_scores_are_not_flipped():
    table = pd.DataFrame({'model':['worse','best'], 'score_val':[-4.,-1.]})
    trials = _leaderboard_trials(table, metric='r2', higher_is_better=True)
    assert [t.mean_score for t in trials] == [-1.,-4.]


def test_unknown_metrics_and_missing_validation_scores_are_refused():
    with pytest.raises(ValidationError, match='ranking_metric'):
        _autogluon_metric('made_up_metric', task='classification')
    with pytest.raises(ValidationError, match='validation'):
        _leaderboard_trials(pd.DataFrame({'model':['a'], 'fit_time':[1.]}),
                            metric='rmse', higher_is_better=False)


@pytest.mark.parametrize("metric", ["r2", "rmse"])
def test_real_autogluon_identifies_the_score_source_with_singleton_validation(
    metric, monkeypatch, tmp_path,
):
    pytest.importorskip("autogluon.tabular")
    from buildml import Session

    monkeypatch.chdir(tmp_path)
    x, y = make_regression(n_samples=120, n_features=2, noise=1.0, random_state=3)
    frame = pd.DataFrame(x, columns=["signal", "noise"]).assign(target=y)
    session = Session.ingest(frame).set_roles({"target": "target"}).inject_split(
        train_indices=list(range(80)),
        validation_indices=[80],
        test_indices=list(range(81, 120)),
    )
    with pytest.warns(UserWarning, match=r"R\^2 score is not well-defined"):
        result = session.automl.run(
            backend="autogluon", task="regression", selection="validation",
            ranking_metric=metric, time_budget=30, random_state=9,
        )

    plan = session.automl_plan
    wrapper = session.fit_result.estimator
    if metric == "r2":
        predictor = wrapper._predictor
        leaderboard = predictor.leaderboard(silent=True).set_index("model")
        expected = leaderboard.loc[predictor.model_best, "score_val"]
        source = "provider_internal"
        assert any("finite 'r2'" in message for message in result.warnings)
    else:
        validation = session.partition("validation")
        predictions = wrapper.predict(validation[list(plan.feature_columns)])
        expected = np.sqrt(np.mean((validation["target"].to_numpy() - predictions) ** 2))
        source = "session_validation"
        assert result.warnings == ()
    assert np.isfinite(expected)
    assert result.best_score == pytest.approx(expected)
    assert plan.best_score == pytest.approx(expected)
    assert plan.warnings == result.warnings
    assert plan.config["extras"]["best_score_source"] == source
    assert result.config["extras"]["best_score_source"] == source
    json.dumps(plan.to_dict(), allow_nan=False)
    json.dumps(result.to_dict(), allow_nan=False)
    assert session.automl.evaluate(partition="test").n_rows == 39


@pytest.mark.parametrize('task,classes,metric,selection', [
    ('classification',3,'f1_weighted','validation'),
    ('classification',2,'f1_weighted','cv'),
    ('regression',0,'rmse','cv'),
])
def test_real_autogluon_selection_preserves_train_boundary_and_metric(
    task, classes, metric, selection, monkeypatch, tmp_path,
):
    ag = pytest.importorskip('autogluon.tabular')
    from buildml import Session

    monkeypatch.chdir(tmp_path)
    if task == 'classification':
        x,y = make_classification(n_samples=180, n_features=3, n_informative=3,
            n_redundant=0, n_classes=classes, n_clusters_per_class=1, random_state=3)
    else:
        x,y = make_regression(n_samples=180, n_features=3, noise=1., random_state=3)
    frame = pd.DataFrame(x, columns=['a','b','c'])
    frame['row_id'] = np.arange(len(frame))
    frame['target'] = y
    session = Session.ingest(frame).set_roles({**{c:'feature' for c in frame.columns[:-1]},
        'target':'target'}).split(test_size=.2, validation_size=.2, random_state=5,
                                stratify=task == 'classification')
    original = ag.TabularPredictor.fit
    captured = {}
    def fit(predictor, train_data, **kwargs):
        captured['rows'] = set(train_data.row_id)
        captured['kwargs'] = kwargs
        return original(predictor, train_data, **kwargs)
    monkeypatch.setattr(ag.TabularPredictor, 'fit', fit)
    result = session.automl.run(backend='autogluon', task=task, selection=selection,
        ranking_metric=metric, time_budget=30., random_state=9)
    assert captured['rows'] == set(session.split_plan.train_indices)
    assert captured['kwargs']['hyperparameters']['GBM']['seed'] == 9
    predictor = session.automl_plan.estimator_._predictor
    assert predictor.problem_type == ('multiclass' if classes > 2 else
                                      'binary' if classes == 2 else 'regression')
    assert predictor.eval_metric.name == ('root_mean_squared_error' if metric == 'rmse' else metric)
    leaderboard = predictor.leaderboard(silent=True)
    scores = {str(row['model']):float(row['score_val']) for _,row in leaderboard.iterrows()}
    sign = -1. if metric == 'rmse' else 1.
    for trial in result.trials:
        assert trial.mean_score == pytest.approx(sign*scores[trial.params['model']])
    if selection == 'cv':
        assert result.best_score == pytest.approx(sign*scores[predictor.model_best])
    evaluation = session.automl.evaluate(partition='test')
    assert evaluation.n_rows == len(session.split_plan.test_indices)
    assert np.isfinite(evaluation.metrics[metric])
    if task == 'classification':
        from buildml.automl.checkpoint import load_automl_bundle

        wrapper = session.automl_plan.estimator_
        features = session.partition('test')[list(session.automl_plan.feature_columns)]
        provider = predictor.predict_proba(features)
        assert list(wrapper.classes_) == list(provider.columns) == predictor.class_labels
        raw = provider.to_numpy(dtype=float)
        expected = raw / raw.sum(axis=1, keepdims=True)
        probabilities = wrapper.predict_proba(features)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-14, rtol=0)
        np.testing.assert_allclose(probabilities, expected, atol=1e-14, rtol=0)
        labels = session.partition('test')['target'].to_numpy()
        row_label_probabilities = expected[np.arange(len(labels)), provider.columns.get_indexer(labels)]
        loss = -np.log(np.clip(row_label_probabilities, np.finfo(float).eps, 1)).mean()
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message='The y_prob values do not sum to one')
            evaluation = session.automl.evaluate(partition='test')
        assert evaluation.metrics['log_loss'] == pytest.approx(loss, abs=1e-12, rel=0)

        bundle = session.automl.save_bundle(tmp_path/'bundle')
        original_path = Path(predictor.path).resolve()
        assert original_path.is_relative_to(tmp_path.resolve())
        nested = original_path/'nested-bundle'
        with pytest.raises(ValidationError, match='outside'):
            session.automl.save_bundle(nested)
        assert not nested.exists()
        meta_path = Path(bundle)/'meta.json'
        original_meta = meta_path.read_text()
        meta = json.loads(original_meta)
        for invalid in ('../outside', str(tmp_path.resolve()), '.', 42):
            meta['autogluon_assets'] = invalid
            meta_path.write_text(json.dumps(meta))
            with pytest.raises(ValidationError, match='assets'):
                load_automl_bundle(bundle, trusted=True)
        meta_path.write_text(original_meta)
        retired = (tmp_path/'retired-original').resolve()
        assert retired.is_relative_to(tmp_path.resolve()) and not retired.exists()
        original_path.rename(retired)
        relocated = (tmp_path/'relocated-bundle').resolve()
        assert relocated.is_relative_to(tmp_path.resolve()) and not relocated.exists()
        Path(bundle).rename(relocated)
        with pytest.raises(ValidationError, match='trusted'):
            load_automl_bundle(relocated)
        loaded, _ = load_automl_bundle(relocated, trusted=True)
        np.testing.assert_array_equal(loaded.estimator_.classes_, wrapper.classes_)
        np.testing.assert_allclose(loaded.estimator_.predict_proba(features), expected, atol=1e-14, rtol=0)
