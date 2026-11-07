"""Real Optuna studies agree with independent objectives and frozen consumers."""

import copy

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from buildml import Session
from buildml.automl.types import AutoMLBudget
from buildml.preprocess import PreprocessRecipe

optuna = pytest.importorskip("optuna")
BACKINGS = ["pandas", "pandas-parquet", "polars-memory", "polars-lazy", "duckdb-memory", "duckdb-lazy"]
TRAIN = [*range(1,64,2), *range(0,64,2)]


def _frame(task="regression", labels="text"):
    rng = np.random.default_rng(73)
    x, z = rng.normal(size=(2,96))
    y = 1.5+2*x-.8*z+rng.normal(scale=.6,size=96)
    if task == "classification":
        y = np.where(x+.5*z > 0,"yes","no") if labels == "text" else x+.5*z > 0
    frame = pd.DataFrame(dict(x=x,z=z,y=y,weight=np.where(np.arange(96)%7,1.,7.)))
    frame.index = [f"row-{i//2}" for i in range(96)]
    return frame


def _session(frame, tmp_path, backing="pandas"):
    engine = backing.split("-")[0]
    if engine != "pandas":
        pytest.importorskip(engine)
    if backing.endswith(("parquet","lazy")):
        path = tmp_path / "source.parquet"
        frame.to_parquet(path,index=False)
        session = Session.ingest(path,engine=engine,mode="memory" if engine == "pandas" else "lazy")
    else:
        session = Session.ingest(frame,engine=engine)
    session.set_roles({"x":"feature","z":"feature","y":"target","weight":"weight"})
    return session.inject_split(train_indices=TRAIN,validation_indices=list(range(64,80)),test_indices=list(range(80,96)))


def _prediction(fit, score, task, params):
    if task == "classification":
        model = DecisionTreeClassifier(random_state=17,**params)
        model.fit(fit[["x","z"]],fit.y,sample_weight=fit.weight)
        return model.predict(score[["x","z"]])
    design = np.column_stack([np.ones(len(fit)),fit[["x","z"]]])
    weighted = design.T*fit.weight.to_numpy()
    beta = np.linalg.solve(weighted@design+np.diag([0.,params["alpha"],params["alpha"]]),weighted@fit.y)
    return np.column_stack([np.ones(len(score)),score[["x","z"]]])@beta


def _metrics(frame, predicted, task):
    weight = frame.weight.to_numpy()
    truth = frame.y.to_numpy()
    if task == "regression":
        error = truth-predicted
        mse = np.average(error**2,weights=weight)
        variance = np.average((truth-np.average(truth,weights=weight))**2,weights=weight)
        return dict(mae=np.average(abs(error),weights=weight),mse=mse,rmse=np.sqrt(mse),r2=1-mse/variance)
    recalls, f1s, supports = [], [], []
    for value in np.unique(np.concatenate([truth,predicted])):
        actual, chosen = truth == value, predicted == value
        tp = weight[actual & chosen].sum()
        support = weight[actual].sum()
        predicted_support = weight[chosen].sum()
        recalls.append(tp/support if support else 0.)
        f1s.append(2*tp/(support+predicted_support) if support+predicted_support else 0.)
        supports.append(support)
    return dict(accuracy=weight[truth == predicted].sum()/weight.sum(),balanced_accuracy=np.mean(recalls),
                f1_weighted=np.average(f1s,weights=supports),f1_macro=np.mean(f1s))


def _scores(session, task, selection, params):
    train = session.partition("train").reset_index(drop=True)
    pairs = [(train,session.partition("validation"))] if selection == "validation" else [
        (train.iloc[fit],train.iloc[score]) for fit,score in KFold(2,shuffle=True,random_state=42).split(train)]
    rows = [_metrics(score,_prediction(fit,score,task,params),task) for fit,score in pairs]
    means = {key:np.mean([row[key] for row in rows]) for key in rows[0]}
    stds = {key:np.std([row[key] for row in rows],ddof=1) if len(rows)>1 else 0. for key in rows[0]}
    return means,stds


def _run(session, tmp_path, task, primary, secondary, pruning=True, storage=True,
         multi=True, selection="cv", route="backend"):
    uri = "sqlite:///"+(tmp_path / "study.db").as_posix() if storage else None
    budget = AutoMLBudget(max_trials=2,enable_pruning=pruning,multi_objective=multi,
                          secondary_metric=secondary,study_storage=uri)
    requested = copy.deepcopy(budget.to_dict())
    result = session.automl.run(backend="optuna" if route == "backend" else "native",
        method="optuna",task=task,selection=selection,ranking_metric=primary,
        families=("ridge" if task == "regression" else "decision_tree",),n_trials=2,cv=2,
        cv_strategy="kfold",include_recipe_search=False,preprocess=PreprocessRecipe(impute=None),
        budget=budget,random_state=17)
    assert budget.to_dict() == requested
    assert len(result.trials) == 2
    for trial in result.trials:
        expected,std = _scores(session,task,selection,trial.params)
        assert trial.mean_score == pytest.approx(expected[primary],abs=1e-10)
        assert trial.std_score == pytest.approx(std[primary],abs=1e-10)
        for key in expected:
            assert trial.mean_metrics[key] == pytest.approx(expected[key],abs=1e-10)
    optimum = (min if primary in {"mae","mse","rmse"} else max)(t.mean_score for t in result.trials)
    assert result.best_score == pytest.approx(optimum)
    if multi and pruning:
        assert any("prun" in d.lower() and "disabl" in d.lower() for d in result.disclosures)
    return result,uri


def _bundle_consumer(session, tmp_path, task, result):
    before, split = session.to_pandas().copy(), session.split_plan
    winner = result.trials[0]
    expected = _prediction(session.partition("train"),session.partition("test"),task,winner.params)
    if task == "regression":
        np.testing.assert_allclose(session.predict(partition="test"),expected,atol=2e-10)
    else:
        np.testing.assert_array_equal(session.predict(partition="test"),expected)
    path = session.automl.save_bundle(tmp_path / "winner")
    session.automl.load_bundle(path,trusted=True)
    frozen = joblib.hash(session.fit_result)
    if task == "regression":
        np.testing.assert_allclose(session.predict(partition="test"),expected,atol=2e-10)
    else:
        np.testing.assert_array_equal(session.predict(partition="test"),expected)
    assert joblib.hash(session.fit_result) == frozen and session.split_plan is split
    pd.testing.assert_frame_equal(session.to_pandas(),before)


REGRESSION_PAIRS = [("mse","r2"),("r2","mae"),("rmse","mse"),("mae","r2")]


@pytest.mark.parametrize("primary,secondary",REGRESSION_PAIRS)
@pytest.mark.parametrize("selection",["cv","validation"])
@pytest.mark.parametrize("pruning",[False,True])
@pytest.mark.parametrize("storage",[False,True])
def test_regression_objectives_match_weighted_numerics_and_saved_winner(primary,secondary,selection,pruning,storage,tmp_path):
    session = _session(_frame(),tmp_path)
    result,uri = _run(session,tmp_path,"regression",primary,secondary,pruning,storage,selection=selection)
    if uri:
        study = optuna.load_study(study_name="buildml_automl",storage=uri)
        assert [d.name.lower() for d in study.directions] == ["maximize" if p == "r2" else "minimize" for p in [primary,secondary]]
        for trial in study.trials:
            expected,_ = _scores(session,"regression",selection,{"alpha":trial.params["param__ridge__alpha"]})
            np.testing.assert_allclose(trial.values,[expected[primary],expected[secondary]],atol=1e-10)
            assert trial.state == optuna.trial.TrialState.COMPLETE
    _bundle_consumer(session,tmp_path,"regression",result)


@pytest.mark.parametrize("primary,secondary",[("accuracy","f1_macro"),("balanced_accuracy","f1_weighted")])
@pytest.mark.parametrize("selection",["cv","validation"])
@pytest.mark.parametrize("pruning",[False,True])
@pytest.mark.parametrize("storage",[False,True])
@pytest.mark.parametrize("labels",["text","boolean"])
def test_classification_objectives_preserve_raw_labels_and_weighted_confusion(primary,secondary,selection,pruning,storage,labels,tmp_path):
    session = _session(_frame("classification",labels),tmp_path)
    result,uri = _run(session,tmp_path,"classification",primary,secondary,pruning,storage,selection=selection)
    if uri:
        study = optuna.load_study(study_name="buildml_automl",storage=uri)
        assert [d.name for d in study.directions] == ["MAXIMIZE","MAXIMIZE"]
        for trial in study.trials:
            params = {k.removeprefix("param__decision_tree__"):None if v == "None" else int(v)
                      for k,v in trial.params.items() if k.startswith("param__decision_tree__")}
            expected,_ = _scores(session,"classification",selection,params)
            np.testing.assert_allclose(trial.values,[expected[primary],expected[secondary]],atol=1e-10)
    _bundle_consumer(session,tmp_path,"classification",result)


@pytest.mark.parametrize("backing",BACKINGS)
def test_actual_persisted_multiobjective_study_appends_without_changing_prior_trials(backing,tmp_path):
    session = _session(_frame(),tmp_path,backing)
    try:
        first,uri = _run(session,tmp_path,"regression","mse","r2")
        study = optuna.load_study(study_name="buildml_automl",storage=uri)
        before = [(t.number,t.state,t.values,t.params) for t in study.trials]
        second,_ = _run(session,tmp_path,"regression","mse","r2")
        study = optuna.load_study(study_name="buildml_automl",storage=uri)
        assert [(t.number,t.state,t.values,t.params) for t in study.trials[:2]] == before
        assert len(study.trials) == 4 and all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        assert len(first.trials) == len(second.trials) == 2
        _bundle_consumer(session,tmp_path,"regression",second)
    finally:
        session.close_native()


@pytest.mark.parametrize("task",["regression","classification"])
@pytest.mark.parametrize("pruning",[False,True])
@pytest.mark.parametrize("route",["backend","method"])
def test_single_objective_pruning_and_routing_remain_valid(task,pruning,route,tmp_path):
    session = _session(_frame(task),tmp_path)
    primary = "mse" if task == "regression" else "accuracy"
    result,uri = _run(session,tmp_path,task,primary,None,pruning,multi=False,route=route)
    study = optuna.load_study(study_name="buildml_automl",storage=uri)
    assert len(study.directions) == 1 and all(len(t.values) == 1 for t in study.trials)
    _bundle_consumer(session,tmp_path,task,result)


@pytest.mark.parametrize("task",["regression","classification"])
@pytest.mark.parametrize("pruning",[False,True])
def test_cv_objectives_and_fitted_models_ignore_validation_and_test_outcomes(task,pruning,tmp_path):
    raw = _frame(task)
    changed = raw.copy()
    changed.iloc[64:,changed.columns.get_loc("y")] = 1e6 if task == "regression" else "holdout-only"
    first = _session(raw,tmp_path)
    second = _session(changed,tmp_path)
    primary,secondary = ("mse","r2") if task == "regression" else ("accuracy","f1_macro")
    a,_ = _run(first,tmp_path,task,primary,secondary,pruning,storage=False)
    b,_ = _run(second,tmp_path,task,primary,secondary,pruning,storage=False)
    assert [(t.params,t.mean_metrics) for t in a.trials] == [(t.params,t.mean_metrics) for t in b.trials]
    np.testing.assert_array_equal(first.predict(partition="test"),second.predict(partition="test"))
