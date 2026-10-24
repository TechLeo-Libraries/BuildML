"""Real probabilistic consumers checked against density algebra and quadrature."""

import math
from statistics import NormalDist

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from packaging.version import Version
from scipy.integrate import quad

from buildml import Session
from buildml.data.dataset import Dataset
from buildml.probabilistic.evaluate import evaluate_probabilistic


@pytest.fixture(scope="module", params=["bayesian_ridge", "gaussian_process_regressor"])
def gaussian_session(request):
    rng = np.random.default_rng(984)
    x = rng.normal(size=(64, 2))
    frame = pd.DataFrame(
        {"x": x[:, 0], "z": x[:, 1], "y": 1.3 * x[:, 0] - 0.7 * x[:, 1] + rng.normal(0, 0.35, 64)}
    )
    s = Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "y": "target"})
    s.split(test_size=0.25, validation_size=0.25, random_state=8)
    s.probabilistic.fit(
        backend="native",
        estimator=request.param,
        task="regression",
        conformal=False,
        interval_method="posterior_std",
    )
    return s


def gaussian_oracle(session, x):
    plan = session.probabilistic_plan
    est = plan.estimator_
    train = session.partition("train")
    xt = train[list(plan.columns)].to_numpy()
    yt = train.y.to_numpy()
    if plan.estimator_name == "bayesian_ridge":
        centered = xt - xt.mean(axis=0)
        covariance = np.linalg.inv(est.lambda_ * np.eye(2) + est.alpha_ * centered.T @ centered)
        coef = est.alpha_ * covariance @ centered.T @ (yt - yt.mean())
        mean = (x - xt.mean(axis=0)) @ coef + yt.mean()
        # sklearn 1.9 centers predictive variance; <=1.8 exposes the offset
        # attribute too but evaluates variance on the raw design. Both mean
        # predictions still use the centered training posterior above.
        offset = xt.mean(axis=0) if Version(sklearn.__version__).release >= (1, 9) else np.zeros(2)
        design = x - offset
        variance = np.einsum("ij,jk,ik->i", design, covariance, design) + 1 / est.alpha_
    else:
        scale = float(est.kernel_.k1.length_scale)
        noise = float(est.kernel_.k2.noise_level)

        def rbf(a, b):
            return np.exp(-np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2) / (2 * scale**2))

        gram = rbf(xt, xt) + (noise + float(est.alpha)) * np.eye(len(xt))
        cross = rbf(x, xt)
        mean = cross @ np.linalg.solve(gram, (yt - yt.mean()) / yt.std()) * yt.std() + yt.mean()
        variance = (
            1 + noise - np.sum(cross * np.linalg.solve(gram, cross.T).T, axis=1)
        ) * yt.std() ** 2
    return mean, np.sqrt(variance)


@pytest.mark.parametrize("partition", ["validation", "test"])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("alpha", [0.05, 0.2, 0.4])
def test_gaussian_scores_bounds_and_bundle_match_independent_oracle(
    gaussian_session, partition, missing, alpha, tmp_path
):
    s = gaussian_session
    frame = s.to_pandas().copy()
    positions = getattr(s.split_plan, partition + "_indices")
    if missing:
        frame.loc[list(positions)[::3], "y"] = np.nan
    dataset = Dataset.from_pandas(frame, roles=s.dataset.roles)
    kept = frame.iloc[list(positions)].dropna(subset=["y"])
    x, y = kept[["x", "z"]].to_numpy(), kept.y.to_numpy()
    mean, std = gaussian_oracle(s, x)
    before = joblib.hash(s.probabilistic_plan)
    result = evaluate_probabilistic(
        dataset, s.probabilistic_plan, s.split_plan, partition=partition, alpha=alpha
    )
    z = NormalDist().inv_cdf(1 - alpha / 2)
    lower, upper = mean - z * std, mean + z * std
    errors = y - mean
    expected = {
        "mae": np.abs(errors).mean(),
        "rmse": np.sqrt((errors**2).mean()),
        "r2": 1 - (errors**2).sum() / ((y - y.mean()) ** 2).sum(),
        "nll": np.mean(np.log(std * math.sqrt(2 * math.pi)) + errors**2 / (2 * std**2)),
        "interval_coverage": np.mean((y >= lower) & (y <= upper)),
        "mean_interval_width": np.mean(upper - lower),
        "interval_score": np.mean(
            upper - lower + 2 / alpha * (np.maximum(lower - y, 0) + np.maximum(y - upper, 0))
        ),
    }
    # CRPS is the integral of squared CDF error, independently of its closed form.
    scores = []
    for actual, m, sd in zip(y, mean, std, strict=True):
        distribution = NormalDist(float(m), float(sd))
        left = quad(
            lambda t, distribution=distribution: distribution.cdf(t) ** 2,
            m - 12 * sd,
            actual,
            epsabs=1e-9,
        )[0]
        right = quad(
            lambda t, distribution=distribution: (1 - distribution.cdf(t)) ** 2,
            actual,
            m + 12 * sd,
            epsabs=1e-9,
        )[0]
        scores.append(left + right)
    expected["crps"] = np.mean(scores)
    assert result.n_rows == len(kept)
    for key, value in expected.items():
        assert result.metrics[key] == pytest.approx(value, rel=2e-6, abs=1e-8), key
    assert any("independence" in d for d in result.disclosures)
    path = tmp_path / "bundle"
    s.probabilistic.save_bundle(path)
    other = Session.ingest(frame).set_roles(s.dataset.roles)
    other.probabilistic.load_bundle(path, trusted=True)
    replay = evaluate_probabilistic(
        other.dataset, other.probabilistic_plan, s.split_plan, partition=partition, alpha=alpha
    )
    assert replay.metrics == pytest.approx(result.metrics)
    assert joblib.hash(s.probabilistic_plan) == before
    pd.testing.assert_frame_equal(dataset._ensure_pandas(), frame)


@pytest.fixture(
    scope="module",
    params=[("strings", 2), ("strings", 3), ("integers", 2), ("integers", 3), ("booleans", 2)],
)
def naive_bayes_session(request):
    kind, count = request.param
    rng = np.random.default_rng(488)
    labels = np.arange(150) % count
    x = rng.normal(size=(150, 2)) + labels[:, None] * 0.8
    vocabulary = {
        "strings": ["zebra", "apple", "middle"],
        "integers": [2, 10, 30],
        "booleans": [False, True],
    }[kind]
    frame = pd.DataFrame({"x": x[:, 0], "z": x[:, 1], "y": [vocabulary[i] for i in labels]})
    s = Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "y": "target"})
    s.split(test_size=0.2, validation_size=0.2, random_state=18)
    s.probabilistic.fit(
        backend="native", estimator="gaussian_nb", task="classification", conformal=False
    )
    return s


@pytest.mark.parametrize("partition", ["validation", "test"])
@pytest.mark.parametrize("missing", [False, True])
def test_classification_proper_scores_follow_density_columns_after_reload(
    naive_bayes_session, partition, missing, tmp_path
):
    s = naive_bayes_session
    plan = s.probabilistic_plan
    frame = s.to_pandas().copy()
    positions = list(getattr(s.split_plan, partition + "_indices"))
    if missing:
        frame["y"] = frame.y.astype(object)
        frame.loc[positions[::4], "y"] = None
    kept = frame.iloc[positions].dropna(subset=["y"])
    train = s.partition("train")
    xt = train[["x", "z"]].to_numpy()
    x = kept[["x", "z"]].to_numpy()
    classes = sorted(train.y.astype(str).unique())
    epsilon = 1e-9 * np.var(xt, axis=0).max()
    log_densities = []
    for label in classes:
        values = xt[train.y.astype(str).to_numpy() == label]
        variance = values.var(axis=0) + epsilon
        log_densities.append(
            np.log(len(values) / len(train))
            - 0.5
            * np.sum(
                np.log(2 * np.pi * variance) + (x - values.mean(axis=0)) ** 2 / variance, axis=1
            )
        )
    logs = np.column_stack(log_densities)
    probabilities = np.exp(logs - logs.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    actual = kept.y.astype(str).to_numpy()
    positions_y = [classes.index(label) for label in actual]
    expected_nll = -np.log(probabilities[np.arange(len(kept)), positions_y]).mean()
    before = joblib.hash(plan)
    result = evaluate_probabilistic(
        Dataset.from_pandas(frame, roles=s.dataset.roles), plan, s.split_plan, partition=partition
    )
    assert list(plan.classes_) == classes
    assert result.metrics["nll"] == pytest.approx(expected_nll)
    assert result.metrics["accuracy"] == pytest.approx(
        np.mean(np.array(classes)[probabilities.argmax(axis=1)] == actual)
    )
    if len(classes) == 2:
        p = probabilities[:, 1]
        outcomes = (actual == classes[1]).astype(float)
        assert result.metrics["brier"] == pytest.approx(np.mean((outcomes - p) ** 2))
        assignments = np.minimum((10 * p).astype(int), 9)
        ece = sum(
            abs(np.sum(outcomes[assignments == i] - p[assignments == i])) for i in range(10)
        ) / len(p)
        assert result.metrics["ece"] == pytest.approx(ece)
    else:
        assert "brier" not in result.metrics and "ece" not in result.metrics
    path = tmp_path / "bundle"
    s.probabilistic.save_bundle(path)
    other = Session.ingest(frame).set_roles(s.dataset.roles)
    other.probabilistic.load_bundle(path, trusted=True)
    replay = evaluate_probabilistic(
        other.dataset, other.probabilistic_plan, s.split_plan, partition=partition
    )
    assert replay.metrics == pytest.approx(result.metrics)
    assert joblib.hash(plan) == before


def test_catalog_lists_native_consumer_keys_and_keeps_returned_lists_independent():
    from buildml.probabilistic.catalog import backend_available, list_probabilistic_estimators

    native = {
        "bayesian_ridge",
        "gaussian_process_regressor",
        "gaussian_process_classifier",
        "gaussian_nb",
    }
    assert set(list_probabilistic_estimators(backend="native")) == native
    assert set(list_probabilistic_estimators(backend="mapie")) == {
        "split",
        "cv_plus",
        "jackknife_plus",
    }
    assert set(list_probabilistic_estimators(backend="ngboost")) == {
        "ngboost_regressor",
        "ngboost_classifier",
    }
    all_keys = list_probabilistic_estimators()
    assert len(all_keys) == 9 and native.issubset(all_keys)
    all_keys.clear()
    assert len(list_probabilistic_estimators()) == 9
    assert backend_available("native") is True
    assert list_probabilistic_estimators(backend="unsupported") == []
    assert backend_available("unsupported") is False
