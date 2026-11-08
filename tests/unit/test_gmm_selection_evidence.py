"""GMM model selection uses the requested information criterion on train."""

import numpy as np
import pandas as pd
import pytest
from sklearn.mixture import GaussianMixture

from buildml import Session
from buildml.core.errors import ValidationError


@pytest.fixture
def populations():
    rng = np.random.default_rng(1)
    train = np.r_[rng.normal(-3, 1, 36), rng.normal(2, 0.7, 36), rng.normal(3.6, 0.32, 8)]
    holdout = np.array([-35, -4, -2, 0, 1, 2, 4, 35], dtype=float)
    session = (
        Session.ingest(pd.DataFrame({"x": np.r_[train, holdout]}))
        .set_roles({"x": "feature"})
        .inject_split(
            train_indices=list(range(len(train))),
            test_indices=list(range(len(train), len(train) + len(holdout))),
        )
    )
    return session, train, holdout


@pytest.mark.parametrize(("criterion", "expected_k"), [("bic", 2), ("aic", 4)])
def test_gmm_criterion_selects_train_model_and_preserves_holdout_assignment(
    populations, criterion, expected_k
):
    session, train, holdout = populations
    models = {}
    scores = {}
    for k in range(2, 5):
        model = GaussianMixture(n_components=k, random_state=13, n_init=1).fit(train[:, None])
        models[k] = model
        # A one-dimensional full-covariance mixture has k means, k variances
        # and k-1 free mixture weights.
        penalty = 2 if criterion == "aic" else np.log(len(train))
        scores[str(k)] = -2 * model.score_samples(train[:, None]).sum() + penalty * (3 * k - 1)
    assert int(min(scores, key=scores.__getitem__)) == expected_k

    result = session.unsupervised.fit(
        method="gmm", auto_k=True, auto_k_max=4, gmm_select_by=criterion, random_state=13
    )
    assert result.n_clusters == expected_k
    key = f"gmm_{criterion}"
    assert result.diagnostics[key] == pytest.approx(scores)
    assert result.diagnostics["gmm_selected_k"] == expected_k
    assert any(f"by {criterion.upper()}" in disclosure for disclosure in result.disclosures)

    model = models[expected_k]
    means = model.means_[:, 0]
    variances = model.covariances_[:, 0, 0]
    log_density = np.log(model.weights_) - 0.5 * (
        np.log(2 * np.pi * variances) + (holdout[:, None] - means) ** 2 / variances
    )
    assigned = session.unsupervised.assign(partition="test")
    np.testing.assert_array_equal(assigned.labels, log_density.argmax(axis=1))
    evaluated = session.unsupervised.evaluate(partition="test")
    assert evaluated.diagnostics[key] == pytest.approx(scores)
    assert evaluated.diagnostics["gmm_selected_k"] == expected_k


def test_invalid_gmm_criterion_preserves_prior_fit(populations):
    session, _, _ = populations
    prior_fit = session.unsupervised.fit(method="gmm", n_clusters=2, random_state=13)
    prior_plan = session.cluster_plan
    history = list(session.history)
    with pytest.raises(ValidationError, match="gmm_select_by"):
        session.unsupervised.fit(method="gmm", auto_k=True, gmm_select_by="likelihood")
    assert session.cluster_plan is prior_plan
    assert session.cluster_fit_result is prior_fit
    assert session.history == history
