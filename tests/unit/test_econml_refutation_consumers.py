"""Real EconML refutation adjusts confounding on train-only populations."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('econml')
from buildml import Session
from buildml.causal.fit import fit_causal
from buildml.causal.refute import refute_causal
from buildml.causal.types import CausalAssumptions
from buildml.data.splits import SplitPlan


@pytest.fixture(scope='module', params=['dml', 'causal_forest', 'policy_tree'])
def fitted(request):
    rng = np.random.default_rng(89)
    z = rng.normal(size=(700, 2))
    treatment = (rng.random(700) < 1/(1+np.exp(-.8*z[:, 0]))).astype(int)
    y = 2*treatment+3*z[:, 0]+.3*z[:, 1]+rng.normal(scale=.15, size=700)
    frame = pd.DataFrame({'z': z[:, 0], 'w': z[:, 1], 't': treatment, 'y': y}, index=[4]*700)
    # Poisoned holdouts would dominate any accidentally expanded refit population.
    frame.iloc[500:, 0] = 1e5
    frame.iloc[500:, 3] = -1e6
    session = Session.ingest(frame).set_roles({'z': 'feature', 'w': 'feature', 't': 'feature', 'y': 'target'})
    split = SplitPlan('injected', None, None, 0, None, tuple(range(500)), tuple(range(500,600)), tuple(range(600,700)))
    assumptions = CausalAssumptions(treatment='t', outcome='y', confounders=('z','w'),
        acknowledge_unconfoundedness=True, acknowledge_positivity=True)
    plan, _ = fit_causal(session.dataset, split, assumptions, backend='econml',
                         method=request.param, bootstrap_samples=0, random_state=11)
    assert abs(plan.ate-2) < .8
    train = frame.iloc[:500]
    unadjusted = train.loc[train.t == 1, 'y'].mean()-train.loc[train.t == 0, 'y'].mean()
    assert abs(unadjusted-2) > 1
    return session.dataset, split, plan, frame


@pytest.mark.parametrize('kind', ['random_confounder', 'placebo_treatment'])
@pytest.mark.parametrize('seed', [7, 19])
def test_refutation_uses_adjusted_train_effects_and_preserves_original_plan(fitted, kind, seed):
    dataset, split, plan, frame = fitted
    artifact = plan.backend_artifact_
    x_train = artifact['x_train'].copy()
    original = artifact['estimator']
    before = np.asarray(original.predict_value(x_train) if plan.method == 'policy_tree' else original.effect(x_train)).copy()
    result = refute_causal(dataset, plan, split, kind=kind, random_state=seed)
    target = 2 if kind == 'random_confounder' else 0
    assert abs(result.refute_ate-target) < .8
    assert result.ate_shift == pytest.approx(result.refute_ate-plan.ate)
    assert result.original_ate == plan.ate
    assert result.n_rows == 500
    assert result.method == plan.method
    assert any('not proof' in line for line in result.disclosures)
    np.testing.assert_array_equal(artifact['x_train'], x_train)
    after = np.asarray(original.predict_value(x_train) if plan.method == 'policy_tree' else original.effect(x_train))
    # Parallel forest reductions may differ by ~2e-15 across predictions.
    np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(dataset._ensure_pandas(), frame)
