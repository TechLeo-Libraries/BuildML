"""DoWhy partition effects match independent OLS with distinct train/holdout effects."""
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.causal.estimate import estimate_causal
from buildml.causal.evaluate import evaluate_causal
from buildml.causal.fit import fit_causal
from buildml.causal.types import CausalAssumptions


@pytest.mark.parametrize('seed',[7,19])
@pytest.mark.parametrize('partition',['validation','test'])
@pytest.mark.parametrize('entrypoint',[estimate_causal,evaluate_causal])
def test_real_dowhy_partition_ate_is_not_train_ate(seed,partition,entrypoint):
    pytest.importorskip('dowhy')
    rng=np.random.default_rng(seed)
    n=500
    w=rng.normal(size=n)
    t=(rng.random(n)<1/(1+np.exp(-w))).astype(int)
    frame=pd.DataFrame({'w':w,'t':t,'y':2*t+3*w+rng.normal(scale=.05,size=n)})
    def session(data):
        return Session.ingest(data).set_roles({'w':'feature','t':'feature','y':'target'}).split(test_size=.25,validation_size=.25,random_state=3)
    s=session(frame)
    # Keep exact split positions and assign a different real effect to each holdout.
    frame.loc[list(s.split_plan.validation_indices),'y']+=4*frame.loc[list(s.split_plan.validation_indices),'t']
    frame.loc[list(s.split_plan.test_indices),'y']+=7*frame.loc[list(s.split_plan.test_indices),'t']
    s=session(frame)
    assumptions=CausalAssumptions(treatment='t',outcome='y',confounders=('w',),acknowledge_unconfoundedness=True,acknowledge_positivity=True)
    plan,_=fit_causal(s.dataset,s.split_plan,assumptions,backend='dowhy',method='backdoor_linear',random_state=11,bootstrap_samples=0)
    part=frame.iloc[list(getattr(s.split_plan,partition+'_indices'))]
    design=np.column_stack([np.ones(len(part)),part.t,part.w])
    expected=np.linalg.lstsq(design,part.y,rcond=None)[0][1]
    assert abs(expected-plan.ate)>3
    actual=entrypoint(s.dataset,plan,s.split_plan,partition=partition,bootstrap_samples=0)
    assert actual.ate==pytest.approx(expected,abs=1e-8)
    assert actual.n_rows==len(part)
    assert actual.ate_ci_low is None and actual.ate_ci_high is None
    assert any('fresh' in x.lower() and 'partition' in x.lower() for x in actual.disclosures)
    # Changing only train outcomes cannot alter the re-estimate on the holdout.
    s.dataset._ensure_pandas().iloc[list(s.split_plan.train_indices),s.to_pandas().columns.get_loc('y')]=1e6
    again=entrypoint(s.dataset,plan,s.split_plan,partition=partition,bootstrap_samples=0)
    assert again.ate==pytest.approx(expected,abs=1e-8)
