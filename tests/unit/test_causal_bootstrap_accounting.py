"""Bootstrap summaries describe performed work and match independent arithmetic."""
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.causal.estimate import estimate_causal
from buildml.causal.fit import fit_causal
from buildml.causal.types import CausalAssumptions
from buildml.core.errors import ValidationError


@pytest.fixture(scope='module')
def native():
    rng=np.random.default_rng(72)
    w=rng.normal(size=200)
    t=(rng.random(200)<.5).astype(int)
    frame=pd.DataFrame({'w':w,'t':t,'y':2*t+w+rng.normal(scale=.2,size=200)})
    s=Session.ingest(frame).set_roles({'w':'feature','t':'feature','y':'target'}).split(test_size=.25,random_state=9)
    assumptions=CausalAssumptions(treatment='t',outcome='y',confounders=('w',),acknowledge_unconfoundedness=True,acknowledge_positivity=True)
    plan,_=fit_causal(s.dataset,s.split_plan,assumptions,backend='native',method='t_learner',bootstrap_samples=0)
    return s,plan,assumptions


@pytest.mark.parametrize('count',[0,1,20])
def test_native_bootstrap_matches_numpy_resampled_fixed_effects(native,count):
    s,plan,_=native
    result=estimate_causal(s.dataset,plan,s.split_plan,partition='test',bootstrap_samples=count,random_state=17)
    assert result.bootstrap_samples==count
    if not count:
        assert result.ate_ci_low is None
        return
    frame=s.partition('test')
    x=frame[['w']].to_numpy()
    effects=plan.mu1_.predict(x)-plan.mu0_.predict(x)
    indices=np.random.default_rng(17).integers(0,len(frame),size=(count,len(frame)))
    values=effects[indices].mean(axis=1)
    assert result.ate_ci_low==pytest.approx(np.quantile(values,.025))
    assert result.ate_ci_high==pytest.approx(np.quantile(values,.975))
    if count>1:
        assert result.ate_std==pytest.approx(values.std(ddof=1))


@pytest.mark.parametrize('count',[-1,True,1.5,'3',np.nan,np.inf])
def test_invalid_bootstrap_controls_refused(native,count):
    s,plan,_=native
    with pytest.raises(ValidationError,match='bootstrap_samples.*non-negative integer'):
        estimate_causal(s.dataset,plan,s.split_plan,bootstrap_samples=count)


@pytest.mark.parametrize('backend,method',[('dowhy','backdoor_linear'),('econml','dml')])
def test_unperformed_backend_bootstraps_are_not_advertised(native,backend,method):
    pytest.importorskip(backend)
    s,_,assumptions=native
    plan,_=fit_causal(s.dataset,s.split_plan,assumptions,backend=backend,method=method,bootstrap_samples=0)
    result=estimate_causal(s.dataset,plan,s.split_plan,partition='test',bootstrap_samples=3)
    assert result.bootstrap_samples==0
    assert result.ate_ci_low is None and result.ate_ci_high is None
    assert any('requested=3' in w and 'not performed' in w for w in result.warnings)
