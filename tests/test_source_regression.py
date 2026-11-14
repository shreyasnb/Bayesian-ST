from pathlib import Path
from numpy.testing import assert_allclose
import json
import numpy as np
from lib.filter import BinghamFilter
from lib.geometry import center
from lib.block_em import BlockData,fit_block_em
from lib.causal_variance import CausalVarianceTracker
from lib.beamforming_numerics import complex_moments
from lib.contact_process_prediction import fast_moments
from benchmarks.common.great import GREAT


def fixture_values():
    rng=np.random.default_rng(73021)
    values={}
    for complex_data in (False,True):
        n=3
        U=rng.normal(size=(8,n))*.6
        if complex_data:U=(U+1j*rng.normal(size=U.shape)*.6)/np.sqrt(2)
        raw=rng.normal(size=(n,n))
        if complex_data:raw=raw+1j*rng.normal(size=(n,n))
        G=np.linalg.qr(raw)[0]
        f=BinghamFilter(n,1,.3,tau2=.8,beta=.91,complex_data=complex_data)
        tr=CausalVarianceTracker(n,1,.3,.8,beta=.91,block_length=4,em_iterations=2,complex_data=complex_data)
        states=[];priors=[];scores=[];scales=[]
        for u in U:
            priors.append(tr.predict_parameter(G))
            scores.append(tr.logscore(u,G=G)[0])
            tr.update(u,G=G);f.update(u,G=G)
            states.append(tr.A.copy());scales.append([tr.sigma2,tr.tau2])
        prefix='complex' if complex_data else 'real'
        values[prefix+'_filter_A']=f.A
        values[prefix+'_causal_A']=np.asarray(states)
        values[prefix+'_causal_prior']=np.asarray(priors)
        values[prefix+'_score']=np.asarray(scores)
        values[prefix+'_scales']=np.asarray(scales)
        fit=fit_block_em(BlockData(U,1,complex_data=complex_data,boundary_prior=G@np.diag([.3,-.1,-.2])@G.conj().T),.7,.4,iterations=3)
        values[prefix+'_em_A']=fit.posterior.A
        values[prefix+'_em_mean']=fit.posterior.mean
        values[prefix+'_em_scale_objective']=np.asarray([fit.s,fit.tau2,*fit.objectives])
        great=GREAT(n,1,stepsize=.01,inner_steps=3,window=4,complex_data=complex_data,initial_basis=G[:,:1])
        for u in U:great.update(u,G=G)
        values[prefix+'_great_covariance']=great.covariance
        values[prefix+'_great_projector']=great.projector
    for name,A in [('repeated',np.diag([0.,6.,6.])),('close',np.diag([0.,6.,6.+1e-8])),('large',np.diag([0.,2.,200.]))]:
        result=complex_moments(A)
        values[name+'_complex_mean']=result.mean
        values[name+'_complex_logz']=np.asarray(result.log_normalizer)
    for name,A in [('broad',np.diag([0.,2.,6.])),('ridge',np.diag([0.,0.,80000.]))]:
        result=fast_moments(A,checked=True)
        values[name+'_contact_mean']=result.mean
        values[name+'_contact_logz']=np.asarray(result.log_normalizer)
    return values



def test_fixed_seed_parameters_moments_scales_scores_and_great_match_audited_source():
    fixture=json.loads((Path(__file__).parent/'fixtures/audited_core.json').read_text())
    assert fixture['source_commit']=='ec998a933fd693c2f5bdb62c6bf5654e9345853a'
    actual=fixture_values()
    assert set(actual)==set(fixture['values'])
    for key,value in actual.items():
        stored=fixture['values'][key]
        reference=np.asarray(stored['real'])+1j*np.asarray(stored['imag'])
        assert_allclose(value,reference,rtol=2e-11,atol=2e-11,err_msg=key)
