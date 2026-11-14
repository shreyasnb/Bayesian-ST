from pathlib import Path
import json
import numpy as np
from numpy.testing import assert_allclose
from lib.final_beamforming import (beam_episode,decision_records,weighted_statistics,
    generate_stream,covariance_power,modal_weight)
from lib.models import increment
from lib.geometry import center

FIXTURES=Path(__file__).parent/'fixtures'


def test_beamforming_states_weights_and_rng_match_pinned_source():
    spec=json.loads((FIXTURES/'audited_beamforming.json').read_text())
    cfg,policy=spec['config'],spec['policy']
    expected=np.load(FIXTURES/'audited_beamforming.npz')
    stream=beam_episode(cfg,32,'abrupt',0)
    assert_allclose(stream['U'],expected['U'],atol=2e-13,rtol=2e-13)
    records=decision_records(stream['U'],stream['sigma2'],stream['tau2'],stream['a'],stream['initial'],policy,cfg,True)
    assert [r['t'] for r in records]==[8,16]
    for record in records:
        for key in ['A','M','P','M_iso']:
            assert_allclose(record[key],expected[f"t{record['t']}_{key}"],rtol=2e-11,atol=2e-11)
        for method,w in record['weights'].items():
            assert_allclose(w,expected[f"t{record['t']}_w_{method}"],atol=2e-11)


def test_known_zero_mean_statistic_and_proper_complex_increment():
    rng=np.random.default_rng(104)
    U=(rng.normal(size=(7,5))+1j*rng.normal(size=(7,5)))/np.sqrt(2)
    U+=1+.5j
    sigma=np.array([.2,.2,.7,.7,.4,.4,.4]);tau=np.array([1.,1.,1.,2.,2.,2.,.5]);beta=.83
    state=weighted_statistics(U,sigma,tau,beta,[7])[7]
    w=beta**np.arange(6,-1,-1)
    expected=sum(wj*q/(s*(s+q))*np.outer(u,u.conj()) for wj,u,s,q in zip(w,U,sigma,tau))
    assert_allclose(state['A'],center(expected),atol=2e-12)
    assert_allclose(state['S'],U.T@(w[:,None]*U.conj()),atol=2e-12)
    centered=(U-U.mean(axis=0)).T@(w[:,None]*(U-U.mean(axis=0)).conj())
    assert np.linalg.norm(state['S']-centered)>1
    assert_allclose(increment(U[0],sigma[0],tau[0],complex_data=True),tau[0]/(sigma[0]*(sigma[0]+tau[0]))*np.outer(U[0],U[0].conj()),atol=2e-12)


def test_stationary_horizon_is_200_and_last_observation_is_used():
    spec=json.loads((FIXTURES/'audited_beamforming.json').read_text())
    cfg=spec['config']
    for n in [32,64,128]:
        stream=beam_episode(cfg,n,'static',2,stationary=True)
        assert stream['U'].shape==(200,n)
        words=np.random.SeedSequence([cfg['test_seed'],n,0,2]).generate_state(4).tolist()
        explicit=generate_stream(n,200,'static',words)
        assert_allclose(stream['U'],explicit['U'],atol=0,rtol=0)
        A=weighted_statistics(stream['U'],stream['sigma2'],stream['tau2'],1.,[199,200])
        expected=center(A[199]['A']+increment(stream['U'][-1],.2,1.,complex_data=True))
        assert_allclose(A[200]['A'],expected,atol=2e-11)
        assert not np.allclose(A[200]['A'],A[199]['A'])


def test_proper_complex_power_matches_dense_covariance():
    rng=np.random.default_rng(103);n=8
    h=rng.normal(size=n)+1j*rng.normal(size=n);h/=np.linalg.norm(h)
    a=rng.normal(size=n)+1j*rng.normal(size=n);a/=np.linalg.norm(a)
    w=modal_weight(h,a,.2,1.)
    true=.2*np.eye(n)+np.outer(h,h.conj())
    assert_allclose(covariance_power(w,h,.2,1.),np.vdot(w,true@w).real,atol=2e-13)
    assert_allclose(np.vdot(w,a),1.,atol=2e-13)


def test_noise_adaptation_scores_actions_and_future_scales_match_pinned_source():
    from scripts.run_beamforming import primary_stream
    from lib.final_prediction import adaptation_trial
    spec=json.loads((FIXTURES/'audited_noise.json').read_text())
    stream=primary_stream(spec['config'],0)
    rows=adaptation_trial('beamforming',stream,spec['policies'],spec['config'],0,'test','noise_change')[0]
    index={(row['method'],row['control'],row['t']):row for row in rows}
    assert len(index)==len(spec['rows'])
    for original in spec['rows']:
        actual=index[(original['method'],original['control'],original['t'])]
        for field,value in original.items():
            if field not in ['method','control','t']:
                assert_allclose(actual[field],value,rtol=3e-11,atol=3e-11,err_msg=field)
