"""End-to-end protocol checks; these small budgets are not accuracy studies."""
import numpy as np
import pytest
from lib.experiments import run_case1,run_case2,_field_data,_array_data
from lib.geometry import projector


def test_generators_follow_dimensions_and_distinguish_protocols():
    U,X,a,R,G=_array_data(10,5,.003,.25,'Matched')
    assert U.shape==(5,16) and X.shape==(5,16,3)
    for x,r,desired in zip(X,R,a):
        np.testing.assert_allclose(x.conj().T@x,np.eye(3),atol=1e-12)
        np.testing.assert_allclose(r,.3*np.eye(16)+2*projector(x),atol=1e-12)
        assert abs(np.linalg.norm(desired)-1)<1e-12
    Phi,trajectory,G=_field_data(9,4,conditional=True)
    np.testing.assert_allclose(Phi.T@Phi,np.eye(8),atol=1e-12)
    for x,u,clean,v,w in trajectory:
        np.testing.assert_allclose(clean,projector(x)@v)
        np.testing.assert_allclose(u,Phi.T@w)


def test_case1_runs_and_certifies_bounds():
    out=run_case1(seed=8,repeats=1,T=12,draws=60,burnin=40,max_draws=60,max_burnin=40)
    m,r,t=out['metrics'],out['regions'],out['tuning']
    assert len(m)==7*3*4 and len(r)==7*3*2
    assert np.isfinite(m.output_power).all() and (m.output_power>0).all()
    assert (m.ew_modal_distance<1e-6).all()
    assert r.feasible.all() and (r.upper>=r.lower-1e-8).all()
    assert (r.optimization_gap<1e-5).all()
    assert (t.groupby(['scenario','method']).size()==3).all()
    oracle=m[m.method=='Oracle'].set_index(['scenario','repeat','t']).output_power
    for name,g in m[m.method!='Oracle'].groupby('method'):
        indexed=g.set_index(['scenario','repeat','t']).output_power
        assert (indexed>=oracle-1e-10).all()
    assert out['config']['coverage_guaranteed'] is False


def test_case2_runs_and_preserves_separate_noise_study():
    out=run_case2(seed=11,repeats=1,T=12,draws=60,burnin=40,max_draws=60,max_burnin=40)
    m,c,n,t=out['metrics'],out['calibration'],out['noise'],out['tuning']
    assert len(m)==3*3*6 and len(c)==3*3
    assert np.isfinite(m.field_squared_error).all()
    assert (c.modal_excess_risk>=0).all()
    assert not n.numerical_ascent_verified.any()
    assert set(n.protocol)=={'Passive','Calibration'}
    assert (n.s>0).all() and n[n.protocol=='Passive'].tau2.ge(0).all()
    assert (t.groupby(['scenario','method']).size()==3).all()
    calibration=n[n.protocol=='Calibration']
    np.testing.assert_allclose(calibration.expected_residual_variance,calibration.naive_residual_variance+calibration.uncertainty_variance)
    assert np.isfinite(m[m.method=='Posterior mean'].predictive_log_score).all()
    assert m[m.method=='Oracle'].predictive_log_score.isna().all()
    assert out['fields'].shape==(64,7)
