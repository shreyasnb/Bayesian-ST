"""Native-complex beamforming controls, not performance-ranking assertions."""
import numpy as np
from numpy.testing import assert_allclose

from lib.beamforming_decisions import (isotropized_mean,conditional_excess,
    scalar_statistics,scalar_validation_loss,optimize_scalar,weighted_paths,
    default_config,trajectory,choose_policies)
from lib.beamforming_numerics import complex_moments
from lib.beamforming_protocol import proper_complex,haar_line
from lib.models import increment
from lib.geometry import center
from benchmarks.case1.baselines import mvdr,output_power


def test_complex_posterior_expected_power_optimality_for_arbitrary_feasible_weights():
    rng=np.random.default_rng(6619);n=8;a=haar_line(rng,n)
    Z=proper_complex(rng,(n,n));R=Z@Z.conj().T+.3*np.eye(n)
    optimal=mvdr(R,a)
    for _ in range(20):
        v=proper_complex(rng,n);w=optimal+v-a*np.vdot(a,v)
        left,right,error=conditional_excess(w,optimal,R)
        assert right>=0 and left>=-1e-12 and error<1e-10
        assert_allclose(np.vdot(w,a),1,atol=1e-12)


def test_one_complex_snapshot_has_exact_matched_isotropic_mean_and_decision():
    rng=np.random.default_rng(27715)
    for n in [2,4,8]:
        u=proper_complex(rng,n);A=center(increment(u,.2,1.,complex_data=True))
        h=u/np.linalg.norm(u);P=np.outer(h,h.conj());M=complex_moments(A).mean
        matched,rho=isotropized_mean(M,P);a=haar_line(rng,n)
        assert_allclose(M,matched,atol=2e-10)
        assert_allclose(mvdr(.2*np.eye(n)+M,a),mvdr(.2*np.eye(n)+matched,a),atol=2e-10)
        assert 0<=rho<=1
    # More than one observation need not be axially isotropic.
    U=proper_complex(rng,(4,8));A=center(5*U.T@U.conj());h=np.linalg.eigh(A)[1][:,-1]
    M=complex_moments(A).mean;P=np.outer(h,h.conj());matched,_=isotropized_mean(M,P)
    assert np.linalg.norm(M-matched)>.001
    assert_allclose(np.trace(P@M),np.trace(P@matched),atol=1e-12)


def test_scalar_covariance_loss_matches_independent_mvdr_and_continuous_scan():
    rng=np.random.default_rng(91227);n=8;s=.2;q=1.;a=haar_line(rng,n);statistics=[];records=[]
    for _ in range(12):
        h=haar_line(rng,n);m=haar_line(rng,n);P=np.outer(m,m.conj());R=s*np.eye(n)+q*np.outer(h,h.conj())
        statistics.append(scalar_statistics(P,a,R));records.append((P,R))
    for rho in [0.,.2,.8,1.]:
        direct=np.mean([output_power(mvdr(s*np.eye(n)+q*(rho*P+(1-rho)*np.eye(n)/n),a),R) for P,R in records])
        assert_allclose(scalar_validation_loss(rho,statistics,s,q,n),direct,atol=2e-12)
    fit=optimize_scalar(statistics,s,q,n)
    grid=[scalar_validation_loss(r,statistics,s,q,n) for r in np.linspace(0,1,1001)]
    assert fit['loss']<=min(grid)+1e-8


def test_passive_scatter_is_same_complex_bingham_mode_and_truth_is_not_an_input():
    rng=np.random.default_rng(5188);U=proper_complex(rng,(24,8));s=.2;q=1.;beta=.95
    path=weighted_paths(U,[beta])[beta];A=np.zeros((8,8),complex)
    for j,u in enumerate(U):
        A=center(beta*A+increment(u,s,q,complex_data=True))
        v=np.linalg.eigh(A)[1][:,-1];P=np.outer(v,v.conj())
        assert_allclose(path['mode'][j],P,atol=1e-12)
        assert_allclose(A,center(q/(s*(s+q))*path['scatter'][j]),atol=1e-12)
    altered=U.copy();altered[13]+=10
    other=weighted_paths(altered,[beta])[beta]
    assert_allclose(path['mode'][:13],other['mode'][:13],atol=0)
    assert np.linalg.norm(path['mode'][13:]-other['mode'][13:])>.1


def test_validation_selection_rejects_missing_trajectory_and_has_no_test_data():
    c=default_config('quick');c['validation_trials']=2
    rows=[]
    for beta,loss in [(1.,1.2),(.9,1.)]:
        for trial in range(2):rows.append(dict(condition='ula_moving',s=.2,t=8,family='mean',beta=beta,
            trial=trial,loss=loss,mean_ok=not(beta==.9 and trial==1)))
    selected,_=choose_policies(rows,c)
    assert selected[0]['beta']==1.
    data=trajectory(c,'haar',.2,0,'test');validation=trajectory(c,'haar',.2,0,'validation')
    assert np.linalg.norm(data['U']-validation['U'])>1
