"""Independent numerical checks for the redesigned passive-array comparisons."""
import numpy as np
from lib.array_geometry_study import (
    configuration,dataset,sample_covariance,covariance_shrinkage,ledoit_wolf,
    scm_statistics,scm_validation_loss,continuous_minimum,validation_task,
    select_policies,test_task as evaluate_task,fixture)
from lib.beamforming_numerics import complex_moments
from lib.beamforming_decisions import isotropized_mean
from lib.geometry import center
from benchmarks.case1.baselines import mvdr,output_power


def observations(seed=701,n=6,m=19):
    rng=np.random.default_rng(seed)
    return (rng.normal(size=(m,n))+1j*rng.normal(size=(m,n)))/np.sqrt(2)


def test_hermitian_ledoit_wolf_matches_direct_outer_product_formula():
    U=observations();m,n=U.shape
    S=sum(np.outer(u,u.conj()) for u in U)/m
    mu=np.trace(S).real/n
    phi=sum(np.linalg.norm(np.outer(u,u.conj())-S,'fro')**2 for u in U)/(m*m)
    delta=np.linalg.norm(S-mu*np.eye(n),'fro')**2
    alpha=min(1.,phi/delta)
    R,actual=ledoit_wolf(U)
    assert abs(actual-alpha)<1e-12
    assert np.allclose(R,(1-alpha)*S+alpha*mu*np.eye(n),atol=1e-12,rtol=0)
    assert np.linalg.eigvalsh(R)[0]>0


def test_singleton_covariance_convention_and_exact_infinite_loading_endpoint():
    U=observations(m=1);S=sample_covariance(U);n=U.shape[1]
    R,alpha=ledoit_wolf(U)
    assert alpha==1.
    assert np.allclose(R,np.trace(S).real/n*np.eye(n),atol=1e-13)
    a=observations(seed=702,m=1)[0];a/=np.linalg.norm(a)
    assert np.allclose(mvdr(covariance_shrinkage(S,1.),a),a,atol=1e-12)


def test_complex_scm_validation_formula_matches_direct_covariance_solves():
    rng=np.random.default_rng(703);n=6;a=observations(704,n=n,m=1)[0];a/=np.linalg.norm(a)
    rows=[];direct={alpha:[] for alpha in [.05,.3,.8,1.]}
    for i in range(7):
        U=observations(710+i,n=n,m=11);h=rng.normal(size=n)+1j*rng.normal(size=n);h/=np.linalg.norm(h)
        S=sample_covariance(U);R=.2*np.eye(n)+np.outer(h,h.conj())
        rows.append(scm_statistics(S,a,h))
        for alpha in direct:direct[alpha].append(output_power(mvdr(covariance_shrinkage(S,alpha),a),R))
    for alpha,values in direct.items():
        assert abs(scm_validation_loss(alpha,rows,.2,1)-np.mean(values))<2e-12


def test_singular_unloaded_covariance_is_not_a_successful_candidate():
    U=observations(m=1);n=U.shape[1];a=np.eye(n,dtype=complex)[0];h=np.eye(n,dtype=complex)[1]
    rows=[scm_statistics(sample_covariance(U),a,h)]
    assert not np.isfinite(scm_validation_loss(0.,rows,.2,1.))
    assert np.isfinite(scm_validation_loss(1.,rows,.2,1.))
    fit=continuous_minimum(lambda alpha:scm_validation_loss(alpha,rows,.2,1.))
    grid=np.linspace(.001,1,1001)
    assert fit['loss']<=min(scm_validation_loss(alpha,rows,.2,1.) for alpha in grid)+1e-7


def test_snapshots_use_disjoint_frozen_validation_and_test_streams():
    c=configuration('quick')
    x=dataset(c,8,'ula10',.2,0,'test');repeat=dataset(c,8,'ula10',.2,0,'test')
    validation=dataset(c,8,'ula10',.2,0,'validation')
    assert np.array_equal(x['U'],repeat['U'])
    assert not np.array_equal(x['U'],validation['U'])
    assert not np.array_equal(x['U'][:16],x['heldout'][:16])
    assert np.isclose(np.vdot(x['h'],x['h']),1)


def test_one_passive_snapshot_has_no_anisotropy_beyond_modal_overlap():
    for n in [8,16]:
        u=observations(seed=720+n,n=n,m=1)[0]
        A=center(np.outer(u,u.conj())/(.2*1.2));M=complex_moments(A).mean
        P=np.outer(u,u.conj())/np.vdot(u,u).real;Mi,_=isotropized_mean(M,P)
        assert np.linalg.norm(M-Mi,'fro')<1e-10


def test_controlled_fixture_recovers_passive_data_and_entire_radial_law():
    c=configuration('quick');c['fixture']['draws']=1024
    rows,arrays,checks=fixture(c)
    assert checks['same_mode'] and checks['same_spectrum']
    assert max(checks['radius_pair_error'],checks['predictive_mean_rotation_error'],checks['passive_selected_data_recovery_error'])<2e-10
    assert np.allclose(arrays['radius1'],arrays['radius2'],atol=2e-12)
    assert np.linalg.norm(arrays['M1']-arrays['M2'],'fro')>.01
    assert all(r['excess']>=-1e-12 for r in rows)


def test_small_end_to_end_validation_and_paired_decisions_keep_all_methods():
    c=configuration('quick');c['cohorts']=[dict(n=8,validation_trials=3,test_trials=2)]
    c['horizons']=[1,4];c['grouse_steps']=[.01,.1];c['bracket_points']=17;c['tail_trials']=0
    records=[r for trial in range(3) for r in validation_task((c,8,'ula10',.2,trial))]
    selected,curves=select_policies(records,c)
    result=evaluate_task((c,8,'ula10',.2,0,selected))
    assert len(result['metrics'])==16 and len(result['shape'])==2
    assert set(r['method'] for r in result['metrics'])=={'mean','mode','iso','scalar','scm','lw','grouse','oracle'}
    assert max(r['unit_gain_error'] for r in result['metrics'])<1e-10
    assert max(r['optimality_residual'] for r in result['metrics'])<1e-10
    assert max(r['pca_mode_error'] for r in result['shape'])<1e-10
    assert max(r['parameter_scatter_error'] for r in result['shape'])<1e-10


def test_continuous_validation_refines_optima_between_endpoint_and_first_grid_cell():
    # Regression for a coarse grid whose best sampled endpoint hides an
    # interior minimizer; both covariance families use this search.
    for optimum in [.003,.997]:
        fit=continuous_minimum(lambda x:(x-optimum)**2,65)
        assert abs(fit['value']-optimum)<1e-6
        assert fit['loss']<1e-12
