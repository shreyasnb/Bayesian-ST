import numpy as np
import pandas as pd

from lib.final_beamforming import (
    configuration, matched_overlap_angle, generate_stream, modal_weight,
    covariance_power, weighted_statistics, decision_records, episode_average,
    paired_summary, block_episode, beam_episode, evaluate_episode, great_grid)
from lib.beamforming_protocol import ula, haar_line, proper_complex
from lib.geometry import center
from lib.array_geometry_study import distortionless_weight
from lib.final_beamforming import checked_moments
from lib.final_beamforming import great_likelihood_observations
from lib.beamforming_numerics import complex_moments


def _policy(n=4):
    return dict(n=n, condition='static', stationary=True, times=[1,4,8],
        primary_times=[1,4,8], shared_beta=1.,
        policies=[dict(times=[1,4,8],shared_beta=1.,scalar_beta=1.,rho=.4,
                       covariance_beta=1.,alpha=.3)],
        great_window=dict(window=4,beta=None,stepsize=.01,inner_steps=1),
        great_ew=dict(window=None,beta=1.,stepsize=.01,inner_steps=1),grouse_step=.01)


def _stationary_policy(n=4):
    policy=_policy(n)
    policy['times']=[25,50,100,200]
    policy['primary_times']=policy['times']
    policy['primary_evaluation_time']=200
    policy['policies'][0]['times']=policy['times']
    return policy


def test_stationary_stream_horizon_and_checkpoints_are_absolute_in_every_dimension():
    for stage in ('verify','pilot','paper'):
        cfg=configuration(stage)
        assert cfg['stationary_T']==200
        assert cfg['stationary_times']==[25,50,100,200]
        assert cfg['stationary_primary_t']==200
        assert cfg['story']['stationary_n']==64
        assert cfg['story']['stationary_time']==200
    cfg=configuration('verify')
    for n in cfg['dimensions']:
        stream=beam_episode(cfg,n,'static',2,stationary=True)
        explicit=beam_episode(cfg,n,'static',2,stationary=True,length=200)
        assert stream['U'].shape==(200,n)
        assert stream['h'].shape==(200,n)
        assert stream['time'].shape==(200,)
        assert np.array_equal(stream['U'],explicit['U'])
        assert np.array_equal(stream['sigma2'],np.full(200,.2))
        assert np.array_equal(stream['tau2'],np.ones(200))
        assert np.allclose(stream['h'],stream['h'][0])
        # The dimension stays in the predeclared seed channel; fixing T does
        # not replace the independent training stream with a shared one.
        words=np.random.SeedSequence([cfg['test_seed'],n,0,2]).generate_state(4).tolist()
        independently_seeded=generate_stream(n,200,'static',words)
        assert np.array_equal(stream['U'],independently_seeded['U'])


def test_fixed_stationary_horizon_preserves_moving_streams_and_great_window_scales():
    cfg=configuration('verify')
    for condition in cfg['moving_conditions']:
        stream=beam_episode(cfg,8,condition,1,stationary=False)
        explicit=beam_episode(cfg,8,condition,1,stationary=False,length=cfg['moving_T'])
        assert stream['U'].shape==(cfg['moving_T'],8)
        assert np.array_equal(stream['U'],explicit['U'])
    altered=dict(cfg,stationary_T=400,stationary_times=[50,100,200,400])
    for n in cfg['dimensions']:
        for beta in cfg['betas']:
            for exponential in (False,True):
                grid=great_grid(cfg,n,.2,1.,beta,exponential)
                assert grid==great_grid(altered,n,.2,1.,beta,exponential)
                if not exponential:
                    assert sorted({item['window'] for item in grid})==[n//2,n,2*n]


def test_stationary_final_state_includes_measurement_200_and_keeps_earlier_states_causal():
    cfg=configuration('verify');policy=_stationary_policy()
    stream=beam_episode(cfg,4,'static',0,stationary=True)
    U=stream['U'];s=stream['sigma2'];q=stream['tau2']
    records=decision_records(U,s,q,stream['a'],stream['initial'],policy,cfg,True)
    assert [record['t'] for record in records]==[25,50,100,200]
    eta=q/(s*(s+q))
    expected=center(sum(coefficient*np.outer(u,u.conj()) for coefficient,u in zip(eta,U)))
    assert np.allclose(records[-1]['A'],expected,rtol=2e-13,atol=2e-11)
    before_final=center(sum(coefficient*np.outer(u,u.conj()) for coefficient,u in zip(eta[:-1],U[:-1])))
    assert not np.allclose(records[-1]['A'],before_final)
    changed=U.copy();changed[-1]=np.array([8.,-3j,2.+1j,-4.])
    other=decision_records(changed,s,q,stream['a'],stream['initial'],policy,cfg,True)
    for first,second in zip(records[:-1],other[:-1]):
        assert np.array_equal(first['A'],second['A'])
        for method in first['weights']:
            assert np.allclose(first['weights'][method],second['weights'][method],atol=1e-12)
    assert not np.allclose(records[-1]['A'],other[-1]['A'])
    assert not np.allclose(records[-1]['P'],other[-1]['P'])


def test_stationary_story_uses_final_corrected_state_and_holdouts_only_score(monkeypatch):
    import lib.final_beamforming as beam
    cfg=configuration('verify');cfg['story']=dict(cfg['story'],stationary_n=4)
    policy=_stationary_policy()
    calls=[]
    original=beam.decision_records
    def record_decisions(*args,**kwargs):
        records=original(*args,**kwargs)
        calls.append(records)
        return records
    monkeypatch.setattr(beam,'decision_records',record_decisions)
    def holdout(scale):
        def samples(stream,t,draws):
            assert calls and calls[-1][-1]['t']==200
            return scale*np.ones((draws,stream['U'].shape[1]),complex)
        return samples
    monkeypatch.setattr(beam,'heldout_snapshots',holdout(1.))
    rows1,story1=evaluate_episode(cfg,policy,0,stationary=True,save_story=True)
    monkeypatch.setattr(beam,'heldout_snapshots',holdout(10.))
    rows2,story2=evaluate_episode(cfg,policy,0,stationary=True,save_story=True)
    frame1,frame2=pd.DataFrame(rows1),pd.DataFrame(rows2)
    assert np.allclose(frame1.true_power,frame2.true_power)
    assert np.allclose(frame1.heldout_power*100,frame2.heldout_power)
    assert story1['setup_U'].shape==(200,4)
    assert np.array_equal(story1['setup_A'],story1['static_n4_t200_A'])
    assert np.array_equal(story1['setup_M'],story1['static_n4_t200_M'])
    assert not np.allclose(story1['setup_A'],story1['static_n4_t100_A'])
    for key in ('setup_A','setup_M','setup_weights','setup_eigenvalues'):
        assert np.allclose(story1[key],story2[key])


def test_dimension_sweep_controls_physical_overlap_and_power_convention():
    for n in (8,32,64,128):
        angle=matched_overlap_angle(n,.5)
        a,h=ula(n,0.),ula(n,angle)
        assert abs(abs(np.vdot(a,h))**2-.5)<2e-12
        assert np.allclose(np.vdot(h,h),1.)
        # The total rank-one interference power is tau2, not n*tau2.
        assert np.allclose(np.trace(1.7*np.outer(h,h.conj())),1.7)


def test_rank_one_actions_match_dense_solve_and_include_isotropic_endpoint():
    rng=np.random.default_rng(103)
    n=8;h=haar_line(rng,n);a=haar_line(rng,n)
    P=np.outer(h,h.conj())
    for rho in (0.,.03,.6,1.):
        M=rho*P+(1-rho)*np.eye(n)/n
        R=.2*np.eye(n)+1.4*M
        w=modal_weight(h,a,.2,1.4,rho)
        assert np.allclose(w,distortionless_weight(R,a),atol=2e-13)
        assert abs(np.vdot(w,a)-1)<1e-13
    assert np.allclose(modal_weight(h,a,.2,1.4,0.),a)


def test_singleton_statistics_use_actual_varying_coefficients_without_covariance_division():
    rng=np.random.default_rng(104)
    U=proper_complex(rng,(7,5))
    sigma=np.array([.2,.2,.7,.7,.4,.4,.4])
    tau=np.array([1.,1.,1.,2.,2.,2.,.5])
    beta=.83
    state=weighted_statistics(U,sigma,tau,beta,[7])[7]
    w=beta**np.arange(6,-1,-1)
    independent=sum(wj*q/(s*(s+q))*np.outer(u,u.conj())
                    for wj,u,s,q in zip(w,U,sigma,tau))
    assert np.allclose(state['A'],center(independent),atol=2e-13)
    assert np.allclose(state['S'],U.T@(w[:,None]*U.conj()))
    assert np.allclose(state['mass'],w.sum())
    assert not np.allclose(state['A'],center(independent/len(U)))


def test_estimates_at_a_checkpoint_do_not_use_future_snapshots():
    cfg=configuration('verify');policy=_policy()
    rng=np.random.default_rng(105)
    U=proper_complex(rng,(8,4));initial=haar_line(rng,4);a=ula(4,0.)
    s=np.full(8,.2);q=np.ones(8)
    other=U.copy();other[4:]=100*proper_complex(rng,(4,4))
    first=decision_records(U,s,q,a,initial,policy,cfg)[1]
    second=decision_records(other,s,q,a,initial,policy,cfg)[1]
    later_s=s.copy();later_q=q.copy();later_s[4:]=.9;later_q[4:]=4.
    third=decision_records(U,later_s,later_q,a,initial,policy,cfg)[1]
    for method in first['weights']:
        assert np.allclose(first['weights'][method],second['weights'][method],atol=1e-12)
        assert np.allclose(first['weights'][method],third['weights'][method],atol=1e-12)


def test_one_marginal_observation_has_exact_axial_mean_control():
    cfg=configuration('verify');policy=_policy()
    rng=np.random.default_rng(106)
    U=proper_complex(rng,(8,4));a=ula(4,0.)
    record=decision_records(U,np.full(8,.2),np.ones(8),a,haar_line(rng,4),policy,cfg,True)[0]
    assert np.allclose(record['M'],record['M_iso'],atol=2e-10)
    assert np.allclose(record['weights']['mean'],record['weights']['isotropic'],atol=2e-10)


def test_declared_change_controls_are_separate_and_keep_training_interference_only():
    arguments=dict(n=8,length=16,seed=107,sigma2=.2,tau2=1.,overlap=.5)
    static=generate_stream(condition='static',**arguments)
    noise=generate_stream(condition='noise_change',**arguments)
    power=generate_stream(condition='power_change',**arguments)
    assert np.array_equal(static['h'],noise['h'])
    assert np.array_equal(static['h'],power['h'])
    assert np.array_equal(noise['tau2'],static['tau2'])
    assert np.array_equal(power['sigma2'],static['sigma2'])
    assert np.all(noise['sigma2'][8:]==.8)
    assert np.all(power['tau2'][8:]==4.)
    assert np.array_equal(noise['U'][:8],static['U'][:8])
    assert np.array_equal(power['U'][:8],static['U'][:8])


def test_episode_is_averaged_before_log_power_and_confidence_interval():
    rows=[]
    for t,ratio in [(8,100.),(16,1.),(24,9.)]:
        rows.append(dict(n=8,condition='smooth',trial=0,method='mean',t=t,
            heldout_ratio=ratio,true_ratio=ratio,response_map_error=.3,numerical_ok=True,mean_ok=True))
    result=episode_average(rows,16)[0]
    assert result['checkpoints']==2
    assert result['heldout_ratio']==5.
    assert np.isclose(result['heldout_ratio_db'],10*np.log10(5.))


def test_unresolved_pairs_remain_in_paired_diagnostic_denominator():
    rows=[]
    for trial in range(3):
        rows.append(dict(condition='static',trial=trial,method='mean',
            heldout_ratio_db=1.,true_ratio_db=1.,numerical_ok=True))
        rows.append(dict(condition='static',trial=trial,method='mode',
            heldout_ratio_db=np.nan if trial==2 else 2.,
            true_ratio_db=2.,numerical_ok=trial!=2))
    pairs=paired_summary(pd.DataFrame(rows),['condition'])
    row=pairs[pairs.endpoint.eq('heldout_ratio_db')].iloc[0]
    assert row.episodes==3 and row.resolved_pairs==2
    assert not row.numerical_ok
    assert row.gain==1.


def test_block_cadence_assimilates_once_and_never_reports_before_endpoint():
    cfg=configuration('verify')
    cfg.update(block_n=4,block_T=32,block_lengths=[1,4,16,32],heldout_snapshots=8)
    rows=block_episode(cfg,'smooth',0)
    exact=[row for row in rows if row['kind']=='delayed_exact']
    for row in exact:
        L=row['block_length']
        assert row['assimilated_observations']==32
        assert row['outputs']==32//L
        assert row['available_from_observation']==L
        assert row['maximum_parameter_error']<2e-11
        assert row['maximum_delay_observations']==L-1


def test_regular_cap_matches_safe_logarithmic_branch_at_moderate_high_dimension():
    A=np.diag(np.linspace(-80.,80.,128))
    regular,extended=checked_moments(A,tol=1e-10,max_order=4096,fallback_max_order=8192)
    logarithmic=complex_moments(A,tol=1e-10,max_order=8192)
    assert not extended
    assert regular.numerical_ok and logarithmic.numerical_ok
    assert np.linalg.norm(regular.mean-logarithmic.mean,2)<2e-9
    assert abs(regular.log_normalizer-logarithmic.log_normalizer)<2e-9


def test_matched_great_ew_uses_each_actual_variance_coefficient():
    from benchmarks.common.great import GREAT
    rng=np.random.default_rng(108)
    U=proper_complex(rng,(7,5));sigma=np.array([.2,.2,.8,.8,.4,.4,.4])
    tau=np.array([1.,1.,1.,4.,4.,4.,.5]);beta=.83
    Y=great_likelihood_observations(U,sigma,tau)
    tracker=GREAT(5,1,beta=beta,stepsize=.001,inner_steps=1,
                  complex_data=True,initial_basis=haar_line(rng,5)[:,None])
    for y in Y:tracker.update(y)
    state=weighted_statistics(U,sigma,tau,beta,[len(U)])[len(U)]
    eta0=tau[0]/(sigma[0]*(sigma[0]+tau[0]))
    assert np.allclose(center(eta0*tracker.covariance),state['A'],atol=2e-13)
    assert np.array_equal(great_likelihood_observations(U,.2,1.),U)


def test_observed_covariance_loading_matches_direct_distortionless_power():
    from lib.final_beamforming import (covariance_validation_statistics,
        covariance_validation_loss)
    rng=np.random.default_rng(109);n=7
    a=haar_line(rng,n);statistics=[];fixtures=[]
    for training_count in (3,11):
        U=proper_complex(rng,(training_count,n));Z=proper_complex(rng,(19,n))
        S=U.T@U.conj()/len(U);R=Z.T@Z.conj()/len(Z)
        statistics.append(covariance_validation_statistics(S,a,R))
        fixtures.append((S,R))
    for alpha in (.001,.17,.85,1.):
        direct=[]
        for S,R in fixtures:
            C=(1-alpha)*S+alpha*np.trace(S).real/n*np.eye(n)
            w=distortionless_weight(C,a)
            direct.append(np.vdot(w,R@w).real)
        assert np.isclose(covariance_validation_loss(alpha,statistics),np.mean(direct),
                          rtol=2e-10,atol=1e-12)
    assert np.isinf(covariance_validation_loss(0.,statistics))
    assert np.isclose(covariance_validation_loss(1.,statistics),
                      np.mean([np.vdot(a,R@a).real for S,R in fixtures]))


def test_validation_selection_is_invariant_to_hidden_truth_with_observations_fixed(monkeypatch):
    from copy import deepcopy
    from lib.final_beamforming import validation_case,heldout_snapshots
    import lib.final_beamforming as beam
    cfg=configuration('verify')
    cfg.update(validation_episodes=2,heldout_snapshots=17,loading_bracket_points=5,
        great_step_multipliers=[.02,.08],great_inner_steps=[1],
        great_window_ratios=[.5,1.],grouse_steps=[.01,.1],
        stationary_T=8,stationary_times=[1,2,4,8],stationary_primary_t=8)
    n=4;times=[1,2,4,8]
    selection_times=[]
    score=beam._candidate_score
    def record_selection_times(lines,data,selected):
        selection_times.append(tuple(selected))
        return score(lines,data,selected)
    monkeypatch.setattr(beam,'_candidate_score',record_selection_times)
    measured=[]
    for episode in range(2):
        stream=generate_stream(n,8,'static',110+episode)
        Z={t:heldout_snapshots(stream,t,cfg['heldout_snapshots']) for t in times}
        stream['validation_covariances']={t:row.T@row.conj()/len(row) for t,row in Z.items()}
        measured.append(stream)
    changed=deepcopy(measured)
    for stream in changed:
        # NaNs fail loudly if any tuning procedure accidentally uses truth.
        stream['h']=np.full((8,n),np.nan+1j*np.nan)
        stream['angles']=np.full(8,np.nan)
        stream['true_drift']=np.inf
        stream.pop('heldout_seed')
    first,curves1=validation_case(cfg,n,'static',True,validation_data=measured)
    second,curves2=validation_case(cfg,n,'static',True,validation_data=changed)
    assert first==second
    assert first['times']==times
    assert first['primary_times']==times
    assert first['primary_evaluation_time']==8
    assert first['validation_scored_times']==times
    assert first['shared_beta']==1.
    assert set(selection_times)=={tuple(times)}
    pd.testing.assert_frame_equal(pd.DataFrame(curves1),pd.DataFrame(curves2))
