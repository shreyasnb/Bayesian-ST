import numpy as np
from scipy.spatial.transform import Rotation
from lib.causal_variance import CausalVarianceTracker, align_completed_block
from lib.block_em import BlockData, fit_block_em, ordered_variance_update


def centered(A):
    return A-np.trace(A)*np.eye(len(A))/len(A)


def test_completed_fit_is_future_only_and_does_not_reassimilate():
    rng=np.random.default_rng(8301)
    U=rng.normal(size=(10,3))
    tracker=CausalVarianceTracker(3,1,.4,1.,beta=.93,block_length=4,em_iterations=4)
    expected=np.zeros((3,3))
    for j,u in enumerate(U):
        s,q=tracker.sigma2,tracker.tau2
        expected=centered(.93*expected+q*np.outer(u,u)/(2*s*(s+q)))
        record=tracker.update(u)
        assert np.allclose(tracker.A,expected,rtol=2e-12,atol=2e-12)
        assert record.sigma2_used==s and record.tau2_used==q
        assert record.state_change_from_fit==0
        assert record.fit_performed==((j+1)%4==0)
    assert [h['recorded_observations'] for h in tracker.fit_history]==[4,4]
    assert all(h['boundary_prior_count']==1 for h in tracker.fit_history)


def test_future_observations_cannot_change_past_state_or_scale():
    rng=np.random.default_rng(8302); U=rng.normal(size=(12,3))
    changed=U.copy();changed[8:]*=100
    def run(data):
        tr=CausalVarianceTracker(3,1,.4,1.,beta=.9,block_length=4,em_iterations=3)
        states=[]
        for u in data:
            tr.update(u);states.append((tr.A.copy(),tr.sigma2,tr.tau2))
        return states
    a,b=run(U),run(changed)
    for x,y in zip(a[:8],b[:8]):
        assert np.array_equal(x[0],y[0]) and x[1:]==y[1:]
    assert not np.allclose(a[-1][0],b[-1][0])


def test_noncommuting_wrist_alignment_and_fixed_prior():
    rng=np.random.default_rng(8303);L=8
    R=np.concatenate([np.eye(3)[None],Rotation.from_rotvec(rng.normal(size=(L,3))*.5).as_matrix()])
    G=np.array([R[j+1].T@R[j] for j in range(L)])
    Vworld=rng.normal(size=(L,3));normal=np.array([0.,0.,1.]);P=np.eye(3)-np.outer(normal,normal)
    Uworld=Vworld@P+.02*rng.normal(size=(L,3))
    U=np.einsum('tij,tj->ti',R[1:].transpose(0,2,1),Uworld)
    V=np.einsum('tij,tj->ti',R[1:].transpose(0,2,1),Vworld)
    A0=np.diag([2.,1.,-3.])
    ua,va,prior,product=align_completed_block(U,V,G,A0)
    assert np.allclose(ua,Uworld@R[-1])
    assert np.allclose(va,Vworld@R[-1])
    assert np.allclose(prior,R[-1].T@A0@R[-1])
    tracker=CausalVarianceTracker(3,2,.002,None,beta=.85,block_length=L,
        em_iterations=4,A0=A0)
    for u,v,g in zip(U,V,G):tracker.update(u,v,g)
    assert np.allclose(tracker.last_fit_data.U,ua)
    assert np.allclose(tracker.last_fit_data.V,va)
    assert np.allclose(tracker.last_fit_data.boundary_prior,prior)
    independent=fit_block_em(BlockData(ua,2,V=va,boundary_prior=prior),.002,
        iterations=4,floor=1e-8,tol=1e-9,ascent_tolerance=1e-7)
    assert np.isclose(tracker.sigma2,independent.s)
    assert tracker.records[-1].state_change_from_fit==0


def test_score_is_before_assimilation_and_does_not_mutate():
    tr=CausalVarianceTracker(3,1,.3,1.,block_length=2,em_iterations=2)
    tr.update(np.array([1.,.2,.1]))
    state,scales,t=tr.A.copy(),(tr.sigma2,tr.tau2),tr.t
    score,ok=tr.logscore(np.array([.1,.3,-.2]))
    assert np.isfinite(score) and ok
    assert np.array_equal(tr.A,state) and (tr.sigma2,tr.tau2)==scales and tr.t==t


def test_zero_data_and_joint_variance_floors():
    assert ordered_variance_update(.1,.3,3,1,floor=.4)==(.4,0.)
    assert ordered_variance_update(.8,.2,3,1,floor=.4)==(.4,.4)
    tr=CausalVarianceTracker(3,1,.3,1.,block_length=4,em_iterations=4,floor=.02)
    for _ in range(4):tr.update(np.zeros(3))
    assert tr.sigma2==.02 and tr.tau2==0
    assert np.array_equal(tr.A,np.zeros((3,3)))
    # With a nonuniform boundary prior, tau2=0 is not asserted to be absorbing.
    data=BlockData(np.tile([3.,0.,0.],(8,1)),1,boundary_prior=np.diag([20.,-10.,-10.]))
    fit=fit_block_em(data,.5,0.,iterations=2,floor=.001)
    assert fit.tau2>0


def test_known_resistance_transforms_input_and_noise_together():
    from lib.final_prediction import configuration,calibration_stream
    cfg=configuration('verify')
    matched=calibration_stream('contact','fixed_state',cfg,1,'test',
                               'known_resistance_correct_noise',True)
    raw=calibration_stream('contact','fixed_state',cfg,1,'test',
                           'known_resistance_raw_isotropic',True)
    assert np.allclose(matched['clean'],np.einsum('tij,tj->ti',matched['P'],matched['V']))
    root=np.diag(np.sqrt(cfg['mismatch']['anisotropic_damping']))
    assert np.allclose((raw['U']-raw['clean'])@np.linalg.inv(root),
                       matched['U']-matched['clean'])
    assert matched['model_correct'] and not raw['model_correct']


def test_contact_loss_scores_the_actual_free_response():
    from lib.final_prediction import configuration,calibration_stream,score_only_prediction
    cfg=configuration('verify')
    stream=calibration_stream('contact','fixed_state',cfg,2,'test','contact_loss',True)
    query=np.array([1.,0.,0.])
    score=score_only_prediction(np.zeros((3,3)),stream['U'][0],stream['V'][0],
        stream,0,'contact',.2,None,cfg,query)
    # A uniform plane mean is (2/3)I, while actual unconstrained velocity is v.
    assert np.isclose(score['response_loss_mean'],1/9,atol=1e-10)
    assert not score['state_target_applicable']


def test_observed_validation_is_invariant_to_hidden_truth_fields():
    from copy import deepcopy
    import pandas as pd
    from lib.final_prediction import configuration,adaptation_trial
    from scripts.run_final_prediction import primary_stream,select_observed_policies
    cfg=configuration('verify')
    cfg.update(primary_beam_n=3,primary_beam_T=32,primary_contact_T=32,
               primary_report_every=16,heldout_draws=16)
    candidates=cfg['policy_candidates'][:2]
    for example in ['beamforming','contact']:
        stream=primary_stream(example,cfg,'noise_change',0,'validation')
        hidden=deepcopy(stream)
        hidden['P']=np.full_like(stream['P'],np.nan)
        if example=='beamforming':hidden['h']=np.full_like(stream['h'],np.nan)
        else:hidden['world_normal']=np.full_like(stream['world_normal'],np.nan)
        frames=[]
        with np.errstate(invalid='ignore',divide='ignore'):
            for data in [stream,hidden]:
                records=[]
                for i,policy in enumerate(candidates):
                    p={name:policy for name in ['known','misspecified','learned']}
                    metrics=adaptation_trial(example,data,p,cfg,0,'validation','noise_change')[0]
                    d=pd.DataFrame(metrics)
                    for method,g in d[d.control.eq('mean')].groupby('method'):
                        records.append(dict(method=method,candidate=i,
                            condition='noise_change',episode=0,
                            loss=float(g.future_task_loss.mean()),
                            true_task_loss=float(g.true_task_loss.mean()),
                            numerical_ok=bool(g.mean_ok.all())))
                frames.append(pd.DataFrame(records))
        assert np.array_equal(frames[0].loss,frames[1].loss)
        assert frames[1].true_task_loss.isna().all()
        assert select_observed_policies(frames[0],candidates)==select_observed_policies(frames[1],candidates)


def test_validation_missing_or_failed_episodes_cannot_win():
    import pandas as pd
    from scripts.run_final_prediction import select_observed_policies
    candidates=[dict(beta=1.),dict(beta=.9),dict(beta=.8)]
    rows=[]
    for method in ['known','misspecified','learned']:
        # Candidate 0 has deceptively excellent loss but is missing episode 1.
        rows.append(dict(method=method,candidate=0,condition='a',episode=0,loss=0.,numerical_ok=True))
        for ep in [0,1]:
            rows.append(dict(method=method,candidate=1,condition='a',episode=ep,loss=2.,numerical_ok=True))
            rows.append(dict(method=method,candidate=2,condition='a',episode=ep,
                             loss=0. if ep==0 else np.nan,numerical_ok=ep==0))
    selected=select_observed_policies(pd.DataFrame(rows),candidates,{('a',0),('a',1)})
    assert all(p==candidates[1] for p in selected.values())


def test_validation_requires_every_requested_checkpoint(tmp_path,monkeypatch):
    from lib.final_prediction import configuration
    import scripts.run_final_prediction as runner
    cfg=configuration('verify');cfg.update(validation_episodes=1,primary_beam_T=64,
        beam_conditions=['noise_change'])
    def fake_execute(tasks,*args):
        output=[]
        for task in tasks:
            policy=task[-2][task[-1]]
            times=[32] if policy['beta']==1 else [32,64]
            output.append(dict(kind='adaptation',rows=([
                dict(method=task[-1],control='mean',t=t,
                     future_task_loss=0. if policy['beta']==1 else 1.,mean_ok=True)
                for t in times],[],{})))
        return output
    monkeypatch.setattr(runner,'execute',fake_execute)
    frame,selected,failures=runner.validation('beamforming',cfg,tmp_path,{},1)
    assert not failures
    assert all(not p for p in frame.loc[frame.beta.eq(1.),'numerical_ok'])
    assert all(p['beta']!=1 for p in selected.values())


def test_contact_friction_zero_is_matched_and_reduces_tangential_speed():
    from lib.final_prediction import configuration,calibration_stream,score_only_prediction
    cfg=configuration('verify');cfg['mismatch']['friction_coefficient']=0.
    zero=calibration_stream('contact','fixed_state',cfg,0,'test','friction',True)
    expected=np.einsum('tij,tj->ti',zero['P'],zero['V'])
    assert np.allclose(zero['clean'],expected,atol=1e-12)
    query=np.array([.7,-.5,.9]);P=zero['P'][0]
    # Concentration is irrelevant here: all three rules must compare to Pq.
    score=score_only_prediction(np.zeros((3,3)),zero['U'][0],zero['V'][0],
        zero,0,'contact',.2,None,cfg,query)
    assert np.isclose(score['response_loss_mean'],np.linalg.norm(2*query/3-P@query)**2)
    cfg['mismatch']['friction_coefficient']=.35
    friction=calibration_stream('contact','fixed_state',cfg,0,'test','friction',True)
    assert np.all(np.linalg.norm(friction['clean'],axis=1)<=np.linalg.norm(expected,axis=1)+1e-12)
    assert np.any(np.linalg.norm(friction['clean'],axis=1)<np.linalg.norm(expected,axis=1)-1e-6)


def test_score_summaries_keep_independent_cohorts_separate():
    import pandas as pd
    from scripts.run_final_prediction import summaries
    rows=[]
    for cohort,count,offset in [('calibration',3,0.),('mismatch',2,10.)]:
        for ep in range(count):
            rows.append(dict(example='contact',cohort=cohort,regime='fixed_state',
                stress='matched',split='test',method='learned',episode=ep,
                mixture_log_score=offset+ep,gaussian_log_score=ep,
                risk_realized=1.,risk_predicted=1.,numerical_ok=True,
                response_loss_mean=1.,response_loss_mode=2.,response_loss_iso=3.))
    result=summaries(pd.DataFrame(),pd.DataFrame(),pd.DataFrame(rows))['score_summary']
    assert dict(zip(result.cohort,result.episodes))==dict(calibration=3,mismatch=2)
    assert dict(zip(result.cohort,result.score_gain))==dict(calibration=0.,mismatch=10.)
