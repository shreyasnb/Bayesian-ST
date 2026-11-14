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


