import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import null_space
from lib.geometry import orth, projector
from lib.moments import quadrature, sample_bingham
from lib.theory import fisher_information


def test_uniform_derivative_and_complement():
    H=np.diag([-.7,.2,.5]);e=1e-3
    plus=quadrature(e*H).mean;minus=quadrature(-e*H).mean
    # 2*k*r/[n(n-1)(n+2)] = 2/15 for n3 k1.
    assert_allclose((plus-minus)/(2*e),2/15*H,atol=1e-8)
    # Complement law q_A(k=2) equals I-P(-A,k=1).
    A=np.diag([0.,-1.,-3.])
    r=sample_bingham(A,2,21,draws=16000,burnin=2000)
    assert_allclose(r.mean,np.eye(3)-quadrature(-A).mean,atol=max(.008,5*r.mcse))


def test_response_risk_and_modal_excess():
    q=quadrature(np.diag([0.,-2.,-15.]),order=64)
    v=np.array([.5,1.,2.]); M=q.mean; mode=np.diag([1.,0.,0.])
    projections=np.einsum('bni,i,bmi->bnm',q.bases,np.ones(1),q.bases) @ v
    loss=np.sum((projections-M@v)**2,axis=1)@q.weights
    modal_loss=np.sum((projections-mode@v)**2,axis=1)@q.weights
    assert_allclose(loss,v@(M-M@M)@v,atol=1e-12)
    assert_allclose(modal_loss-loss,np.linalg.norm((mode-M)@v)**2,atol=1e-12)


def test_marginal_information_gaussian_covariance():
    n,k,s,tau=5,2,.3,1.1
    P=np.diag([1.,1.,0.,0.,0.]);R=s*np.eye(n)+tau*P; inv=np.linalg.inv(R)
    values=[]
    for i in range(k):
        for a in range(k,n):
            D=np.zeros((n,n));D[i,a]=D[a,i]=tau
            values.append(.5*np.trace(inv@D@inv@D))
    assert_allclose(values,tau*tau/(s*(s+tau)),atol=1e-12)


def test_local_identifiability_does_not_imply_global():
    # Only e1 is excited: local full information at X=e1 in Gr(1,3),
    # but distinct lines e2/e3 elsewhere give identical zero responses.
    V=np.eye(3)[:,:1];X=V
    assert np.linalg.eigvalsh(fisher_information(X,V,1.))[0]>0
    assert_allclose(np.diag([0,1,0])@V,np.diag([0,0,1])@V)


def test_trace_shift_scores():
    A=np.diag([0.,-2.,-5.]);B=np.diag([1.,2.,3.]);b=np.trace(B)/3
    direct=quadrature(A+B).log_normalizer-quadrature(A).log_normalizer
    centered=b+quadrature(A+B-b*np.eye(3)).log_normalizer-quadrature(A).log_normalizer
    assert_allclose(direct,centered,atol=1e-12)
