"""Independent references and safety guards for the appendix-inspired backend."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import i0e,i1e
from lib.spectral_series import rank_one_bingham
from lib.moments import mean_bingham,log_normalizer,quadrature_gr13,sample_bingham


def test_real_and_complex_analytic_normalizers():
    for d in [.1,8.,80.,600.]:
        A=np.diag([d,0.])
        r=rank_one_bingham(A,tol=1e-12)
        assert_allclose(r.log_normalizer,d+np.log(i0e(d/2)),atol=2e-10)
        assert_allclose(r.mean[0,0],.5+.5*i1e(d/2)/i0e(d/2),atol=2e-11)
        c=rank_one_bingham(A,complex_data=True,tol=1e-12)
        assert_allclose(c.log_normalizer,d+np.log(-np.expm1(-d))-np.log(d),atol=2e-10)
        assert_allclose(c.mean[0,0],1/(-np.expm1(-d))-1/d,atol=2e-11)
        assert r.accuracy_met and c.accuracy_met


def test_shift_equivariance_and_repeated_spectrum():
    rng=np.random.default_rng(537)
    for complex_data in [False,True]:
        V=rng.normal(size=(8,8))
        if complex_data:V=V+1j*rng.normal(size=(8,8))
        U=np.linalg.qr(V)[0];A=np.diag([-3.,-3.,0.,0.,0.,2.,2.,2.])
        r=rank_one_bingham(A,complex_data=complex_data,tol=1e-12)
        shifted=rank_one_bingham(U@A@U.conj().T+11*np.eye(8),complex_data=complex_data,tol=1e-12)
        assert_allclose(shifted.mean,U@r.mean@U.conj().T,atol=3e-11)
        assert_allclose(shifted.log_normalizer,r.log_normalizer+11,atol=3e-11)
        assert_allclose(np.trace(r.mean),1.,atol=1e-11)
        assert_allclose(r.mean.diagonal()[2:5],r.mean[2,2],atol=1e-14)


def test_gradient_of_same_fixed_approximation_including_uniform_point():
    H=np.array([[1.,.2,0],[.2,-.3,.4],[0,.4,-.1]])
    for A,order in [(np.zeros((3,3)),0),(np.zeros((3,3)),5),(np.diag([0.,-8.,-80.]),150)]:
        r=rank_one_bingham(A,fixed_order=order);h=1e-5
        plus=rank_one_bingham(A+h*H,fixed_order=order)
        minus=rank_one_bingham(A-h*H,fixed_order=order)
        assert_allclose((plus.log_normalizer-minus.log_normalizer)/(2*h),np.trace(r.mean@H),atol=3e-9)
        assert_allclose(np.trace(r.mean),1,atol=1e-11)
    # An insufficient fixed-order diagnostic is deliberately not accepted.
    assert not rank_one_bingham(np.diag([1.,-1.,0.]),fixed_order=0).accuracy_met


def test_rank_one_quadrature_and_complement_api():
    A=np.diag([0.,-8.,-80.]);q=quadrature_gr13(A,96)
    r=rank_one_bingham(A,tol=1e-12)
    assert_allclose(r.mean,q.mean,atol=2e-11)
    assert_allclose(r.log_normalizer,q.log_normalizer,atol=2e-11)
    assert_allclose(mean_bingham(A,1),r.mean,atol=1e-10)
    assert_allclose(mean_bingham(A,2),np.eye(3)-rank_one_bingham(-A).mean,atol=1e-10)
    assert_allclose(log_normalizer(A,2),np.trace(A)+rank_one_bingham(-A).log_normalizer,atol=1e-10)
    H=np.diag([.3,-.2,.7]);h=1e-5
    derivative=(log_normalizer(A+h*H,2)-log_normalizer(A-h*H,2))/(2*h)
    assert_allclose(derivative,np.trace(mean_bingham(A,2)@H),atol=4e-9)


def test_explicit_sampling_budget_preserves_reference_kernel():
    A=np.diag([0.,-1.,-2.,-3.])
    reference=sample_bingham(A,1,514,draws=128,burnin=32).mean
    result=mean_bingham(A,1,514,draws=128,burnin=32)
    assert_allclose(result,reference,atol=0,rtol=0)
    with pytest.raises(ValueError):log_normalizer(np.eye(6),3)
    with pytest.raises(ValueError):mean_bingham(np.diag(np.arange(6.)),3,backend='series')
    with pytest.raises(ArithmeticError):rank_one_bingham(np.diag([0.,600.]),max_order=8)


def test_complex_realification_energy_and_measure_are_distinct():
    x=np.array([1,1j])/np.sqrt(2);P=np.outer(x,x.conj())
    A=np.array([[1.,.2+.3j],[.2-.3j,-.4]])
    def realify(B):return np.block([[B.real,-B.imag],[B.imag,B.real]])
    assert_allclose(np.trace(realify(A)@realify(P)),2*np.trace(A@P).real,atol=1e-14)
    # Exact uniform variances for complex Gr(1,2) versus unrestricted real Gr(2,4).
    complex_variance=1/12
    real_duplicated_half_variance=(1/12+1/12-2/36)/4
    assert_allclose(real_duplicated_half_variance,1/36)
    assert complex_variance!=real_duplicated_half_variance
