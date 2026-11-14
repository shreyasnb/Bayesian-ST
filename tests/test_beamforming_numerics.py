import numpy as np
from numpy.testing import assert_allclose
from numpy.polynomial.legendre import leggauss
from lib.beamforming_numerics import complex_moments
from lib.spectral_series import rank_one_bingham


def unitary(n,rng):
    Q,R=np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))
    return Q


def simplex3(lam,order=96):
    """Independent Cartesian simplex integration, including its Jacobian."""
    nodes,weights=leggauss(order);x=(nodes+1)/2;w=weights/2
    first,second=np.meshgrid(x,x,indexing='ij')
    X=np.stack([first,(1-first)*second,(1-first)*(1-second)],axis=-1).reshape(-1,3)
    base=(2*np.outer(w,w)*(1-first)).ravel()
    mass=base*np.exp(X@np.asarray(lam)-max(lam));Z=mass.sum();mass/=Z
    return X.T@mass,np.einsum('b,bi,bj->ij',mass,X,X),float(max(lam)+np.log(Z))


def test_native_complex_cp1_formula_matches_series_and_shift():
    rng=np.random.default_rng(89127);Q=unitary(2,rng)
    for gap in [0.,1e-7,.4,8.,200.]:
        A=Q@np.diag([-gap/2,gap/2])@Q.conj().T
        exact=complex_moments(A);series=rank_one_bingham(A,complex_data=True,tol=1e-12)
        assert_allclose(exact.mean,series.mean,atol=2e-12)
        assert_allclose(exact.log_normalizer,series.log_normalizer,atol=3e-12)
        shifted=complex_moments(A+13*np.eye(2))
        assert_allclose(shifted.mean,exact.mean,atol=2e-13)
        assert_allclose(shifted.log_normalizer,exact.log_normalizer+13,atol=2e-13)


def test_complex_simplex_covariance_matches_independent_quadrature():
    for lam in [[3.,3.,0.],[0.,.001,40.],[0.,0.,0.]]:
        expected,second,logz=simplex3(lam)
        result=complex_moments(np.diag(lam),tol=1e-12)
        assert_allclose(result.mean.diagonal().real,expected,atol=4e-12)
        assert_allclose(result.log_normalizer,logz,atol=3e-12)




def test_repeated_complex_eigenvalues_match_80_digit_beta_integrals():
    import mpmath as mp
    with mp.workdps(80):
        for n,m,gap in [(3,2,80.),(32,16,600.)]:
            z=mp.hyp1f1(m,n,gap)
            top_group=mp.mpf(m)/n*mp.hyp1f1(m+1,n+1,gap)/z
            spectrum=np.r_[np.zeros(n-m),np.full(m,gap)]
            result=complex_moments(np.diag(spectrum),tol=1e-12)
            expected=np.r_[np.full(n-m,float((1-top_group)/(n-m))),
                           np.full(m,float(top_group/m))]
            assert_allclose(result.mean.diagonal().real,expected,atol=3e-11)
            assert_allclose(result.log_normalizer,float(mp.log(z)),atol=3e-10)
            assert result.numerical_ok
