"""Independent exact-moment and direct-Gaussian aggregation references."""
import itertools
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import null_space
from scipy.stats import multivariate_normal
from lib.beamforming_theory import passive_fisher_information
from lib.guide_information_counts import score_prefixes


def test_observation_count_information_from_exact_gaussian_cubature():
    # Three-point Gauss--Hermite integrates all score products exactly, since
    # Gaussian-covariance scores are quadratic and their products have degree 4.
    for cx,n in [(False,3),(True,2)]:
        s,q=.7,1.3;d=2*n if cx else n
        indices=np.array(list(itertools.product(range(3),repeat=d)))
        nodes=np.array([-np.sqrt(3),0.,np.sqrt(3)])[indices]
        weights=np.prod(np.array([1/6,2/3,1/6])[indices],axis=1)
        scale=np.r_[np.sqrt(s+q),np.full(n-1,np.sqrt(s))]
        U=(nodes[:,:n]+1j*nodes[:,n:])*scale/np.sqrt(2) if cx else nodes*scale
        a=np.repeat(np.arange(len(U)),len(U));b=np.tile(np.arange(len(U)),len(U))
        streams=np.stack([U[a],U[b]],axis=1);probability=weights[a]*weights[b]
        X=np.eye(n)[:,:1];Q=np.eye(n)[:,1:]
        scores=score_prefixes(streams,X,s,q,[1,2],cx,Q)
        for L in (1,2):
            mean=probability@scores[L]
            moment=(scores[L]*probability[:,None]).T@scores[L]
            expected=passive_fisher_information(n,1,s,q,observations=L,complex_data=cx)
            assert_allclose(mean,0,atol=2e-14)
            assert_allclose(moment,expected,atol=3e-13)


def test_grouped_scores_match_direct_normalized_gaussian_finite_differences():
    rng=np.random.default_rng(2026110931);s,q=.6,1.4;n=3
    for cx in (False,True):
        def normal(shape):
            a=rng.normal(size=shape)
            return (a+1j*rng.normal(size=shape))/np.sqrt(2) if cx else a
        X=np.linalg.qr(normal((n,1)))[0];Q=null_space(X.conj().T)
        U=normal((3,4,n));counts=[1,3,4]
        summed=score_prefixes(U,X,s,q,counts,cx,Q)
        dimension=(2 if cx else 1)*(n-1);step=1e-5
        for coordinate in range(dimension):
            z=np.zeros(dimension);z[coordinate]=step
            def direct(sign):
                vector=sign*(z[:n-1]+1j*z[n-1:] if cx else z)
                Y=X+Q@vector[:,None];Y/=np.linalg.norm(Y)
                R=s*np.eye(n)+q*Y@Y.conj().T
                if cx:
                    C=.5*np.block([[R.real,-R.imag],[R.imag,R.real]])
                    data=np.concatenate([U.real,U.imag],axis=-1)
                    return multivariate_normal.logpdf(data,cov=C)
                return multivariate_normal.logpdf(U,cov=R)
            derivative=(direct(1)-direct(-1))/(2*step)
            for L in counts:
                assert_allclose(summed[L][:,coordinate],derivative[:,:L].sum(axis=1),atol=3e-8,rtol=2e-8)
