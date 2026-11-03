"""Reproduce the Appendix J energy and invariant-measure counterexample."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def realify(A):
    return np.block([[A.real,-A.imag],[A.imag,A.real]])


def main():
    a=2.;draws=250000;seed=26
    x=np.array([1.,1j])/np.sqrt(2)
    P=np.outer(x,x.conj());A=np.array([[1.,.2+.3j],[.2-.3j,-.4]])
    assert np.isclose(np.trace(realify(A)@realify(P)),2*np.trace(A@P).real)
    rng=np.random.default_rng(seed)
    X=np.linalg.qr(rng.normal(size=(draws,4,2)))[0]
    # Half of the duplicated parameter has energy (a/2)(P_00+P_22).
    values=np.exp((a/2)*np.sum(X[:,[0,2],:]**2,axis=(1,2)))
    mean=float(values.mean());mcse=float(values.std(ddof=1)/np.sqrt(draws))
    exact=float(np.expm1(a)/a)
    assert abs(mean-exact)>10*mcse
    result=dict(realified_energy_factor=2,complex_uniform_variance=1/12,
                full_real_half_duplicated_variance=1/36,
                real_Gr24_normalizer_mc=mean,mcse=mcse,
                complex_Gr12_normalizer=exact,draws=draws,seed=seed,
                paper_sha256='6ebeba4fb472276ce70d85eed79feece53bb82723775c339edcad117d721bd02',
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'results/paper_audit_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
