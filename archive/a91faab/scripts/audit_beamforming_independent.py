"""Independent development-only math audit; no formal test outcomes are opened.

Run from any directory with Python, NumPy and SciPy. The JSONL output retains
an intentionally unresolved tied-top rejection-budget stress. Exact-integral
and Monte Carlo checks have separate tolerances and uncertainty scopes.
"""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.integrate import quad
from scipy.special import betaln,logsumexp
from scipy.stats import invgamma
from scipy.linalg import null_space
from lib.beamforming_numerics import complex_moments,sample_complex_bingham,invert_complex_mean,sample_complex_transition,transition_coefficients_complex,cp1_power_cdf
from lib.beamforming_theory import passive_fisher_information,passive_scores,passive_covariance_certificate
from lib.beamforming_decisions import isotropized_mean
from lib.beamforming_inference import PowerPrior,ordered_prior_normalizer,truncated_inverse_gamma,kappa_log_density,joint_covariance_mean
from lib.beamforming_protocol import proper_complex,haar_line
from lib.models import increment
rng=np.random.default_rng(81536029)
checks=[]
def save(name,**data):checks.append(dict(check=name,**data));print(json.dumps(checks[-1]),flush=True)
def unitary(n):return np.linalg.qr(proper_complex(rng,(n,n)))[0]
start=time.perf_counter()
for n,m,k in [(8,3,.001),(8,3,60.),(8,3,400.),(16,7,100.),(8,1,1000.)]:
    # Repeated eigenvalues reduce this independently to an exponentially tilted Beta law.
    if k<1:
        den=quad(lambda r:np.exp(k*(r-1)+(m-1)*np.log(r)+(n-m-1)*np.log1p(-r)-betaln(m,n-m)),0,1,epsabs=1e-13)[0]
        num=quad(lambda r:r*np.exp(k*(r-1)+(m-1)*np.log(r)+(n-m-1)*np.log1p(-r)-betaln(m,n-m)),0,1,epsabs=1e-13)[0]
        logden=np.log(den)
    else:
        den=quad(lambda y:np.exp(-y)*y**(n-m-1)*(1-y/k)**(m-1),0,k,epsabs=1e-12,epsrel=1e-12)[0]
        num=quad(lambda y:(1-y/k)*np.exp(-y)*y**(n-m-1)*(1-y/k)**(m-1),0,k,epsabs=1e-12,epsrel=1e-12)[0]
        logden=np.log(den)-(n-m)*np.log(k)-betaln(m,n-m)
    A=np.diag(np.r_[np.full(m,k),np.zeros(n-m)]);answer=complex_moments(A)
    logerr=abs(answer.log_normalizer-(k+logden));meanerr=abs(np.trace(answer.mean[:m,:m]).real-num/den)
    assert answer.numerical_ok and logerr<2e-8 and meanerr<2e-9,(n,m,k,logerr,meanerr)
    Q=unitary(n);rotated=complex_moments(Q@A@Q.conj().T+17.3*np.eye(n))
    inverror=np.linalg.norm(rotated.mean-Q@answer.mean@Q.conj().T,2)
    assert inverror<2e-10 and abs(rotated.log_normalizer-answer.log_normalizer-17.3)<2e-9
    sample=sample_complex_bingham(A,draws=8192,rng=rng,strict=False)
    observed=sample.diagnostics['acceptance'];predicted=np.exp(answer.log_normalizer-k+max(0.,sample.diagnostics['log_box_to_uniform_acceptance_ratio']))
    predicted=float(np.clip(predicted,0.,1.));se=np.sqrt(predicted*(1-predicted)/sample.diagnostics['proposals'])
    assert abs(observed-predicted)<7*se+.001
    assert sample.diagnostics['numerical_ok'] or (n,m,k)==(16,7,100.)
    save('tilted_beta',n=n,m=m,kappa=k,logZ_error=logerr,mean_error=meanerr,rotation_error=inverror,acceptance=observed,acceptance_theory=predicted,proposals=sample.diagnostics['proposals'],sampling_ok=sample.diagnostics['numerical_ok'],retained_draws=len(sample.vectors),status='verified' if sample.diagnostics['numerical_ok'] else 'explicit budget limit')
S=sample_complex_bingham(np.zeros((8,8)),draws=16384,rng=rng).vectors
p=abs(S)**2;ph=S[:,0]*S[:,1].conj()/np.maximum(abs(S[:,0]*S[:,1]),1e-300)
assert abs(p[:,0].mean()-1/8)<.004 and abs(np.mean(p[:,0]**2)-2/(8*9))<.002
assert abs(ph.mean())<.03 and abs(np.mean(ph**2))<.03
assert np.max(abs(np.mean(S[:-1].conj()*S[1:],axis=0)))<.015
save('native_uniform_measure',coordinate_mean=float(p[:,0].mean()),coordinate_second=float(np.mean(p[:,0]**2)),phase_first=float(abs(ph.mean())),phase_second=float(abs(np.mean(ph**2))))
# Boundary layers are checked at native CP1 concentration far beyond the study.
for gap in [.01,300.,1e4,1e6]:
    for aligned in [True,False]:
        query=np.array([0.,1.]) if aligned else np.array([1.,0.])
        threshold=1-1/gap if gap>1 and aligned else 1/gap if gap>1 else .37
        expected=(np.exp(-gap*(1-threshold))*(-np.expm1(-gap*threshold))/(-np.expm1(-gap)) if aligned else -np.expm1(-gap*threshold)/(-np.expm1(-gap)))
        computed=cp1_power_cdf(np.diag([0.,gap]),query,threshold)
        assert abs(computed-expected)<2e-10
        save('CP1_boundary_CDF',gap=gap,aligned=aligned,threshold=threshold,CDF=computed,analytic_error=abs(computed-expected))
query=np.array([np.sqrt(.77),np.sqrt(.23)],complex)
ns=32768;native=sample_complex_bingham(np.diag([0.,1e6]),draws=ns,rng=rng).vectors
power=abs(native.conj()@query)**2
for value in [.2295,.23,.2305]:
    computed=cp1_power_cdf(np.diag([0.,1e6]),query,value);empirical=float(np.mean(power<=value))
    se=float(np.sqrt(max(computed*(1-computed),1/ns)/ns))
    assert abs(computed-empirical)<6*se+.0005
    save('CP1_nonaligned_boundary_CDF',gap=1e6,threshold=value,CDF=computed,iid_empirical=empirical,iid_CDF_standard_error=se)
# Ill-conditioned warm starts must still recover the unitary-transformed interior mean.
A=np.diag([35.,35.,8.,0.,-2.,-5.,-9.,-15.]);Q=unitary(8);A=Q@A@Q.conj().T;target=complex_moments(A).mean
fit=invert_complex_mean(target,initial=np.diag(np.linspace(-50,50,8)),tol=1e-8)
assert fit.converged and np.linalg.norm(complex_moments(fit.A).mean-target,2)<1e-8
save('inverse_warm_start',residual=float(fit.residual),iterations=fit.iterations)
# Independent covariance-likelihood KL Hessian uses no tangent derivative helper.
n=4;s=.2;q=1.;Q=unitary(n);h=Q[:,0];Z=Q[:,1:];R=s*np.eye(n)+q*np.outer(h,h.conj());d=2*(n-1);eps=1e-4
KL=[]
for j in range(d):
    v=np.zeros(n-1,complex);v[j%(n-1)]=eps*(1 if j<n-1 else 1j)
    g=np.sqrt(1-eps**2)*h+Z@v;Rg=s*np.eye(n)+q*np.outer(g,g.conj())
    kl=np.linalg.slogdet(Rg)[1]-np.linalg.slogdet(R)[1]+np.trace(np.linalg.solve(Rg,R)).real-n
    KL.append(2*kl/eps**2)
expected=2*q*q/(s*(s+q));assert max(abs(np.asarray(KL)-expected))<1e-5
U=proper_complex(rng,32768,q)[:,None]*h+proper_complex(rng,(32768,n),s)
scores=passive_scores(U,h[:,None],s,q,complement=Z);I=scores.T@scores/len(U)
assert np.linalg.norm(I-expected*np.eye(d),'fro')/(expected*np.sqrt(d))<.04
save('complex_fisher',coefficient=expected,KL_hessian_max_error=float(max(abs(np.asarray(KL)-expected))),MC_frobenius_relative_error=float(np.linalg.norm(I-expected*np.eye(d),'fro')/(expected*np.sqrt(d))))
# Certificate fixture, plus a genuinely transported moving state with zero drift.
h0=haar_line(rng,4);certificate=passive_covariance_certificate(np.broadcast_to(h0,(4096,4)),.2,1.)
assert abs(certificate.eta-.2570269391)<1e-10 and abs(certificate.bound-.3459438204)<1e-10
G=np.array([unitary(4) for _ in range(11)]);hs=[h0]
for j in range(1,len(G)):hs.append(G[j]@hs[-1])
transported=passive_covariance_certificate(np.asarray(hs),.2,1.,G=G)
assert transported.drift<1e-13
save('fixed_time_bound',eta=certificate.eta,bound=certificate.bound,transported_drift=transported.drift)
# Conditional moment projection is no more and no less than a first-moment identity.
for k in [0.,.001,4.,20.,1e4]:
    h=haar_line(rng,8);states=sample_complex_transition(h,k,rng=rng,size=8192)
    c=transition_coefficients_complex(8,k);M=states.T@states.conj()/len(states);E=c.rho*np.outer(h,h.conj())+(1-c.rho)*np.eye(8)/8
    assert np.linalg.norm(M-E,'fro')<.027
    save('transition_mean',kappa=k,rho=c.rho,moment_error=float(np.linalg.norm(M-E,'fro')))
# Single-snapshot equality follows from exact symmetry in all n-1 orthogonal directions.
u=proper_complex(rng,8);A=increment(u,.2,1.,complex_data=True);M=complex_moments(A).mean;P=np.outer(u,u.conj())/np.vdot(u,u).real
iso=isotropized_mean(M,P)
if isinstance(iso,tuple):iso=iso[0]
assert np.linalg.norm(M-iso)<1e-11
save('single_snapshot_isotropization',residual=float(np.linalg.norm(M-iso)))
# Extreme proper ordered-prior constants against integration of the original independent prior.
for prior in [PowerPrior(),PowerPrior(a_shape=.3,a_scale=.07,b_shape=7.,b_scale=3.),PowerPrior(a_shape=9.,a_scale=2.,b_shape=.4,b_scale=10.)]:
    exact=quad(lambda lb:np.exp(invgamma.logpdf(np.exp(lb),prior.b_shape,scale=prior.b_scale)+lb)*invgamma.sf(np.exp(lb),prior.a_shape,scale=prior.a_scale),-100,100,epsabs=1e-11)[0]
    computed=ordered_prior_normalizer(prior)
    assert abs(computed-exact)<5e-10
    save('ordered_prior',normalizer=computed,integration_error=abs(computed-exact))
# Slice density is independently checked with actual Beta-integrated transition Z, including small k.
prior=PowerPrior();n=8;T=4;overlap=2.1
vals=[]
for k in [.0001,.01,20.,1000.]:
    z=(quad(lambda r:(n-1)*(1-r)**(n-2)*np.exp(k*(r-1)),0,1,epsabs=1e-14)[0] if k<1 else (n-1)*k**(-(n-1))*quad(lambda y:y**(n-2)*np.exp(-y),0,k,epsabs=1e-12,epsrel=1e-12)[0])
    direct=(prior.kappa_shape-1)*np.log(k)-prior.kappa_rate*k+k*overlap-T*(k+np.log(z))
    err=abs(kappa_log_density(k,overlap,T,n,prior)-direct);assert err<1e-8
    vals.append(err)
save('kappa_density_normalizers',max_error=max(vals))
# Full response covariance is not M-M^2; that matrix only supplies summed risk.
hs=sample_complex_bingham(np.diag([6.,1.,-2.]),draws=32768,rng=rng).vectors
v=np.array([.2+.3j,.8-.2j,-.4j]);M=complex_moments(np.diag([6.,1.,-2.])).mean
response=hs*(hs.conj()@v)[:,None];centered=response-M@v;C=centered.T@centered.conj()/len(hs)
risk=float(np.vdot(v,(M-M@M)@v).real)
assert abs(np.trace(C).real-risk)<.004
assert np.linalg.norm(C-(M-M@M))>.03
save('response_covariance_scope',MC_risk=float(np.trace(C).real),analytic_risk=risk,full_covariance_difference=float(np.linalg.norm(C-(M-M@M))))
print(json.dumps(dict(completed=True,development_seed=81536029,checks=len(checks),seconds=time.perf_counter()-start)),flush=True)
