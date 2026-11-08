"""Build only the prerequisite notebook; example generation is separately gated."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
nb=nbf.v4.new_notebook()
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):cells.append(nbf.v4.new_code_cell(s))
md(r'''# Theory and numerical validation
This notebook is the prerequisite for both case studies and the main debugging entry point. It tests finite, stated budgets—not unrestricted scalability or a universal MCMC convergence claim. A failed cell prevents the validation certificate from being written. See `docs/claims.md` for the theory map and `docs/draft_notes.md` for Algorithm 1 clarifications.

Checks include normalized real/complex likelihoods, transport, excitation, ambiguous subspaces, fixed-time bounds, directional posterior risk, credible regions, certified extrema, EM ascent, boundary cases, independent chains, and measured dense scaling. Seeds and budgets are recorded below.''')
code('''from pathlib import Path
import sys, os, json, time, subprocess
ROOT=Path.cwd() if (Path.cwd()/'lib').exists() else Path.cwd().parent
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/tmp/bayesian-st-matplotlib')
from lib.plotting import plt, save
import numpy as np
import pandas as pd
from IPython.display import display
from lib.validation import core_hash
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
# Remove stale success before running any checks.
(OUT/'validation.json').unlink(missing_ok=True)
assert plt.rcParams['text.usetex'] is True
validated_hash=core_hash(ROOT)
started=time.perf_counter()
''')
md('''## Library regression tests
The tests use Gaussian densities and finite differences independent of the update implementation, deterministic quadrature and continuous CDF references, exact boundary fixtures, and independent chains at high concentration. The required suite must execute nonzero tests and return zero failures.''')
code('''run=subprocess.run([sys.executable,'-m','pytest','-q','--junitxml='+str(OUT/'pytest.xml')],cwd=ROOT,capture_output=True,text=True)
print(run.stdout)
if run.returncode: print(run.stderr)
assert run.returncode==0, 'Library tests failed'
import xml.etree.ElementTree as ET
suite=ET.parse(OUT/'pytest.xml').getroot()
tests=sum(int(s.attrib.get('tests',0)) for s in suite.iter('testsuite'))
assert tests>=30
''')
md(r"""## Computational appendix audit
The real normalizer and its gradient agree with Appendices C, F, and H of the supplied latent-information paper. Low truncation orders used for latent geodesics are not guaranteed accurate for concentrated tracking posteriors. The new rank-one spectral series adapts its order and reports truncation bounds. It supports real and native complex measures separately, and also gives codimension-one means by complement duality.

Appendix J's algebraic realification requires both a trace-factor correction and the correct structured measure. It does not license use of the unrestricted real Grassmann normalizer. The tests include the trace identity and an exact uniform-variance counterexample. The paper's distribution-parameter Fisher metric is a different object from the Gaussian observation Fisher matrix below. See `docs/latent_paper_audit.md`.
""")
code("""from lib.spectral_series import rank_one_bingham
from lib.moments import mean_bingham, log_normalizer
rows=[]
for n in (3,8,16,64):
    for complex_data in (False,True):
        Atest=np.diag(np.linspace(-20.,0.,n))
        begin=time.perf_counter()
        evaluated=rank_one_bingham(Atest,complex_data=complex_data)
        rows.append(dict(n=n,complex_data=complex_data,order=evaluated.order,
            seconds=time.perf_counter()-begin,truncation_bound=evaluated.moment_error_bound,
            trace_error=evaluated.trace_error))
        assert evaluated.accuracy_met and evaluated.moment_error_bound<=1e-10
        assert evaluated.trace_error<1e-10
series_table=pd.DataFrame(rows);display(series_table)
series_table.to_csv(OUT/'spectral_series_timing.csv',index=False)
# A low fixed order is a diagnostic approximation, not an accepted mean budget.
assert not rank_one_bingham(np.diag([0.,-8.,-80.]),fixed_order=3).accuracy_met
print('Complex uniform variance =',1/12,'; energy-corrected unrestricted real variance =',1/36)
""")
md(r'''## Excitation and identifiability
The score covariance checks Eq. (20). Complementary input directions can give information even when the noiseless response vanishes. Local positive information is distinct from global injectivity; the tests explicitly construct two globally ambiguous subspaces and an orientation attaining Eq. (22).''')
code('''from lib.studies import information_study, tracking_study, scalability_study
info=information_study()
display(info)
assert info.relative_error.max()<.08
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
axs[0].bar(info.input,info.lambda_min);axs[0].set_ylabel(r'$\\lambda_{\\min}(F)$')
axs[1].bar(info.input,info.relative_error);axs[1].set_ylabel('Score covariance error')
for ax in axs: ax.tick_params(axis='x',rotation=25)
save(fig,OUT/'theory_information');plt.show()
''')
md(r'''## Tracking, forgetting, and drift
Inputs are preassigned equal-norm coordinate cycles. The variance is fixed at the true value; no EM is used. All terms of Eq. (28) use the complete history. Invalid theorem conditions are retained and marked; a missing bound is not a failure of the theorem. The repeated-noise check below uses one predeclared terminal time, not simultaneous coverage over the curve.''')
code('''track=tracking_study(T=480,repeats=128)
display(track)
terminal=track[(track.drift==0)&(track.beta==1)&(track.t==480)].iloc[0]
assert bool(terminal.valid), 'Nonvacuous fixed-time fixture is required'
assert terminal.violations==0
# Exact one-sided 95% binomial upper bound for zero violations.
violation_upper=1-.05**(1/int(terminal.repeats))
assert violation_upper<.05
assert (~track.valid).any(), 'Stress cases must include invalid conditions'
fig,axs=plt.subplots(1,3,figsize=(15,4.5))
for (drift,beta),g in track.groupby(['drift','beta']):
    label=rf'$\\omega={drift:g},\\ \\beta={beta:g}$'
    axs[0].plot(g.t,g.error,marker='.',label=label)
    axs[1].plot(g.t,g.bound,marker='.',label=label)
    axs[2].plot(g.t,g.drift_term/g.gamma,marker='.',label=label)
axs[0].set_ylabel(r'$d_c$');axs[1].set_ylabel('Tracking bound');axs[2].set_ylabel(r'$d_t/\\gamma_t$')
for ax in axs:ax.set_xlabel('Observations')
axs[0].legend(ncol=2);save(fig,OUT/'theory_tracking');plt.show()
track.to_csv(OUT/'theory_tracking.csv',index=False)
''')
md(r'''## Directional uncertainty and certified bounds
Appendix I has equal modes and equal ball radii but different directional risks. Continuous tail integration is used for the reference thresholds; accurate moment quadrature alone does not certify a discrete tail quantile. Eq. (33) returns feasible and dual values, with a separate optimization gap.''')
code('''from lib.moments import quadrature, sample_bingham, chain_diagnostics
from lib.reference import rank_one_regions_gr13
from lib.theory import regional_bound
from lib.geometry import risk
A=np.diag([0.,-8.,-80.]);q=quadrature(A);reg=rank_one_regions_gr13(A)
P=np.diag([1.,0.,0.]);rows=[]
for i in (1,2):
    v=np.eye(3)[:,i];B=np.outer(v,v)
    ball=regional_bound(B,P,1-reg.radius**2,1)
    hpd=regional_bound(B,A,-reg.hpd_loss,1)
    rows.append(dict(direction=f'e{i+1}',risk=risk(q.mean,v),ball=np.sqrt(ball.upper),hpd=np.sqrt(hpd.upper),gap=max(ball.gap,hpd.gap)))
directional=pd.DataFrame(rows);display(directional)
assert np.max(abs(q.mean.diagonal()-[.92541154,.06829456,.00629390]))<1e-7
assert abs(reg.radius-.519344)<1e-6 and abs(reg.hpd_loss-3.157974)<1e-6
assert directional.gap.max()<1e-7
assert directional.hpd.iloc[0]>directional.ball.iloc[0] and directional.hpd.iloc[1]<directional.ball.iloc[1]
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
x=np.arange(2)
axs[0].bar(x,directional.risk);axs[0].set_xticks(x,[r'$e_2$',r'$e_3$']);axs[0].set_ylabel('Response risk')
axs[1].bar(x-.17,directional.ball,width=.34,label='Ball');axs[1].bar(x+.17,directional.hpd,width=.34,label='HPD')
axs[1].set_xticks(x,[r'$e_2$',r'$e_3$']);axs[1].set_ylabel('Projection bound');axs[1].legend()
save(fig,OUT/'theory_directional');plt.show()
''')
md('''## Numerical moments and EM
Three independent chains at `n=16, k=3` with spectral gaps up to 600 are checked in the required tests (12,000 retained draws and 4,000 burn-in sweeps per chain, real and complex). Here a separate small-dimensional reference reports moment error and chain diagnostics. Spectral moments, raw samples, credible tails, and normalizer estimates have different error budgets.

Frozen-prior conditional and marginal EM tests verify predictive likelihood ascent using deterministic E-steps, including the variance floor and pooled boundary. A separate changing-prior test verifies that historical data are not corrected twice. Sampled E-steps have no automatic ascent guarantee; case-study noise fits are labelled accordingly.''')
code('''chains=[sample_bingham(A,1,seed,draws=24000,burnin=3000) for seed in (80,81,82)]
diag=chain_diagnostics(chains)
display(diag)
assert diag['max_rhat']<1.05 and diag['min_ess']>100
moment_error=max(np.max(abs(c.mean-q.mean)) for c in chains)
assert moment_error<max(.007,5*diag['max_mcse'])
''')
md('''## Cost and scalability
Dense parameter storage and eigendecomposition are measured at dimensions 8 through 128, rank 3. These checks establish implementability at the tested sizes, not large-scale optimality. Posterior sampling, regional optimization, and EM add costs reported separately in the applications.''')
code('''scale=scalability_study();display(scale)
assert scale.finite.all() and np.max(abs(scale.trace))<1e-8
assert np.allclose(scale.parameter_bytes,8*scale.n**2)
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
axs[0].loglog(scale.n,scale.seconds,'o-');axs[0].set_ylabel('Seconds per update')
axs[1].loglog(scale.n,scale.parameter_bytes/1024,'o-');axs[1].set_ylabel('Parameter storage (KiB)')
for ax in axs:ax.set_xlabel('Dimension')
save(fig,OUT/'theory_scaling');plt.show()
''')
md('''## Validation certificate
Only reaching this cell writes a passing certificate for the exact numerical core. Example notebook generation and execution verify this hash. Empirical coverage and baseline performance in the case studies remain experimental outcomes, not assertions to force a desired ranking.''')
code('''import platform, scipy, matplotlib
assert core_hash(ROOT)==validated_hash, 'Validation sources changed during execution; rerun'
certificate=dict(status='passed',core_hash=validated_hash,tests=tests,
    seed=20,tracking_repeats=128,tracking_terminal=480,
    fixed_time_violations=int(terminal.violations),violation_upper_95=violation_upper,
    sampler=diag,moment_error=float(moment_error),dimensions=scale.n.tolist(),
    seconds=time.perf_counter()-started,
    versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),
    scope='Finite deterministic checks, stochastic diagnostics and stated scalability budgets; no universal guarantee')
(OUT/'validation.json').write_text(json.dumps(certificate,indent=2))
display(certificate)
''')
nb.cells=cells
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}}
nbf.write(nb,ROOT/'notebooks/00_theory.ipynb')
