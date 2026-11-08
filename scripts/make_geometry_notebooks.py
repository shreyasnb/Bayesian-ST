"""Build the two paper narratives after current validation and figure checks.

This command builds only. scripts/execute_geometry_notebooks.py executes both
notebooks and records their execution provenance. No simulation occurs here.
"""
from pathlib import Path
import hashlib,json,sys
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lib.simulation_validation import require_gate,verify_component,verify_legacy,gate_path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_summary(gate):
    """Require findings derived from these final paper figures and inputs."""
    folder=ROOT/'results/simulation_redesign'
    summary=json.loads((folder/'summary_manifest.json').read_text())
    if (summary.get('profile')!='paper' or summary.get('status')!='completed'
            or summary.get('paper_evidence') is not True or summary.get('figures_verified') is not True):
        raise RuntimeError('Generate the completed paper summary after the final figures first')
    if (summary.get('gate_sha256')!=sha(gate_path(ROOT))
            or summary.get('science_hash')!=gate['science_hash']):
        raise RuntimeError('Summary validation provenance is stale')
    producer=summary['generator']
    if sha(ROOT/producer['path'])!=producer['sha256']:
        raise RuntimeError('Summary generator changed')
    for mapping in ['source_hashes','verified_input_hashes','outputs']:
        for name,expected in summary[mapping].items():
            if sha(ROOT/name)!=expected:raise RuntimeError('Stale summary evidence: '+name)
    for name in ['beamforming_summary.md','contact_summary.md']:
        if len((folder/name).read_text().split())>220:
            raise RuntimeError('Keep each application summary below 221 words')
    return summary


def verified():
    gate=require_gate(ROOT);verify_legacy(ROOT)
    for component in ['geometry','beamforming','contact','scaling']:verify_component(ROOT,'paper',component)
    base=ROOT/'results/simulation_redesign/paper';manifest=json.loads((base/'figures/manifest.json').read_text())
    if manifest['profile']!='paper' or manifest['text_usetex'] is not True or manifest['figure_dpi']!=200 or manifest['savefig_dpi']!=200:
        raise RuntimeError('Generate current paper figures with the required protocol first')
    for mapping,folder in [('sources',ROOT),('input_manifests',ROOT),('files',base/'figures')]:
        for name,expected in manifest[mapping].items():
            if sha(folder/name)!=expected:raise RuntimeError('Stale figure input/export: '+name)
    verify_summary(gate)
    from scripts.make_implementation_audit_notebook import verify_audit
    verify_audit()
    return base,manifest


def notebook():
    n=nbf.v4.new_notebook();n.cells=[]
    n.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}}
    return n

def md(n,text):n.cells.append(nbf.v4.new_markdown_cell(text.strip()))
def code(n,text):n.cells.append(nbf.v4.new_code_cell(text.strip()))

SETUP='''from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd
from IPython.display import Image, display, Markdown
ROOT = Path.cwd() if (Path.cwd() / 'lib').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
from lib.plotting import plt
from lib.simulation_story_visuals import load_studies
from lib.simulation_validation import verify_legacy
from scripts.make_geometry_notebooks import verify_summary
D = load_studies(ROOT, 'paper')
from lib.simulation_validation import require_gate
SUMMARY = verify_summary(require_gate(ROOT))
BASE = D['base']
F = BASE / 'figures'
FIG = json.loads((F / 'manifest.json').read_text())
assert FIG['profile'] == 'paper'
assert plt.rcParams['text.usetex'] is True
assert plt.rcParams['figure.dpi'] == plt.rcParams['savefig.dpi'] == 200
for mapping in ('sources', 'input_manifests'):
    for name, expected in FIG[mapping].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected

def show(name):
    path = F / (name + '.png')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FIG['files'][path.name]
    display(Image(filename=str(path), width=1400))
    display(Markdown(FIG['captions'][name]))

print('Verified the current theory gate, four paper components, figures and findings.')'''


def beamforming():
    n=notebook()
    md(n,r'''
# Beamforming with an uncertain spatial subspace

**What does a distribution add beyond its modal subspace, and when is it worth computing?**

We start with the observed array snapshots, make the posterior geometry visible, then examine the beamformer and independent benchmarks. The main study uses native complex lines at $N=8$ and $N=16$; the final experiment measures rank-one computations through $N=128$. A three-dimensional drawing of a high-dimensional posterior will always be labelled as a section, rather than presented as the whole state space.

This notebook follows the attached draft's theory and notation. Its proper-complex beamforming likelihood is a stated extension of the draft's real marginal model. The earlier executed beamforming study, including its different stochastic-process and parameter-inference experiments, is preserved in the [historical notebook](../results/simulation_redesign/archive/01_array_2a8697c.ipynb).
''')
    code(n,SETUP)
    md(n,r'''
## B1. What is measured?

The array observes $u_t=h_tz_t+e_t$, with $\|h_t\|=1$, $z_t\sim\mathcal{CN}(0,\tau^2)$ and $e_t\sim\mathcal{CN}(0,sI)$. Integrating the unknown coefficient gives $u_t\mid\mathcal X_t\sim\mathcal{CN}(0,sI+\tau^2P_{\mathcal X_t})$. The estimator receives only the snapshots. The desired unit steering signature $a$ is supplied later, for the constraint $w^Ha=1$.

The physical array is a half-wavelength ULA. Desired/interferer separations $3^\circ,10^\circ,30^\circ$ span nearby and well-separated arrivals; a fixed Haar spatial signature is the matched ambient-prior control. All settings, independent validation/test seeds and displayed trajectories were fixed before test comparisons. Powers are variances. The desired signature is absent from posterior correction.
''')
    code(n,"show('beamforming_01_setup')")
    md(n,r'''
## B4. What does the posterior look like?

For proper-complex data, $B_t=\tau^2u_tu_t^H/[s(s+\tau^2)]$, with **no factor $1/2$**. Algorithm 1 adds this statistic once per observation and recenters the parameter to trace zero. A uniform prior and stationary $\beta=1$ make the mode the leading sample-covariance eigenspace. We write the modal projector $P_{\widehat{\mathcal X}_t}$ and full mean $\overline P_t^+=\mathbb E[P_{\mathcal X_t}\mid\mathcal D_t]$ exactly as in the draft.

The sphere below evaluates the actual density on $h=U_3x$, $x\in S^2$, where $U_3$ contains the mode and two complementary eigenvectors. It is a **real-coefficient section of a complex $N=8$ law**. The mean spectrum and all eigengaps reveal the directions that this section omits. Complex $N=2$ would map to an axially symmetric Bloch-sphere density and cannot demonstrate general directional anisotropy.
''')
    code(n,"show('beamforming_02_posterior')\nassert D['beamforming']['shape'].pca_mode_error.max() < 1e-7")
    md(n,r'''
## B4. Can the mode and a radius determine a decision?

Here two selected-data posteriors have identical modes, spectra and **every** centered chordal-distance quantile. A unitary transformation fixes the mode and interchanges two complementary directions relative to the desired steering vector. Their directional mean projectors therefore differ.

For the posterior covariance $\overline R=sI+\tau^2\overline P_t^+$, the distortionless expected-power action is $w_\star=\overline R^{-1}a/(a^H\overline R^{-1}a)$. Its conditional excess identity is $(w-w_\star)^H\overline R(w-w_\star)\ge0$. The axial control $q_{\kappa P_{\widehat{\mathcal X}_t}}$ matches the modal mean overlap, hence $r_{t,\mathrm{rms}}$, without using test truth. Matching this one number does not preserve orientation-dependent predictions.

This controlled calculation illustrates information that one radius loses. It is not a claim that the Bayesian action wins on every fixed true covariance. That empirical question comes next.
''')
    code(n,"show('beamforming_03_directional')")
    md(n,r'''
## B5. Does it improve repeated-training performance?

Ledoit--Wolf covariance shrinkage adapts to each observed dataset. Loaded covariance and GROUSE receive independent validation choices; a regularized spectral control uses a continuously selected coefficient. These references test whether a gain over the mode can be explained by cheaper regularization. Every method sees the same training and held-out snapshots.

The paired points are independent trajectories, not steering-grid locations or posterior samples. Below the diagonal favors $\overline P_t^+$. The distribution of paired differences retains both gains and losses. The objective is output power, not expected SINR or expected log-SINR. Pointwise paired intervals and the complete comparison tables are descriptive comparisons, with no multiple-comparison guarantee.
''')
    code(n,"show('beamforming_04_benchmarks')")
    md(n,r'''
## B4. What is uncertain, and where does geometry matter?

The left panel is the actual weight beampattern $|w^Ha(\theta)|^2$. The middle is the different covariance functional $s+\tau^2a(\theta)^HP_{\mathcal X}a(\theta)$, with central 50% and 95% posterior intervals. At the exact true interferer direction this functional is at its support maximum, so a finite continuous equal-tail interval excludes the endpoint. This is not a failure of the posterior sampler; arbitrary scalar intervals are not uniform frequentist confidence bands.

The small complete separation/horizon grid asks where the posterior mean improves on adaptive covariance shrinkage, including losses. It uses every predeclared physical separation at the stated noise level. A posterior action minimizes its conditional model-based loss; a fixed physical geometry can still favor a competing decision rule.
''')
    code(n,"show('beamforming_05_responses')")
    md(n,r'''
## S1. What scales, and what costs extra?

The dense correction has $O(N^2)$ storage and arithmetic. Eigenvector extraction, posterior moments and optional credible reports are separate computations. The scaling study holds spectral span fixed when varying dimension, and retains a separate data-generated case. Rank-one/codimension-one means are not general-rank Grassmann calculations.

Serial warmed timing repeats randomize operation order and record CPU and wall clocks. Whole-stream rows assimilate 64 observations and return one endpoint output; nested component clocks must not be added twice. A mean is needed when choosing a posterior expected-power action, while a costly report can be scheduled less often. Numerical caps and unresolved targets remain in the saved tables.
''')
    code(n,"show('beamforming_06_scaling')")
    md(n,'## What the independent experiments show')
    code(n,"display(Markdown((ROOT / 'results/simulation_redesign/beamforming_summary.md').read_text()))")
    md(n,r'''
## Interpretation and reproducibility

The distribution is useful when its directional predictions or expected decisions improve the actual task enough to justify their cost. The mode already duplicates covariance PCA under matched weights. Conditional expected-power optimality does not promise a universal empirical win, and a fitted ambient prior need not exploit a particular ULA geometry as well as validation-tuned covariance regularization.

Read the [recorded findings](../results/simulation_redesign/findings.md), [protocol and reproduction commands](../results/simulation_redesign/README.md), and [notation and claim scope](../docs/simulation_redesign_notation.md). Complete independent comparisons and numerical flags remain in the component tables. Historical moving-array, unknown-parameter and rank-three results are preserved under their original source manifests; they are not silently presented as results of this redesigned protocol. The [implementation supplement](04_implementation_audit.ipynb) adds the guide checks for Fisher information, revised modal bounds, block EM, pre-observation scores and model mismatch. Its figure IDs and [claim manifest](../results/implementation_audit/claim_manifest.json) identify the evidence and assumptions. No manuscript text is changed.
''')
    return n


def contact():
    n=notebook()
    md(n,r'''
# Tracking a contact-normal line

**How do known requests identify an unknown constraint, and what does its directional uncertainty change?**

A contact normal is an **unoriented line** $[n_t]$: $n_t$ and $-n_t$ describe the same tangent plane. We first connect force balance to the draft's conditional likelihood, then inspect genuine antipodal Bingham densities on the unit sphere. Independent benchmarks separate the value of the likelihood, the mean, and global distribution shape. Higher-dimensional examples represent generalized linear constraints, rather than pretending a high-dimensional posterior is a 3D sphere.

The physical model is maintained ideal bilateral contact with known isotropic resistance. It has no friction cone, detachment, impact or unknown mobility metric. The earlier executed contact experiments remain [archived](../results/simulation_redesign/archive/03_contact_2a8697c.ipynb) with their original certificates and assumptions.
''')
    code(n,SETUP)
    md(n,r'''
## C1. From applied force to the measured response

With known isotropic resistance $dI$, the free requested velocity is $v_t=F_t/d$. The plant independently solves force balance and $n_t^\top\bar u_t=0$, giving the closest admissible velocity $\bar u_t=P_{\mathcal X_t}v_t$, $\mathcal X_t=n_t^\perp$. The measurement is $u_t=\bar u_t+e_t$, $e_t\sim\mathcal N(0,sI)$.

The estimator receives the request and noisy response, not the surface normal or constraint projector. A measured wrist orientation determines the supplied frame transport $G_t$ and is shared with every comparator. Signs of normals are aligned only for display/scoring. The force balance in this first figure explains the geometry; it does not provide the hidden normal to estimation.
''')
    code(n,"show('contact_01_mechanics')")
    md(n,r'''
## C2. Known requests create directional information

The conditional correction is $B_t=(u_tv_t^\top+v_tu_t^\top-v_tv_t^\top)/(2s)$. The negative term matters: a blocked request $u_t\approx0$ still informs the normal. Identical-likelihood spectral least squares is the modal reference. PCA of residuals uses the known requests as well; velocity-only PCA would discard available information and would be a weaker comparison.

For a tangent-plane parameter $A_t^+$, the normal-line density is $q_{-A_t^+}([n])$, and the plane mean is $\overline P_t^+=I-\mathbb E[nn^\top]$. The mean of a signed normal itself is zero by antipodal symmetry. The next figure compares two equal-energy request programs at the same reporting time: cyclic coordinate axes and requests near $e_1$.

The expected Fisher eigenvalues follow the draft's $F_t=(C_{11,t}\otimes I+I\otimes C_{22,t})/s$. They describe expected information at the stated state, not the realized posterior curvature. A small input-Gram eigenvalue alone does not imply weak local information; the direction relative to the normal matters.
''')
    code(n,"show('contact_02_posterior')")
    md(n,r'''
## C3. The same mode and radius can hide different commands

The draft's deterministic example has $A^{(1)}=\operatorname{diag}(0,-8,-80)$ and $A^{(2)}=\operatorname{diag}(0,-80,-8)$. Here these are **normal-line** parameters; their dual tangent-plane parameters are $-A^{(i)}$. The modes and entire centered chordal-distance distributions agree, but the mean-response risks for $e_2$ and $e_3$ differ by about tenfold.

For contact motion, Eq. (34) predicts $\overline P_t^+v$ and gives clean-response risk $v^\top[\overline P_t^+-(\overline P_t^+)^2]v$. The full vector covariance needs second moments; this scalar total risk uses the mean only. Eq. (36) chooses $v_t^{\rm B}=[I+\lambda(I-\overline P_t^+)]^{-1}v_t^{\rm des}$. The two beliefs below yield different commands for the same desired vector, even though one mode and one radius cannot distinguish them.

This is an expected-penalty command, not a guaranteed feasible controller. The HPD and distance ball each have posterior mass 0.95; equal mass does not make one tighter in every direction. Bounds in the tables concern the blocked normal projection $\|nn^\top v\|$, rather than tangent-plane speed.
''')
    code(n,"show('contact_03_directional')")
    md(n,r'''
## C3. Do the mean and its directional structure improve actual loss?

The fresh stationary benchmark uses independent Haar normal lines, equal-energy requests and known sensor variance. References include the identical-likelihood point estimate, matched axial uncertainty, validation-regularized spectral response $\widetilde M_t v$, ridge regression, residual PCA, and a local Gaussian approximation with its validity flags. The axial tangent-plane law $q_{\kappa P_{\widehat{\mathcal X}_t}}$ is equivalently $q_{\kappa P_{[\widehat n_t]}}$ on normal lines; it matches the normal modal overlap and $r_{t,\mathrm{rms}}$.

The paired response plots and soft-command costs answer different questions. Ridge regression is eligible as a response-map estimate; it is not inserted into a formula requiring a positive projector mean. The local $\mathcal N(0,H_t^{-1})$ approximation is pushed through the modal tangent exponential; it is different from the recursive extended Kalman filter used for physical tracking. Broad or unresolved local charts remain flagged, rather than being treated as global distributions.
''')
    code(n,"show('contact_04_benchmarks')")
    md(n,r'''
## C1; C3. Track the normal as the contact and coordinates change

On the physical sphere the admissible plane changes even in the world frame. Wrist motion adds a measured coordinate change, handled by $A_t^-=\beta G_tA_{t-1}^+G_t^\top$. It does not reveal the changing surface geometry. Each practical method receives the same measured rotations and independently selects its declared memory/noise settings on validation data.

The response distributions are **post-correction, current-state** predictions for a fresh request at the present contact, not forecasts of a future surface. Their central intervals are posterior statements. Finite numerical mass checks do not establish frequentist coverage under drifting states. The theorem's fixed-time modal bound pays for residual drift and can fail its sufficient condition; invalid bounds stay invalid in the tables. Known frame transport, improved memory and distribution shape are different effects.
''')
    code(n,"show('contact_05_tracking')")
    md(n,r'''
## S1. Beyond three coordinates

The same conditional model describes an unknown codimension-one linear constraint in $\mathbb R^N$. These generalized velocity examples use $N=8,16,32$ and $t=4N$ requests. No robot Jacobian is hidden from a structured competitor, and these are not claimed as a physical high-DOF robot simulation.

For isotropic test requests, average response MSE contains a factor $1/N$. It can decrease merely because one constraint occupies a smaller fraction of the ambient space. The figure therefore shows full response-map Frobenius error instead. The separate timing cohort has a fixed 64-observation horizon and uses native rank-one complement calculations. It measures computational scaling, not mechanical-controller latency or an arbitrary-rank solver.
''')
    code(n,"show('contact_06_higher_dimension')")
    md(n,'## What the independent experiments show')
    code(n,"display(Markdown((ROOT / 'results/simulation_redesign/contact_summary.md').read_text()))")
    md(n,r'''
## Interpretation and reproducibility

The full mean predicts an expected velocity; it is generally not one feasible contact projector. The matched same-mode/spread and same-parameter Gaussian controls isolate information beyond a point estimate, while the physical comparison also includes dynamics and numerical approximation. Benefits must be supported by independent response/action loss, uncertainty width/score or global geometry, with computation included. A known-force likelihood or supplied frame rotation can benefit point methods too.

The fixed-state visual story uses a coordinate-aligned true normal and coordinate requests. Some scalar responses then lie at support boundaries, so finite central intervals can exclude them even under the correct likelihood. This does not contradict posterior mass or a state-region coverage result. The independent Haar-normal cohort and the single displayed physical trajectory have different probability interpretations.

Read the [findings](../results/simulation_redesign/findings.md), [reproduction protocol](../results/simulation_redesign/README.md), and [notation/assumptions](../docs/simulation_redesign_notation.md). Noise is fixed in this notebook's original comparison cohorts. The added [implementation supplement](04_implementation_audit.ipynb) separately implements and checks fixed-boundary-block EM; it distinguishes that objective from the historical per-observation-prior routine. These fixed-scale comparisons do not claim a scale-learning advantage. Historical temperature, learned-motion and stochastic-process studies remain separately labelled in the archived notebook. No code or inference is rerun merely to adjust a figure layout.
''')
    return n


def main():
    verified()
    for filename,function in [('01_array.ipynb',beamforming),('03_contact.ipynb',contact)]:
        n=function();nbf.write(n,ROOT/'notebooks'/filename)
        print('Built',filename,len(n.cells),'cells; execute separately after review.')

if __name__=='__main__':main()
