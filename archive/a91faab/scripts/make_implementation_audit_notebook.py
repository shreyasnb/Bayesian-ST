"""Build a chronological supplement from completed guide checks and figures."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nbformat as nbf

REQUIRED_IDS = {'B1', 'B2a', 'B2b', 'B3', 'B4', 'B5', 'B6', 'B7a', 'B7b',
                'C1', 'C2', 'C3', 'C4', 'C5', 'C6a', 'C6b', 'S1'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_audit():
    from lib.implementation_validation import require_gate, verify_component
    require_gate(ROOT)
    base = ROOT/'results/implementation_audit'
    for name in ['information', 'information_counts', 'tracking', 'em', 'reliability']:
        verify_component(ROOT, 'paper', name)
    path = base/'paper/figures/manifest.json'
    figures = json.loads(path.read_text())
    if (figures.get('status') != 'completed' or figures.get('profile') != 'paper'
            or not REQUIRED_IDS.issubset(figures['figure_ids'])
            or figures.get('text_usetex') is not True or figures.get('figure_dpi') != 200
            or figures.get('savefig_dpi') != 200):
        raise RuntimeError('Complete the current paper guide figures before building this notebook')
    for mapping, folder in [('sources', ROOT), ('input_manifests', ROOT), ('files', path.parent)]:
        for name, expected in figures[mapping].items():
            if isinstance(expected, dict): expected = expected['sha256']
            if sha(folder/name) != expected: raise RuntimeError('Changed guide evidence: '+name)
    if figures['validation_sha256'] != sha(base/'validation/validation.json'):
        raise RuntimeError('Guide figures use a stale validation gate')
    if not (base/'findings.md').is_file() or not (base/'claim_manifest.json').is_file():
        raise RuntimeError('Write the completed findings and claim manifest before notebook generation')
    from scripts.summarize_implementation_audit import verify_report
    verify_report(ROOT)
    return figures


def md(nb, source): nb.cells.append(nbf.v4.new_markdown_cell(source.strip()))
def code(nb, source): nb.cells.append(nbf.v4.new_code_cell(source.strip()))

SETUP = '''from pathlib import Path
import sys, json, hashlib
from IPython.display import Image, Markdown, display
ROOT = Path.cwd() if (Path.cwd() / 'lib').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
from scripts.make_implementation_audit_notebook import verify_audit
from lib.plotting import plt
FIG = verify_audit()
BASE = ROOT / 'results/implementation_audit'
F = BASE / 'paper/figures'
assert plt.rcParams['text.usetex'] is True
assert plt.rcParams['figure.dpi'] == plt.rcParams['savefig.dpi'] == 200

def show(name):
    path = F / (name + '.png')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FIG['files'][path.name]
    display(Image(filename=str(path), width=1400))
    display(Markdown(FIG['captions'][name]))

print('Verified the current general gate, completed studies and final paper figures.')'''


def build():
    nb = nbf.v4.new_notebook()
    nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}}
    md(nb, r'''
# Computational checks for Bayesian subspace tracking

The [beamforming](01_array.ipynb) and [contact-normal](03_contact.ipynb) notebooks explain the applications. This supplement follows the computational chain behind them: normalized likelihood, information, finite-data curvature, tracking envelope, prediction, block EM, and reliability under changed noise or motion.

Figure IDs follow the implementation guide. The [claim manifest](../results/implementation_audit/claim_manifest.json) maps each requested check to its routine, metric, figure, assumptions and observed status. Exact aliases of main figures are identified in the figure manifest; an alias is not another independent experiment. The [audit note](../docs/implementation_guide_audit.md) identifies which requirements existed and which were added, including unavailable manuscript-source checks.

The guide names `main.tex` and `supplementary.tex`, which were not present in the supplied checkout. Explicit formulas in the guide and the supplied draft are the available specification; no claim of inspecting missing source text is made. Negative, inconclusive and numerically unresolved results remain in the [findings](../results/implementation_audit/findings.md) and raw tables.
''')
    code(nb, SETUP)
    md(nb, r'''
## B2a; C2. From a likelihood to expected information

The real marginal model has $F=L(\tau^2)^2/[s(s+\tau^2)]I$. Proper-complex data double this coefficient in real/imaginary tangent coordinates and double the tangent dimension. The conditional calculation uses recorded inputs; it does not reconstruct an unobserved input from a measured response.

The first panel compares analytical information with empirical score covariance. The second checks expected likelihood Hessians and exact Gaussian KL curvature at several finite-difference steps. The third compares the KL contours and their local Fisher approximation in the same tangent chart. Discounted Hessians use weights $w_j$, whereas discounted score covariance uses $w_j^2$; those separate tests remain in the tables.
''')
    code(nb, '''import numpy as np
from lib.beamforming_theory import passive_fisher_information
F_real = passive_fisher_information(3, 1, s=0.2, q=1.3, observations=4, complex_data=False)
F_complex = passive_fisher_information(3, 1, s=0.2, q=1.3, observations=4, complex_data=True)
assert F_complex.shape[0] == 2 * F_real.shape[0]
assert np.allclose(np.diag(F_complex), 2 * F_real[0, 0])
print('Real tangent dimension:', len(F_real), '; complex real-coordinate dimension:', len(F_complex))
print('Fisher coefficient:', F_real[0, 0], '(real),', F_complex[0, 0], '(proper complex)')''')
    code(nb, "show('B2a_C2_information')")
    md(nb, r'''
## B2a. Check observation count and signal-to-noise ratio separately

The next panel uses a separate declared cohort. Each episode sums its likelihood scores over $L$ independent observations; empirical covariance is then estimated across independent episodes. The reference Fisher matrix scales with $L$. A Monte Carlo draw count is not relabeled as an observation count, and posterior-curvature samples are not used as substitutes for expected information.
''')
    code(nb, "show('B2a_count_error')")
    md(nb, r'''
## B2b. Expected information is not realized posterior curvature

The marginal likelihood has isotropic expected information at its generating state. A finite collection of measured vectors can still produce unequal selected/complementary eigenvalue gaps. The realized negative Bingham log-density Hessian has twice those gaps in the real tangent chart. Flat directions after a small number of observations are retained.

The display fixes $N=3$ and the first declared sensor variance, then changes only the observation count. Boxes describe the spread of realized curvature eigenvalues across datasets, rather than uncertainty in a fitted regression coefficient. Posterior-mode and generating-state tangent frames are distinct, so this is an invariant spectral comparison.
''')
    code(nb, "show('B2b_posterior_curvature')")
    md(nb, r'''
## B3; C4. When does the modal tracking theorem say something?

For real marginal data, $\eta=\tau^2/[2s(s+\tau^2)]$, $h=N\log9+\log(2/\delta)$ and $\gamma_t=\eta\tau^2m_t$. The guide uses the fluctuation allowance $4\eta(s+\tau^2)(\sqrt{b_th}+h)$, with weighted aggregate drift and the initial spectral-gauge radius. The proper-complex extension changes both $\eta$ and $h$. Conditional bounds use their separate recorded-input formula.

The reported envelope is $2\sqrt{k}\epsilon_t/\gamma_t$ only when $\epsilon_t<\gamma_t/2$. A valid sufficient condition can still give an envelope above the geometric distance ceiling. These calculations use fixed true scales, deterministic paths and fixed memory; the true gap and drift are evaluation diagnostics, not an online confidence radius.
''')
    code(nb, "show('B3_C4_modal_bounds')")
    md(nb, r'''
## B3; C4. More time does not always fix a memory-limited bound

At $\beta<1$, the guide's stationary ratio has a positive limit:
$4(1+s/\tau^2)[\sqrt{h(1-\beta)/(1+\beta)}+h(1-\beta)]$.
If this limit is too large, increasing the horizon alone cannot make the sufficient condition informative. The phase diagram varies the declared drift and memory independently. Fixed-time violation counts, eligible episode counts and binomial intervals are kept separately; zero observed violations is not proof of a uniform-in-time guarantee.
''')
    code(nb, '''from lib.guide_tracking import marginal_preflight
for beta in [1.0, 0.99]:
    check = marginal_preflight(n=3, s=0.2, tau2=1.0, t=16384, beta=beta)
    print('beta =', beta, ': epsilon/gamma =', round(check['ratio'], 3),
          '; gap condition =', check['valid'],
          '; permanently vacuous =', check['permanently_vacuous'])''')
    code(nb, "show('B3_C4_validity')")
    md(nb, r'''
## B5. Compare decisions on fresh measurements

The posterior-mean covariance minimizes posterior expected output power under unit desired gain. It does not maximize expected SINR. The paired held-out measurements below are independent of training and shared by the competing decisions. A dot is one independent training episode, after averaging its held-out observations.

The matched-isotropy control and validation-loaded covariance remain in the full comparison tables. Matching modal overlap preserves trace and expected squared chordal loss, but does not preserve directional information. Matching a covariance alone also does not specify a predictive probability law.
''')
    code(nb, "show('B5_heldout_decisions')")
    md(nb, r'''
## B5. Keep changing motion, noise and power as separate interventions

The new tracking experiment uses snapshots alone and identity transport. It retains static, smoothly drifting and abruptly changing signatures, then changes sensor noise and interference power separately. The left panel compares a strong covariance decision with the posterior mean; the right holds the modal decision rule fixed while changing only its independently selected memory. A memory benefit is not an anisotropy benefit.

Every declared condition remains, including losses. Boxes contain independent paired episodes at the final prescribed checkpoint. The complete paired intervals and earlier checkpoints remain in the tables.
''')
    code(nb, "show('B5_tracking_decisions')")
    md(nb, r'''
## B6. State coverage, response risk and predictive density

Prior-predictive calibration first draws the state from the stated prior and then draws its data. Fixed-state and drifting-state repeated-data inclusion answer different questions. State regions, clean-response intervals and noisy predictive intervals are recorded separately, with episode-level uncertainty.

Scores are evaluated before assimilation from the full normalized singleton likelihood and the normalizer ratio. A Gaussian with covariance $sI+\tau^2\overline P$ matches a predictive second moment; it is a separate density from the projector mixture. Neither numerical accuracy of the mean nor a credible-mass identity guarantees fixed-state scalar coverage.
''')
    code(nb, "show('B6_beam_calibration')")
    md(nb, r'''
## C5. Apply the same probability distinctions to contact

The conditional model uses observed requests and directly measured instantaneous velocities with independent Gaussian sensor noise. Clean contact-response risk is evaluated on a fresh request. The calibration displays use the same conventions as beamforming, so observed differences can be traced to the observation model rather than a changed definition of coverage.

Every repeated-time quantity is averaged within an episode before comparing independent episodes. Validation-selected width adjustments and their independent test coverage remain in the tables; adjusting a width on test truth would invalidate that comparison.
''')
    code(nb, "show('C5_contact_calibration')")
    md(nb, r'''
## B7a; C6a. Fit one stationary block under one boundary prior

The revised block EM objective assumes one constant subspace and a boundary prior excluding the entire block. Every E-step reconstructs $A_\partial+\sum_jB_j(\theta)$ from that same boundary. Full normalized evidence restores scalar likelihood terms and trace-gauge constants. This differs from the historical product under separately recorded single-observation priors.

Marginal updates respect ordered noise/signal variances. Conditional noise updates include posterior subspace uncertainty; replacing the projector by its mean inside a squared residual omits that term. Several positive interior starts are compared by normalized evidence. With a uniform prior, a zero signal-variance start can be absorbing and is kept as a declared boundary diagnostic.

Each displayed curve is shifted by its own initial evidence, so its final height does not rank starts. Start selection uses the absolute normalized evidence recorded in `fits.csv`.
''')
    md(nb, 'The small block below illustrates the library API and trace gauge. It is a deterministic replay, separate from the independent Monte Carlo cohorts shown in the figure.')
    code(nb, '''from lib.block_em import BlockData, block_posterior, fit_block_em
U = np.array([[1.0, 0.1, 0.0], [0.8, -0.1, 0.1], [1.1, 0.0, -0.1],
              [-0.9, 0.1, 0.1], [-1.2, -0.1, 0.0], [0.7, 0.0, 0.1]])
A_boundary = np.diag([0.3, -0.2, -0.1])
block = BlockData(U, k=1, boundary_prior=A_boundary)
shifted = BlockData(U, k=1, boundary_prior=A_boundary + 7 * np.eye(3))
post = block_posterior(block, s=0.2, tau2=1.0)
post_shifted = block_posterior(shifted, s=0.2, tau2=1.0)
assert np.allclose(post.mean, post_shifted.mean, atol=1e-9)
assert abs(post.log_evidence - post_shifted.log_evidence) < 1e-8
fit = fit_block_em(block, s=0.2, tau2=1.0, iterations=6)
assert np.min(np.diff(fit.objectives)) >= -1e-8
print('Gauge evidence error:', abs(post.log_evidence - post_shifted.log_evidence))
print('Accepted/rejected steps:', fit.accepted, '/', fit.rejected)
print('Block log-evidence increase:', fit.objectives[-1] - fit.objectives[0])''')
    code(nb, "show('B7a_C6a_block_evidence')")
    md(nb, r'''
## B7b; C6b. Separate sensor changes from model changes

Evidence ascent for a fixed stationary block does not imply monotone online prediction scores, convergence under drift, or a learned forgetting factor. The next intervention changes sensor variance while holding motion fixed, then changes motion while holding sensor variance fixed.

Known scales use the generating variances. Misspecified scales keep $s=0.05$ and, for beamforming, $\tau^2=0.2$ fixed. Block EM rebuilds the fixed-boundary block objective, while residual fitting uses a modal point estimate and omits subspace uncertainty.

Inflation of a fitted noise parameter under motion mismatch does not establish recovery of physical sensor noise. Plug-in-scale intervals condition on those scales and omit their uncertainty. Adaptive runs are therefore separate from the fixed-scale theorem checks above.
''')
    code(nb, "show('B7b_C6b_scale_adaptation')")
    md(nb, r'''
## C5. Keep changed mechanics and sensors visible

The ideal plant independently solves its KKT system. Maintained bilateral contact, known damping, negligible inertia and the noise model are assumptions of the conditional likelihood. Friction, detachment, anisotropic mobility and position-differenced velocity noise change that model.

Known anisotropic damping requires the scaled-coordinate projection and correctly transformed noise. Differencing independent noisy positions produces temporally correlated velocity errors. Those mechanical and covariance identities are checked directly; their mismatched counterparts remain diagnostic experiments, not claimed extensions of the isotropic theorem.
''')
    code(nb, "show('C5_model_mismatch')")
    md(nb, r'''
## S1. Computational feasibility has a defined scope

The existing [scaling panels](../results/implementation_audit/paper/figures/S1_native_scaling.pdf) separate parameter corrections, eigenspaces, native moments and optional uncertainty reports. Native rank-one/codimension-one calculations do not validate arbitrary-rank integration. Serial timing runs have their own fixed streams, warm-up and repetition budget; reference-quality integration has separate accuracy settings.

The main applications and their raw results are retained. Read the [results note](../results/implementation_audit/findings.md), [claim manifest](../results/implementation_audit/claim_manifest.json), [audit protocol](../docs/implementation_guide_audit.md) and [reproduction commands](../results/implementation_audit/README.md) for numerical tolerances, Monte Carlo precision, limitations and the distinction between passed checks and inconclusive performance comparisons.
''')
    return nb


def main():
    verify_audit()
    nb = build()
    path = ROOT/'notebooks/04_implementation_audit.ipynb'
    nbf.write(nb, path)
    print('Built', path.relative_to(ROOT), len(nb.cells), 'cells; execute from a clean kernel.')


if __name__ == '__main__':
    main()
