"""Render the draft-5 study from verified frozen outputs; never resimulate."""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lib.simulation_story_visuals import (load_studies,beam_story,beam_setup,
    complex_posterior_section,beam_directional,beam_benchmarks,beam_responses,
    contact_dynamic_story,contact_mechanics,contact_posterior,paper_geometry,
    contact_benchmarks,contact_tracking,higher_contact,scaling_figure)
from lib.plotting import plt
SOURCES=['lib/plotting.py','lib/distribution_geometry_visuals.py',
    'lib/simulation_story_visuals.py','scripts/plot_geometry_studies.py']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

CAPTIONS={
'beamforming_01_setup':r'Passive interference-plus-noise snapshots identify a spatial line. The desired steering signature $a$ enters only the distortionless decision. The latent coefficients and true interferer enter generation and scoring, never estimation. Powers $s$ and $\tau^2$ are variances. The displayed ULA trajectory was fixed in advance.',
'beamforming_02_posterior':r'This is an actual $N=8$ proper-complex posterior at $t=16$. The sphere is the real-coefficient section $h=U_3x$, $x\in S^2$, through its mode and the least/most concentrated complementary eigenvectors. Colors are $q_A([U_3x])/q_A(\widehat{\mathcal X})$, not a normalized sphere marginal. State-region curves are intersections, not 95%-mass surface regions. All complementary gaps and all mean eigenvalues are shown alongside it.',
'beamforming_03_directional':r'Selected-data complex beliefs have the same mode, spectrum and entire centered chordal-distance distribution, with two complementary directions interchanged relative to the fixed desired steering vector. Both spheres use one fixed $U_3$ frame; gray curves are equal-radius intersections. Conditional expected-power excess is evaluated under each stated belief and is not a repeated-training performance claim.',
'beamforming_04_benchmarks':r'Each paired point is an independent held-out trajectory at $s=0.2$, $t=16$. Ledoit--Wolf adapts its covariance regularization to each dataset. The first two panels use Haar spatial signatures at $N=8$ and $N=16$; the final panel is the $N=8$, $10^\circ$ ULA comparison. Positive reference-minus-mean power differences favor the posterior mean. All other conditions and comparators remain in saved tables; no universal ranking is assumed.',
'beamforming_05_responses':r'The actual weight beampattern and the uncertain covariance scanning response are different quantities. The middle bands are pointwise central posterior 50% and 95% intervals, not simultaneous confidence bands. At the exact interferer direction the truth is the support maximum, so a finite central interval excludes that endpoint. The small grid retains every predeclared ULA separation and horizon at $N=8$, $s=0.2$; open circles mark pointwise paired intervals crossing zero.',
'beamforming_06_scaling':r'Serial, warmed, randomized-order engineering timings use a common fixed spectral span of 80 to separate dimension from concentration. Correction costs, eigenspace extraction and native means have different scopes; the mean already includes its eigensystem, so clocks are not additive. The real mean uses a common rank-one complement backend rather than the specialized 3D contact integrator. Storage is exact array bytes. More concentrations, endpoint streams, numerical flags and optional report costs are retained in the scaling tables; these points are not complexity proofs or worst-case latency guarantees.',
'contact_01_mechanics':r'The plant solves ideal maintained bilateral contact with known isotropic resistance. Its clean velocity is the Euclidean projection of the requested free velocity. True geometry appears only in this explanatory rendering and in scoring. The estimator observes the request, noisy realized response and measured wrist orientation. The normal is an unoriented line $[n]$: $n$ and $-n$ describe the same constraint plane.',
'contact_02_posterior':r'Actual post-correction normal laws $q_{-A_t^+}$ at $t=8$ for two equal-energy request programs. Colors are densities relative to normalized invariant measure $d\Omega/(4\pi)$; antipodal points describe one plane. The expected conditional Fisher information is evaluated at the true fixed normal for explanation, not supplied to the estimator. It is different from the realized posterior curvature. Solid dark outlines are state HPD boundaries; gray outlines are centered chordal balls.',
'contact_03_directional':r'The real $\mathrm{Gr}(1,3)$ example in the draft is interpreted as a contact-normal line: the tangent-plane mean is $I-\mathbb E[nn^\top]$. Both beliefs have the same mode and every centered distance quantile. The right panel uses the same $v_t^{\rm des}$ and $\lambda=20$ in Eq. (36); $v_t^{\rm B}$ is an expected-penalty action, not a guaranteed feasible contact controller. Region bounds in the tables concern the blocked normal component $\|nn^\top v\|$, not admissible-plane speed.',
'contact_04_benchmarks':r'Paired independent stationary trajectories with requests near $e_1$, $t=8$. Response losses and Eq. (35) command costs are in $(\mathrm{mm}/\mathrm{s})^2$. Ridge regression is a strong response-map reference and is not assigned an invalid projector-based command. Matched axial uncertainty preserves the modal normal and its expected squared chordal distance. The command panel uses $\lambda=10$. Complete baselines, local-Gaussian validity flags and both request programs remain available.',
'contact_05_tracking':r'The normal line evolves along an independently generated physical sphere. Measured wrist-frame rotations are supplied equally to every practical method; they do not reveal the changing surface normal. Signs are aligned only for this display/scoring. The middle panel averages modal state errors over independent trajectories; $\overline P_t^+$ is not assigned a modal normal. The response violins are post-correction, current-state counterfactual predictions for $v^\star=v_0e_1$, at predeclared report times. Visual particles are resampled integration nodes; checked central 50%/95% quantiles come from continuous integration. This is not a future forecast or a repeated-data coverage claim. Each dynamic method selects its own validation memory; the same-$A$ shape controls are separate.',
'contact_06_higher_dimension':r'These are generalized single linear maintained constraints, not a fabricated high-DOF robot with known Jacobians withheld from a comparator. The accuracy cohort uses $t=4N$ requests and plots Frobenius response-map error, avoiding the artificial $1/N$ reduction in average isotropic-query MSE. The separate serial timing cohort uses 64 dimensionless observations and one endpoint output; it is not mechanical-controller latency. Generic-rank moments are not substituted for the native rank-one/complement backend.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=['quick','paper'],default='paper');args=p.parse_args()
    data=load_studies(ROOT,args.profile);base=data['base'];out=base/'figures';out.mkdir(exist_ok=True)
    sources={name:sha(ROOT/name) for name in SOURCES}
    inputs={str((base/c/'manifest.json').relative_to(ROOT)):sha(base/c/'manifest.json') for c in ['geometry','beamforming','contact','scaling']}
    b,c,g,s=(data[x] for x in ['beamforming','contact','geometry','scaling'])
    figures=[('beamforming_01_setup',lambda:beam_setup(beam_story(b,16))),
       ('beamforming_02_posterior',lambda:complex_posterior_section(beam_story(b,16))),
       ('beamforming_03_directional',lambda:beam_directional(b)),
       ('beamforming_04_benchmarks',lambda:beam_benchmarks(b)),
       ('beamforming_05_responses',lambda:beam_responses(b)),
       ('beamforming_06_scaling',lambda:scaling_figure(s)),
       ('contact_01_mechanics',lambda:contact_mechanics(contact_dynamic_story(c))),
       ('contact_02_posterior',lambda:contact_posterior(c)),
       ('contact_03_directional',lambda:paper_geometry(g)),
       ('contact_04_benchmarks',lambda:contact_benchmarks(c)),
       ('contact_05_tracking',lambda:contact_tracking(c)),
       ('contact_06_higher_dimension',lambda:higher_contact(c,s))]
    for name,function in figures:
        fig=function();from lib.plotting import save
        save(fig,out/name);plt.close(fig);print('Rendered',name,flush=True)
    if sources!={name:sha(ROOT/name) for name in SOURCES} or inputs!={name:sha(ROOT/name) for name in inputs}:
        raise RuntimeError('A figure producer or scientific input changed during rendering')
    files={q.name:sha(q) for q in sorted(out.glob('*')) if q.suffix in ['.png','.pdf']}
    manifest=dict(profile=args.profile,sources=sources,input_manifests=inputs,files=files,
        groups={name:[key for key,_ in figures if key.startswith(name+'_')] for name in ['beamforming','contact']},
        text_usetex=True,figure_dpi=200,savefig_dpi=200,captions=CAPTIONS,
        selections='Predeclared trajectory zero and fixed reporting times; all comparator outcomes retained')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Verified figure manifest:',len(files),'PDF/PNG exports',flush=True)

if __name__=='__main__':main()
