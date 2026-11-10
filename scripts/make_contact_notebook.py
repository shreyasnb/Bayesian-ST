"""Create the new contact example only after full general/contact validation."""
from pathlib import Path
import sys
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lib.contact_validation import require_contact_gate


def main():
    require_contact_gate(ROOT)
    cells=[]
    def md(value):cells.append(nbf.v4.new_markdown_cell(value))
    def code(value):cells.append(nbf.v4.new_code_cell(value))
    md(r'''# Predicting motion during maintained contact

Can a distribution over a contact tangent plane improve response predictions or provide useful uncertainty beyond a tuned point estimator? This pilot generates force and velocity from a constrained mechanical model, then compares the existing Bayesian tracker with force-aware point and linear-map controls.

This is a separate candidate replacement for the field example. Existing experiments and manuscript files are preserved. A positive result is not required. The model is an idealized **bilateral guide with known viscous resistance**, rather than general unilateral or frictional contact.''')
    md(r'''## From force balance to the observation model

For a surface with unit normal $n(p)$, negligible inertia and isotropic resistance $dI$, the plant solves

$$\begin{bmatrix}dI_3&-n(p)\\n(p)^\top&0\end{bmatrix}
\begin{bmatrix}\dot p\\\lambda\end{bmatrix}
=\begin{bmatrix}F\\0\end{bmatrix}.$$

Thus $\dot p=P F/d$, where $P=I-nn^\top$ projects onto the two-dimensional tangent plane. The filter receives $v=F/d$ and $u=\dot p+e$, with $e\sim N(0,sI_3)$. This is the draft's **conditional known-input model** on $\mathrm{Gr}(2,3)$.

The plant uses geometric normals and an independent force-balance solve. Sphere motion is integrated with surface-drift control. Observations are instantaneous solver velocities, not finite differences of positions. The simulated positions shown below are **plant/scoring records only**; estimators receive the same force and noisy velocity streams. Consequently this pilot makes no claim of practical superiority over methods using position encoders.''')
    code('''from pathlib import Path
import sys,json,hashlib,subprocess
ROOT=Path.cwd() if (Path.cwd()/'lib').exists() else Path.cwd().parent
sys.path.insert(0,str(ROOT))
from lib.contact_validation import require_contact_gate
from lib.contact_reporting import load_contact
from lib.contact_visuals import (physical_setup,paired_losses,coverage_width,
    response_intervals,risk_calibration,benchmark_losses,runtime,mechanical_diagnostics,
    validation_curves)
from lib.plotting import plt, DPI
from IPython.display import display,Markdown
import pandas as pd
OUT=ROOT/'results/contact_pilot'
gate=require_contact_gate(ROOT)
assert plt.rcParams['text.usetex'] is True
assert plt.rcParams['figure.dpi']==DPI and plt.rcParams['savefig.dpi']==DPI
subprocess.run([sys.executable,str(ROOT/'scripts/run_contact_pilot.py'),'--workers','2'],cwd=ROOT,check=True)
frames=load_contact(ROOT)
print('Validation:',gate['status'],'checks:',gate['tests'])
display(pd.Series(frames['config'])[['T','dt','d','v0','sigma','validation_trials','test_trials','primary_condition']])
def show(fig):
    display(fig)
    plt.close(fig)
''')
    md(r'''## Physical trajectories and tangent planes

The plane has a fixed non-axis-aligned normal. Spheres have radii 0.20 m and 0.05 m, with independently randomized initial orientations. The curved guides use a bounded world-frame force program. The weak-input condition reduces two force components by five; its uncertainty is measured, not assumed. The applied force does not depend on a hidden subspace rotation.

Geometry and true normals enter only the plant and scoring. Arrows compare the free velocity $F/d$ with the constrained velocity. A maintained guide permits signed reaction forces: separation, impacts, friction cones and changing contact modes are outside this experiment.''')
    code("show(physical_setup(frames,path=OUT/'figures/physical_setup'))")
    md(r'''## Numerical and model checks

The tangent-plane posterior has the normal-line complement
$$\bar P(A)=I_3-E_{q_{-A},k=1}[nn^\top].$$
The existing spectral backend is used where its truncation budget resolves. The static plane reaches concentration spans of order $8\times10^4$, beyond that backend's default finite-order budget. A separate concentration-adapted sphere integration handles this regime without changing the existing numerical core. Continuous radial CDFs determine response quantiles; weighted node quantiles are avoided.

Moments, state thresholds and scalar response endpoints each receive resolution checks. Independent pole/ridge integrals test concentrations through $10^6$. Resolution agreement and quadrature error estimates are numerical evidence, not rigorous roundoff enclosures. Unresolved flags remain visible.

The mode is also compared with an independently calculated weighted projector least-squares solution. Their equality checks the implementation; it does not establish an accuracy gain.''')
    code('''numeric=json.loads((OUT/'numerics/checks.json').read_text())
for name,digest in numeric['sources'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
display(pd.Series(numeric)[['status','physical_targets','maximum_span',
    'maximum_mean_resolution_error','maximum_quantile_relative_error',
    'maximum_region_radius_scaled_error','maximum_region_hpd_scaled_error']])
display(pd.Series(frames['metadata']['checks']))
mechanics=json.loads((OUT/'numerics/mechanics_checks.json').read_text())
display(pd.Series(mechanics)[['status','maximum_position_difference',
    'maximum_velocity_difference','maximum_velocity_difference_over_sigma']])
''')
    md(r'''## Response predictions against tuned controls

Nine fixed queries have norm $v_0$: coordinate directions and normalized pairwise sums/differences. They represent hypothetical forces applied at the **current** contact position; they do not advance the tool or predict a future trajectory. The primary endpoint averages the squared three-component response error over these queries within each trajectory, at observation 600.

Each method's memory is selected using independent validation task loss and frozen across the curved test conditions. The stationary plane is a separate $\beta=1$ control. Comparisons include the same-posterior mode, independently tuned mode, scalar shrinkage, ridge force-to-velocity regression, and a cheap velocity-only PCA control. For each memory, scalar shrinkage is optimized continuously on pooled validation loss; a coarse endpoint grid cannot create an artificial posterior advantage.

The primary condition is the smaller-radius balanced guide, with two preselected references: shrinkage and ridge. Other pointwise intervals are exploratory. The 64 independent trajectories are the statistical replicates; query coordinates and report times do not increase that count. A gain over velocity-only PCA primarily demonstrates the value of knowing applied force. A gain over ridge can reflect the projector constraint; the mode/shrinkage comparisons isolate what the full distribution adds.''')
    code("display(pd.DataFrame(frames['metadata']['selected']))\nshow(paired_losses(frames,path=OUT/'figures/paired_losses'))\ndisplay(frames['paired'][frames['paired'].primary.astype(bool)])")
    md(r'''## Coverage and interval width

State balls, state HPD regions, and componentwise clean-response intervals answer different questions. Their sizes have different units. State coverage uses Wilson intervals across trajectories; average component coverage first averages the 27 query/component indicators within a trajectory. These are pointwise summaries, not simultaneous coverage of all responses or a motion path.

The balanced stationary control matches the observation and state model. At complete input cycles, its isotropic input Gram and rotation symmetry make state-region coverage a clean check. Fixed-world functional central intervals need not inherit that state-coverage property, particularly near extremal responses.

On curved guides, identity prediction and discounting are approximations. Nominal posterior mass does not guarantee physical-state coverage. Widths normalized by $v_0$ reveal whether an apparently reliable interval is nearly uninformative. No noisy-observation intervals are computed in this pilot; every response band below concerns the clean mechanical response.''')
    code("show(coverage_width(frames,path=OUT/'figures/coverage_width'))\ndisplay(frames['coverage'][frames['coverage'].t==frames['config']['T']])")
    md(r'''## What this pilot supports

The following assessment links the observed outcomes to the scope decision. Numerical correctness, meaningful decision gains and calibrated physical uncertainty are evaluated separately. A stationary success does not establish robustness under curved motion or a wrong resistance law.''')
    code("display(Markdown((OUT/'findings.md').read_text()))")
    md(r'''## Supplementary: directional intervals and posterior risk

These intervals show one preassigned weak-input trajectory at the three reporting times. The coordinates are components of physical velocity. Misses remain visible.

For any query $w$, the maintained posterior predicts response risk $w^\top(\bar P-\bar P^2)w$. Compare this full-vector risk with the corresponding full-vector empirical squared errors, averaged over queries within trajectories. Equality of posterior risk and repeated-data error is not automatic under fixed states, discounting or model mismatch.''')
    code("show(response_intervals(frames,path=OUT/'figures/response_intervals'))\nshow(risk_calibration(frames,path=OUT/'figures/risk_calibration'))")
    md(r'''## Supplementary: geometry, excitation and the theorem

Radius alone does not measure tracking difficulty. The tables retain surface travel, normal rotation and retained force excitation. A small input-Gram eigenvalue does not by itself imply little information about the tangent plane. In normal tangent coordinates, the instantaneous information eigenvalues are
$$\lambda_{\min}=\frac{(n^\top v)^2}{s},\qquad
\lambda_{\max}=\frac{\|v\|^2}{s}.$$
Nearly normal forcing can be informative even when the observed noiseless velocity is close to zero.

The draft's modal bound is evaluated only under its real conditional model, fixed true noise and deterministic identity frame. It includes actual residual drift and may fail its sufficient condition. It is separate from every credible radius plotted here. The anisotropic-resistance stress is excluded from that theorem.''')
    code("show(mechanical_diagnostics(frames,path=OUT/'figures/mechanical_diagnostics'))\ndisplay(frames['bounds'].groupby('condition')[['applicable','valid','violated']].sum())")
    md(r'''## Supplementary: resistance mismatch and cost

With true $D=d\,\mathrm{diag}(1,1.5,2)$, the KKT solve still defines a valid tangent velocity, but the response is generally not $P(F/d)$. The stress therefore scores actual KKT query velocities and retains the geometric projection target separately. Better memory does not repair this likelihood mismatch.

Corrections occur every observation; posterior integration and query decisions occur only at 60, 180 and 600. Runtime totals sum the three reports and the cumulative correction cost. A low correction time does not imply that all response intervals can be computed at the 20 Hz observation rate. These are cloud implementation measurements, not universal complexity claims.''')
    code("show(benchmark_losses(frames,path=OUT/'figures/benchmark_losses'))\nshow(runtime(frames,path=OUT/'figures/runtime'))\nshow(paired_losses(frames,path=OUT/'figures/paired_stress',include_stress=True))\nshow(coverage_width(frames,path=OUT/'figures/coverage_stress',include_stress=True))")
    md('''## Supplementary: validation choices

These are validation curves, not extra test replications. One common curved policy is frozen before all curved tests. Scalar shrinkage minimizes the pooled validation quadratic continuously for each memory; the coarse scalar grid is retained only as a diagnostic. Structural endpoints and finite-grid memory/regularization boundaries are reported in the selected table. They do not establish global optimality.''')
    code("show(validation_curves(frames,path=OUT/'figures/validation_curves'))\ndisplay(frames['tuning'][frames['tuning'].selected.astype(bool)])")
    md(r'''## Interpretation for robotics

This pilot tests a force/velocity response model for a maintained, quasi-static bilateral guide. It is not a contact detector, a frictional polishing simulation, or a validation on a real robot. The Grassmann state represents an unoriented tangent plane; it does not encode the sign of a unilateral normal or whether contact can sustain tension.

The posterior mean $E[P]$ is generally full rank and does not itself define one contact tangent plane. Its output is an expected velocity prediction, not a guarantee of a physically admissible velocity command at the actual surface. The experiment is open loop with prescribed forces; no closed-loop controller, robot-arm dynamics or safe-contact policy is evaluated.

Positions would normally be available on many robots. They are deliberately not a measurement channel here, so practical claims require a later comparison with a causal local-plane estimate from the same noisy encoder stream. Unknown or anisotropic resistance also requires a different observation model; it should not be absorbed into measurement noise or described as repaired by discounting.

No process-noise machinery or paper-scope change is introduced. The code and evidence remain separate so the replacement decision can be made after this bounded test. Robotics context: Lin, Howard and Vijayakumar, *Learning Null Space Projections*, ICRA 2015, DOI 10.1109/ICRA.2015.7139551. The mechanical realization here is the proposed pilot, not a reproduction of that paper.''')
    notebook=nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.12'}})
    nbf.write(notebook,ROOT/'notebooks/03_contact.ipynb')
    print('Created contact notebook after the general/contact gate passed.')


if __name__=='__main__':main()
