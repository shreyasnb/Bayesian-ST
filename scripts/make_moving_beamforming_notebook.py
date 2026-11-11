"""Build an independent, editable notebook from verified moving-array data.

No stream generation, inference, fitting, parameter selection or existing
notebook modification occurs here. --execute runs only the new notebook.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from textwrap import dedent

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = Path('notebooks/beamforming_moving_interference.ipynb')
sys.path.insert(0, str(ROOT))
CONFIG = dict(n=32, T=256, beta=.97, sigma2=.2, tau2=1.,
              conditions=['smooth', 'abrupt'],
              methods=['mean', 'mode', 'iso', 'great_window'])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def markdown(value, cell_id):
    return nbformat.v4.new_markdown_cell(dedent(value).strip().replace(chr(92)*2,chr(92)), id=cell_id)


def code(value, cell_id, **metadata):
    return nbformat.v4.new_code_cell(dedent(value).strip().replace(chr(92)*2,chr(92)), id=cell_id,
                                   metadata=metadata)


STYLE = r'''
STYLE = {
    "mean": (r"Posterior mean $\overline P_t^+$", "#0072B2", "o", "-"),
    "mode": (r"Mode $P_{\widehat{\mathcal X}_t}$", "#D55E00", "s", "-"),
    "iso": (r"Isotropic mean $\overline P_t^{\mathrm{iso}}$", "#CC79A7", "^", "-"),
    "great_window": ("GREAT (sliding window)", "#009E73", "D", "-"),
}
'''


def plot_cell(condition):
    """Emit the full Matplotlib source, leaving every visual choice visible."""
    source = f'''# Edit the controls and plotting statements for this figure directly.
import matplotlib.pyplot as plt

condition = {condition!r}
plot_data = data["conditions"][condition]
DPI = 300
FIGSIZE = (15, 5)
LABEL_FONT_SIZE = 25.
TICK_FONT_SIZE = 23.
LEGEND_FONT_SIZE = 22.
SAVE_PDF = True
SAVE_PNG = True
COPY_MAIN_CANDIDATE = True
PDF_PATH = EDITABLE_OUTPUT / 'moving_interference_{condition}_output_power.pdf'
PNG_PATH = PDF_PATH.with_suffix('.png')
DRAFT_SECTION = 'main'
DRAFT_PATH = ROOT / 'results' / DRAFT_SECTION / 'beamforming' / PDF_PATH.name
DRAFT_PNG_PATH = DRAFT_PATH.with_suffix('.png')

# These are confidence bands for Monte Carlo means, not posterior intervals.
SHOW_CI = {{"mean": True, "mode": False, "iso": False, "great_window": False}}
CI_ALPHA = 0.10
plt.rcParams.update({{
    "text.usetex": True,
    "text.latex.preamble": r"\\usepackage{{amsmath}}\\usepackage{{amssymb}}",
    "font.family": "serif", "font.size": LABEL_FONT_SIZE,
    "axes.labelsize": LABEL_FONT_SIZE,
    "xtick.labelsize": TICK_FONT_SIZE, "ytick.labelsize": TICK_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE, "axes.labelpad": 8.,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": None,
    "lines.linewidth": 1.6, "lines.markersize": 4.,
}})
'''
    source += dedent(STYLE)
    source += r'''
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout="constrained")
for curve in plot_data["curves"]:
    label, color_, marker_, linestyle_ = STYLE[curve["method"]]
    points = curve["points"]  # columns: t, mean, lower95, upper95
    ax.plot(points[:, 0], points[:, 1], color=color_, marker=marker_,
            linestyle=linestyle_, label=label)
    if SHOW_CI[curve["method"]]:
        ax.fill_between(points[:, 0], points[:, 2], points[:, 3],
                        color=color_, alpha=CI_ALPHA)
'''
    if condition == 'abrupt':
        source += r'''
ax.axvline(129, color="#7B3294", linewidth=1.3, linestyle="--", zorder=20,
           label="Direction change")
'''
    source += r'''
ax.set(xlabel=r"$t$", ylabel=r"$\ell_t$ (dB)", xlim=(1, 256))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#BFC4C8", alpha=.30, linewidth=.45)
fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center",
           ncol=3, frameon=False, fontsize=LEGEND_FONT_SIZE)
layout = fig.get_layout_engine()
if layout is not None and hasattr(layout, "set"):
    layout.set(h_pad=.18, w_pad=.16)

# Candidate exports have their own names and never replace another figure.
if SAVE_PDF:
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_PATH, dpi=DPI, bbox_inches=None, metadata={"Title": ""})
    if SAVE_PNG:
        fig.savefig(PNG_PATH, dpi=DPI, bbox_inches=None)
    if COPY_MAIN_CANDIDATE:
        DRAFT_PATH.parent.mkdir(parents=True, exist_ok=True)
        DRAFT_PATH.write_bytes(PDF_PATH.read_bytes())
        if SAVE_PNG:
            DRAFT_PNG_PATH.write_bytes(PNG_PATH.read_bytes())
plt.show()
'''
    return code(source, condition+'-output-power', figure_id='moving_interference_'+condition,
                source_page=1, standalone_figure=True)


def build_notebook(root, evidence):
    cells = [
        markdown(r'''
        # Moving interference

        This independent notebook compares beamformers when the interference
        direction changes smoothly or abruptly. It reuses the existing 256
        Monte Carlo simulations and their settings selected using 64 separate
        validation simulations. It does not generate data or change a tracker.

        The two exports are optional figures for the main paper. Their names
        are separate from the existing beamforming figures; the earlier abrupt
        export remains available as `abrupt_previous_main.pdf`.
        ''', 'moving-introduction'),
        code(r'''
        from pathlib import Path
        import sys
        import numpy as np
        import pandas as pd
        from IPython.display import display

        candidates = [Path.cwd(), *Path.cwd().parents]
        ROOT = next(path for path in candidates if (path / "lib" / "beamforming_moving_evidence.py").is_file())
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from lib.beamforming_moving_evidence import load_moving_evidence
        EDITABLE_OUTPUT = ROOT / "results" / "beamforming_moving" / "figures"
        ''', 'moving-imports'),
        markdown(r'''
        ## Measurements and estimates

        For a half-wavelength uniform linear array, let
        $b(\varphi)=n^{-1/2}[1,e^{\mathrm i\pi\sin\varphi},\ldots,
        e^{\mathrm i\pi(n-1)\sin\varphi}]^\top$ and $a=b(0)$. The measurements are
        $u_t=h_tc_t+e_t$, with $h_t=b(\varphi_t)$,
        $c_t\sim\mathcal{CN}(0,\tau^2)$ and
        $e_t\sim\mathcal{CN}(0,\sigma^2 I_n)$.
        The initial angle $\varphi_0$ satisfies $|a^{\mathrm H}b(\varphi_0)|^2=1/2$.

        The tracker starts from $A_0=0$ and uses $G_t=I_n$ and
        $A_t^+=\beta A_{t-1}^++\tau^2u_tu_t^{\mathrm H}/[\sigma^2(\sigma^2+\tau^2)]$
        up to an irrelevant identity term. The mean, mode and isotropic mean
        use this same corrected parameter and $\beta=0.97$.
        The isotropic mean preserves the mode and expected squared chordal
        distance: with $\zeta_t=1-\operatorname{tr}(\overline P_t^+P_{\widehat{\mathcal X}_t})$,
        $\overline P_t^{\mathrm{iso}}=(1-\zeta_t)P_{\widehat{\mathcal X}_t}
        +\zeta_t(I_n-P_{\widehat{\mathcal X}_t})/(n-1)$.
        GREAT uses the same snapshots with its independently selected sliding
        window and finite number of Grassmann updates.
        ''', 'moving-model'),
        code('''
        # Numerical configuration of the verified saved experiment.
        # Plotting controls are independent and appear in each figure cell.
        CONFIG = dict(n=32, T=256, beta=0.97, sigma2=0.2, tau2=1.0,
                      conditions=["smooth", "abrupt"],
                      methods=["mean", "mode", "iso", "great_window"])
        G_t = np.eye(CONFIG["n"])
        A_0 = np.zeros((CONFIG["n"], CONFIG["n"]), dtype=complex)
        display(pd.DataFrame([{
            "n": CONFIG["n"], "T": CONFIG["T"], "beta": CONFIG["beta"],
            "sigma2": CONFIG["sigma2"], "tau2": CONFIG["tau2"],
            "G_t": "I_n", "A_0": "0",
        }]))
        ''', 'moving-configuration'),
        markdown(r'''
        ## Output power

        For each matrix $D_t$, the beamformer is
        $w_t=(\sigma^2 I_n+\tau^2D_t)^{-1}a/
        [a^{\mathrm H}(\sigma^2 I_n+\tau^2D_t)^{-1}a]$.
        We evaluate
        $\ell_t=10\log_{10}[w_t^{\mathrm H}R_tw_t/p_t^\star]$,
        where $R_t=\sigma^2 I_n+\tau^2h_th_t^{\mathrm H}$ and
        $p_t^\star=(a^{\mathrm H}R_t^{-1}a)^{-1}$.
        The generating covariance is used for evaluation only. A smaller value
        means lower output power under the common unit-gain constraint.

        The loader verifies the saved source hashes, settings and simulation
        identifiers. It computes means and pointwise Student 95% confidence
        intervals over the 256 simulations at each recorded measurement index
        $t=8,16,\ldots,256$. These bands describe Monte Carlo uncertainty in the
        average; they are not posterior credible intervals.
        ''', 'moving-power-definition'),
        code(r'''
        data = load_moving_evidence(ROOT, CONFIG)
        assert list(data["conditions"]) == CONFIG["conditions"]
        assert all(len(curve["points"]) == 32
                   for condition in CONFIG["conditions"]
                   for curve in data["conditions"][condition]["curves"])
        print("Verified saved experiments:", ", ".join(CONFIG["conditions"]))
        print("Reporting indices:", data["conditions"]["smooth"]["time"].tolist())
        print("Initial angle (degrees):", np.rad2deg(data["config"]["phi0_rad"]))
        ''', 'moving-load'),
        markdown(r'''
        ## Numerical comparison

        Define $\overline\ell=\frac1{32}\sum_{j=1}^{32}\ell_{8j}$ within each
        simulation, then average these 256 values. This is an arithmetic
        average of the exact per-measurement dB values, including the initial
        measurements. It differs from the older moving-study summary that
        takes the logarithm after averaging power ratios and omits an initial
        interval.

        The paired table uses the same simulation identifiers for every method.
        Its difference is $\overline\ell_{\mathrm{reference}}-
        \overline\ell_{\mathrm{posterior\ mean}}$; positive values favor the
        posterior mean and negative values favor the reference.
        ''', 'moving-summary-definition'),
        code(r'''
        METHOD_LABELS = {
            "mean": "Posterior mean", "mode": "Mode",
            "iso": "Isotropic mean", "great_window": "GREAT (sliding window)",
        }
        summary = data["time_average_summary"].copy()
        summary["method"] = summary["method"].map(METHOD_LABELS)
        display(summary.rename(columns={"condition": "Condition", "method": "Method",
            "simulations": "Simulations", "mean": "Mean (dB)",
            "lower": "Lower 95%", "upper": "Upper 95%"}).round(4))

        paired = data["paired_time_average_summary"].copy()
        paired["reference"] = paired["reference"].map(METHOD_LABELS)
        display(paired.rename(columns={"condition": "Condition", "reference": "Reference",
            "simulations": "Pairs", "mean": "Difference (dB)",
            "lower": "Lower 95%", "upper": "Upper 95%"}).round(4))

        # Optional checkpoint detail; the complete arrays remain local in data.
        SHOW_CHECKPOINT_TABLE = False
        CHECKPOINT_T = 256
        if SHOW_CHECKPOINT_TABLE:
            display(data["checkpoint_summary"].query("t == @CHECKPOINT_T").round(4))
            display(data["paired_checkpoint_summary"].query("t == @CHECKPOINT_T").round(4))
        ''', 'moving-summaries'),
        markdown(r'''
        ## Smooth motion

        Here $\varphi_t=\varphi_0+
        2\varphi_0(t-1)/(T-1)$. The direction moves continuously from
        $\varphi_0$ to $3\varphi_0$ over all 256 measurements.
        This comparison shows the interaction between accumulated information
        and tracking a changing interference direction.
        ''', 'moving-smooth-explanation'),
        plot_cell('smooth'),
        markdown(r'''
        ## Abrupt motion

        Here $\varphi_t=\varphi_0$ through $t=128$ and
        $\varphi_t=3\varphi_0$ from $t=129$. The dashed line marks that change.
        The transient increase in output power and subsequent decrease show
        the response to a newly changed interference direction.
        ''', 'moving-abrupt-explanation'),
        plot_cell('abrupt'),
        markdown(r'''
        ## Interpretation

        Comparing the posterior mean with the mode tests the effect of
        averaging subspace uncertainty. Comparing it with the isotropic mean
        tests the effect of retaining uncertainty differences between
        directions. The GREAT comparison uses its selected window and update
        settings; the tables retain both favorable and unfavorable differences.
        Sliding-window GREAT has lower time-averaged output power than the
        posterior mean during smooth motion. The ordering reverses in the
        abrupt experiment. The mode has lower time-averaged power than the
        posterior mean in the abrupt case; the smooth comparison with the mode
        has a paired interval containing zero. The isotropic comparisons show
        a modest benefit from retaining directional uncertainty. These are
        comparisons for the declared motion and signal-to-noise settings.

        $G_t=I_n$ specifies the coordinate transport. It does not predict the
        physical direction changes: forgetting handles them here. These plots
        evaluate output power and do not establish calibrated posterior
        coverage during unmodeled motion. The optional main-paper copies use
        new filenames and do not replace an existing figure.
        ''', 'moving-interpretation'),
    ]
    loader_sources = {
        name: sha(root/name) for name in
        ['lib/beamforming_moving_evidence.py', 'scripts/make_moving_beamforming_notebook.py']
    }
    numerical_ids = {'moving-configuration', 'moving-load', 'moving-summaries'}
    numerical_hashes = {cell.id:hashlib.sha256(cell.source.encode()).hexdigest()
                        for cell in cells if cell.id in numerical_ids}
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'name':'python3', 'display_name':'Python 3', 'language':'python'},
        'language_info': {'name':'python', 'version':sys.version.split()[0]},
        'editable_plotting': {
            'mode':'inline_matplotlib',
            'output_directory':'results/beamforming_moving/figures',
            'figure_count':2, 'rendered_figure_count':0,
            'loader_sources':loader_sources,
            'input_hashes':dict(evidence['input_hashes']),
            'numerical_cell_hashes':numerical_hashes,
        },
        'moving_beamforming': {
            'config':CONFIG,
            'source_hashes':{record['path']:record['sha256'] for record in
                evidence['provenance']['verified_source_files'].values()},
            'reuse_scope':'Existing selected settings and all 256 saved simulations; no new inference',
            'candidate_exports':True,
        },
    })
    nbformat.validate(notebook)
    return notebook



def write_figure_manifest(root,notebook,evidence):
    """Record only the two new candidate figures and their verified inputs."""
    folder=root/'results/beamforming_moving'
    data_manifest=folder/'data/manifest.json'
    files={};figures={}
    caption_definition=(r'$\ell_t=10\log_{10}[(w_t^{\mathrm H}R_tw_t)/p_t^\star]$, '
        r'$R_t=\sigma^2I_n+\tau^2h_th_t^{\mathrm H}$, '
        r'$p_t^\star=(a^{\mathrm H}R_t^{-1}a)^{-1}$. '
        'Curves average 256 simulations at each stored t=8,16,...,256; '
        'shading is a pointwise Student 95% interval for the posterior-mean average.')
    for condition in CONFIG['conditions']:
        basename='moving_interference_'+condition+'_output_power'
        source=folder/'figures'/(basename+'.pdf')
        candidate=root/'results/main/beamforming'/(basename+'.pdf')
        outputs={}
        for path in [source,source.with_suffix('.png'),candidate,candidate.with_suffix('.png')]:
            if not path.is_file():raise RuntimeError('Missing new moving figure: '+str(path))
            outputs[str(path.relative_to(root))]=sha(path)
        if source.read_bytes()!=candidate.read_bytes() or source.with_suffix('.png').read_bytes()!=candidate.with_suffix('.png').read_bytes():
            raise RuntimeError('Candidate export differs from its editable source figure')
        motion=(r'$\varphi_t=\varphi_0+2\varphi_0(t-1)/(T-1)$' if condition=='smooth'
            else r'$\varphi_t=\varphi_0$ for $t\leq128$, $3\varphi_0$ for $t\geq129$')
        cell=next(cell for cell in notebook.cells if cell.id==condition+'-output-power')
        figures[basename+'.pdf']=dict(status='completed',condition=condition,
            figure_size_in=[15,5],dpi=300,text_usetex=True,optional_main_candidate=True,
            formula_or_claim=caption_definition+' '+motion+'.',caption=caption_definition+' '+motion+'.',
            source_cell_id=cell.id,source_cell_sha256=hashlib.sha256(cell.source.encode()).hexdigest(),
            event_t=129 if condition=='abrupt' else None,
            uncertainty='Pointwise Student 95% confidence interval across 256 simulations',
            output_hashes=outputs)
        files.update(outputs)
    retained={str(path.relative_to(root)):sha(path) for path in
        [folder/'figures/abrupt_previous_main.pdf',folder/'figures/abrupt_previous_main.png'] if path.exists()}
    result=dict(schema=1,status='completed',example='beamforming_moving',
        created_utc=datetime.now(timezone.utc).isoformat(),optional_main_candidate=True,
        configuration=evidence['config'],config=evidence['config'],
        notebook=str(NOTEBOOK),notebook_sha256=sha(root/NOTEBOOK),
        producer='scripts/make_moving_beamforming_notebook.py',
        source_hashes=dict(notebook.metadata['editable_plotting']['loader_sources']),
        data_manifest=str(data_manifest.relative_to(root)),data_manifest_sha256=sha(data_manifest),
        input_hashes=dict(evidence['input_hashes']),
        original_source_hashes=dict(notebook.metadata['moving_beamforming']['source_hashes']),
        figures=figures,files=files,retained_previous_main=retained,
        figure_count=2,dpi=300,figure_size_in=[15,5],text_usetex=True,
        plotting=dict(labels=25,ticks=23,legend=22,layout='constrained',
            pdf='Native vector text and curves',png_dpi=300),
        numerical_cell_hashes=dict(notebook.metadata['editable_plotting']['numerical_cell_hashes']),
        scope='Existing saved moving data and selected settings; no new simulation, inference, fitting or tuning. Optional candidates do not replace existing main figures.')
    (folder/'figure_manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Execute only the new moving notebook')
    args=parser.parse_args()
    from lib.beamforming_moving_evidence import load_moving_evidence
    evidence=load_moving_evidence(ROOT,CONFIG)
    notebook=build_notebook(ROOT,evidence)
    target=ROOT/NOTEBOOK
    target.parent.mkdir(parents=True,exist_ok=True)
    if args.execute:
        for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
            os.environ[key]='1'
        for key,value in {
            'MPLCONFIGDIR':'/tmp/bayesian-st-matplotlib',
            'JUPYTER_CONFIG_DIR':'/tmp/bayesian-st-jupyter-config',
            'JUPYTER_DATA_DIR':'/tmp/bayesian-st-jupyter-data',
            'JUPYTER_RUNTIME_DIR':'/tmp/bayesian-st-jupyter',
            'IPYTHONDIR':'/tmp/bayesian-st-ipython',
        }.items():os.environ.setdefault(key,value)
        from nbclient import NotebookClient
        try:
            NotebookClient(notebook,timeout=600,kernel_name='python3',
                resources={'metadata':{'path':str(ROOT)}}).execute()
        finally:
            nbformat.write(notebook,target)
        notebook.metadata['moving_beamforming']['executed_utc']=datetime.now(timezone.utc).isoformat()
        notebook.metadata['editable_plotting']['rendered_figure_count']=2
    nbformat.write(notebook,target)
    if args.execute:write_figure_manifest(ROOT,notebook,evidence)
    print('Wrote',NOTEBOOK,flush=True)


if __name__=='__main__':
    main()
