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
    "mean": (r"Posterior mean $\overline P_t^+$", "#0072B2", "-"),
    "mode": (r"Mode $P_{\widehat{\mathcal X}_t}$", "#D55E00", "-"),
    "iso": (r"Isotropic mean $\overline P_t^{\mathrm{iso}}$", "#CC79A7", "-"),
    "great_window": ("GREAT (sliding window)", "#009E73", "-"),
}
'''


def plot_cell(condition=None):
    """Emit ordinary plotting statements for one figure or the combined pair."""
    combined = condition is None
    filename = ('moving_interference_comparison.pdf' if combined
                else 'moving_interference_'+condition+'_output_power.pdf')
    source = f'''# Edit these plotting controls independently for this figure.
import matplotlib.pyplot as plt

DPI = 300
FIGSIZE = {(7, 2.8) if combined else (15, 5)!r}
LABEL_FONT_SIZE = {10.5 if combined else 25.}
TICK_FONT_SIZE = {10. if combined else 23.}
LEGEND_FONT_SIZE = {10.5 if combined else 22.}
PRINT_WIDTH_IN = 7.
PRINT_SCALE = FIGSIZE[0] / PRINT_WIDTH_IN
LINE_WIDTH = 1.0 * PRINT_SCALE
SPINE_WIDTH = 0.95
IQR_ALPHA = 0.22
SHOW_IQR = {{"mean": True, "mode": True, "iso": True, "great_window": True}}
SAVE_PDF = True
SAVE_PNG = True
COPY_MAIN_CANDIDATE = True
PDF_PATH = EDITABLE_OUTPUT / {filename!r}
PNG_PATH = PDF_PATH.with_suffix('.png')
DRAFT_SECTION = 'main'
DRAFT_PATH = ROOT / 'results' / DRAFT_SECTION / 'beamforming' / PDF_PATH.name
DRAFT_PNG_PATH = DRAFT_PATH.with_suffix('.png')

plt.rcParams.update({{
    "text.usetex": True,
    "text.latex.preamble": r"\\usepackage{{amsmath}}\\usepackage{{amssymb}}",
    "font.family": "serif", "font.size": LABEL_FONT_SIZE,
    "axes.labelsize": LABEL_FONT_SIZE,
    "xtick.labelsize": TICK_FONT_SIZE, "ytick.labelsize": TICK_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE, "axes.labelpad": {4. if combined else 8.},
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": None,
    "lines.linewidth": LINE_WIDTH,
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.spines.bottom": True, "axes.spines.left": True,
}})
'''
    source += dedent(STYLE)
    if combined:
        source += '''
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI, layout="constrained")
panels = list(zip(axes, ("smooth", "abrupt")))
'''
    else:
        source += f'''
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout="constrained")
panels = [(ax, {condition!r})]
'''
    source += r'''
for ax, condition in panels:
    plot_data = data["conditions"][condition]
    samples = data["checkpoint_losses"]
    samples = samples[samples["condition"].eq(condition)]
    # Empirical quartiles across the actual 256 simulations, not CI endpoints.
    quantiles = samples.groupby(["method", "t"])["true_ratio_db"].quantile([0.25, 0.75]).unstack()

    # All empirical bands are behind every mean curve.
    for curve in plot_data["curves"]:
        method = curve["method"]
        label, color_, linestyle_ = STYLE[method]
        points = curve["points"]  # columns t, mean, lower95, upper95; mean is unchanged
        quartiles = quantiles.loc[method].reindex(points[:, 0])
        if SHOW_IQR[method]:
            ax.fill_between(points[:, 0], quartiles[0.25].to_numpy(), quartiles[0.75].to_numpy(),
                            color=color_, alpha=IQR_ALPHA, linewidth=0., zorder=1)
    for curve in plot_data["curves"]:
        method = curve["method"]
        label, color_, linestyle_ = STYLE[method]
        points = curve["points"]
        ax.plot(points[:, 0], points[:, 1], color=color_, linestyle=linestyle_,
                label=label, linewidth=LINE_WIDTH, zorder=10)
    if condition == "abrupt":
        ax.axvline(129, color="#7B3294", linewidth=1.3, linestyle="--", zorder=20,
                   label="Direction change")
    ax.set(xlabel=r"$t$", ylabel=r"$\ell_t$ (dB)", xlim=(1, 256))
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True,
                   bottom=True, left=True, labelbottom=True, labelleft=True,
                   labeltop=False, labelright=False, width=SPINE_WIDTH)
    ax.grid(color="#BFC4C8", alpha=.30, linewidth=.45)

handles, labels = [], []
for ax, condition in panels:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        if label not in labels:
            handles.append(handle)
            labels.append(label)
fig.legend(handles, labels, loc="outside lower center", ncol=3,
           frameon=False, fontsize=LEGEND_FONT_SIZE, columnspacing=1.,
           handlelength=1.8, handletextpad=.5, labelspacing=.3)
layout = fig.get_layout_engine()
if layout is not None and hasattr(layout, "set"):
    layout.set(h_pad=.04, w_pad=.04)

if SAVE_PDF:
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_PATH, dpi=DPI, bbox_inches="tight", pad_inches=.04, metadata={"Title": ""})
    if SAVE_PNG:
        fig.savefig(PNG_PATH, dpi=DPI, bbox_inches="tight", pad_inches=.04)
    if COPY_MAIN_CANDIDATE:
        DRAFT_PATH.parent.mkdir(parents=True, exist_ok=True)
        DRAFT_PATH.write_bytes(PDF_PATH.read_bytes())
        if SAVE_PNG:
            DRAFT_PNG_PATH.write_bytes(PNG_PATH.read_bytes())
plt.show()
'''
    return code(source, 'combined-output-power' if combined else condition+'-output-power',
                figure_id='moving_interference_comparison' if combined else 'moving_interference_'+condition,
                source_page=1, standalone_figure=True)


def write_plotting_statistics(root,evidence):
    """Save only the new empirical display quartiles; old summaries stay intact."""
    folder=root/'results/beamforming_moving/plotting'
    folder.mkdir(parents=True,exist_ok=True)
    rows=evidence['checkpoint_losses']
    grouped=rows.groupby(['condition','method','t'])['true_ratio_db']
    stats=grouped.agg(simulations='size',mean='mean',q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75)).reset_index()
    if not stats.simulations.eq(256).all():raise RuntimeError('IQR bands require all 256 simulations')
    reference=evidence['checkpoint_summary'][['condition','method','t','mean']]
    checked=stats.merge(reference,on=['condition','method','t'],suffixes=('_display','_original'),validate='one_to_one')
    import numpy as np
    np.testing.assert_allclose(checked.mean_display,checked.mean_original,rtol=1e-13,atol=1e-13)
    target=folder/'pointwise_iqr.csv';stats.to_csv(target,index=False)
    result=dict(schema=1,status='completed',producer='scripts/make_moving_beamforming_notebook.py',
        producer_sha256=sha(root/'scripts/make_moving_beamforming_notebook.py'),
        inputs=evidence['input_hashes'],artifacts={target.relative_to(root).as_posix():sha(target)},
        display='Pointwise empirical 25th–75th percentiles of true_ratio_db across all 256 simulations.',
        mean_definition='The existing arithmetic Monte Carlo mean at each saved measurement index.',
        points=len(stats),simulations_per_point=256,means_unchanged=True,
        no_smoothing_jitter_or_added_points=True,old_confidence_intervals_unchanged=True)
    (folder/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def build_notebook(root, evidence):
    cells = [
        markdown(r'''
        # Moving interference

        This independent notebook compares beamformers when the interference
        direction changes smoothly or abruptly. It reuses the existing 256
        Monte Carlo simulations and their settings selected using 64 separate
        validation simulations. It does not generate data or change a tracker.

        The individual and combined exports are optional figures for the main paper. Their names
        are separate from the existing beamforming figures; the earlier abrupt
        export remains available as `abrupt_previous_main.pdf`.
        ''', 'moving-introduction'),
        code(r'''
        from pathlib import Path
        import sys
        import os

        # Use the session's existing TeX tools only when they are available.
        LOCAL_TEX = Path('/workspace/.local/tex')
        if (LOCAL_TEX / 'usr/bin/dvipng').exists():
            os.environ['PATH'] = str(LOCAL_TEX / 'usr/bin') + os.pathsep + os.environ['PATH']
            os.environ.setdefault('TEXMFHOME', str(LOCAL_TEX / 'usr/share/texmf'))
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
        identifiers. Curves show means across 256 Monte Carlo simulations.
        Shaded regions show the pointwise 25th–75th percentiles across simulations.
        These are empirical spreads of the simulated output powers, not posterior
        credible regions. The tables retain their original Student 95% confidence
        intervals for the means at the recorded indices $t=8,16,\ldots,256$.

        Blue, orange, pink and green solid curves denote the posterior mean
        $\overline P_t^+$, mode $P_{\widehat{\mathcal X}_t}$, matched isotropic
        mean $\overline P_t^{\mathrm{iso}}$ and GREAT (sliding window), respectively.
        In the abrupt comparison, the purple dashed line at $t=129$ marks the
        interference-direction change.
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
        markdown('## Combined comparison\n\nCurves show means across 256 Monte Carlo simulations. Shaded regions show the pointwise 25th–75th percentiles across simulations.', 'moving-combined-explanation'),
        plot_cell(),
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
            'figure_count':3, 'rendered_figure_count':0,
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
    """Bind the two editable individual plots and the native-width comparison."""
    folder=root/'results/beamforming_moving'
    data_manifest=folder/'data/manifest.json'
    plotting_manifest=folder/'plotting/manifest.json'
    files={};figures={}
    sentence='Curves show means across 256 Monte Carlo simulations. Shaded regions show the pointwise 25th–75th percentiles across simulations.'
    definition=(r'Normalized output power $\ell_t=10\log_{10}[(w_t^{\mathrm H}R_tw_t)/p_t^\star]$, '
        r'$R_t=\sigma^2I_n+\tau^2h_th_t^{\mathrm H}$, '
        r'$p_t^\star=(a^{\mathrm H}R_t^{-1}a)^{-1}$, with $n=32$, $\beta=0.97$. '
        r'Blue, orange, pink and green solid curves denote the posterior mean $\overline P_t^+$, '
        r'mode $P_{\widehat{\mathcal X}_t}$, matched isotropic mean $\overline P_t^{\mathrm{iso}}$ '
        'and GREAT (sliding window), respectively. '+sentence+
        r' Means are arithmetic, using the 32 saved indices $t=8,16,\ldots,256$. '
        'These empirical simulation spreads are not posterior credible regions; table confidence intervals are unchanged.')
    specs=[('moving_interference_smooth_output_power','smooth','smooth-output-power',[15,5]),
           ('moving_interference_abrupt_output_power','abrupt','abrupt-output-power',[15,5]),
           ('moving_interference_comparison','comparison','combined-output-power',[7,2.8])]
    for basename,condition,cell_id,size in specs:
        source=folder/'figures'/(basename+'.pdf')
        candidate=root/'results/main/beamforming'/(basename+'.pdf')
        outputs={}
        for path in [source,source.with_suffix('.png'),candidate,candidate.with_suffix('.png')]:
            if not path.is_file():raise RuntimeError('Missing moving figure: '+str(path))
            outputs[path.relative_to(root).as_posix()]=sha(path)
        if source.read_bytes()!=candidate.read_bytes() or source.with_suffix('.png').read_bytes()!=candidate.with_suffix('.png').read_bytes():
            raise RuntimeError('Manuscript copy differs from the editable figure')
        motion=(r'$\varphi_t=\varphi_0+2\varphi_0(t-1)/(T-1)$' if condition=='smooth' else
                r'$\varphi_t=\varphi_0$ through $t=128$, $3\varphi_0$ from $t=129$. The purple dashed line at $t=129$ marks the interference-direction change' if condition=='abrupt' else
                r'Left: smooth motion. Right: abrupt motion. The purple dashed line at $t=129$ marks the interference-direction change')
        cell=next(cell for cell in notebook.cells if cell.id==cell_id)
        figures[basename+'.pdf']=dict(status='completed',condition=condition,
            figure_size_in=size,dpi=300,text_usetex=True,optional_main_candidate=True,
            formula_or_claim=definition+' '+motion+'.',caption=definition+' '+motion+'.',
            source_cell_id=cell.id,source_cell_sha256=hashlib.sha256(cell.source.encode()).hexdigest(),
            event_t=None if condition=='smooth' else 129,
            uncertainty='Pointwise empirical 25th–75th percentiles across 256 simulations',
            mean_curves_unchanged=True,iqr_alpha=.22,
            plotting=dict(labels=10.5 if condition=='comparison' else 25,
                ticks=10 if condition=='comparison' else 23,
                legend=10.5 if condition=='comparison' else 22,
                line_width=1.0 if condition=='comparison' else 15/7,
                trajectory_style='Solid lines without markers; original colors retained.'),
            output_hashes=outputs)
        files.update(outputs)
    retained={path.relative_to(root).as_posix():sha(path) for path in
        [folder/'figures/abrupt_previous_main.pdf',folder/'figures/abrupt_previous_main.png'] if path.exists()}
    result=dict(schema=1,status='completed',example='beamforming_moving',
        created_utc=datetime.now(timezone.utc).isoformat(),optional_main_candidate=True,
        configuration=evidence['config'],config=evidence['config'],
        notebook=NOTEBOOK.as_posix(),notebook_sha256=sha(root/NOTEBOOK),
        producer='scripts/make_moving_beamforming_notebook.py',
        source_hashes=dict(notebook.metadata['editable_plotting']['loader_sources']),
        data_manifest=data_manifest.relative_to(root).as_posix(),data_manifest_sha256=sha(data_manifest),
        plotting_manifest=plotting_manifest.relative_to(root).as_posix(),plotting_manifest_sha256=sha(plotting_manifest),
        input_hashes=dict(evidence['input_hashes']),
        original_source_hashes=dict(notebook.metadata['moving_beamforming']['source_hashes']),
        figures=figures,files=files,retained_previous_main=retained,
        figure_count=3,dpi=300,figure_size_in=[15,5],text_usetex=True,
        figure_sizes={name:record['figure_size_in'] for name,record in figures.items()},
        plotting=dict(labels=25,ticks=23,legend=22,layout='constrained',
            pdf='Native vector text and curves',png_dpi=300,display_bands='empirical pointwise IQR',
            iqr_alpha=.22,combined_native_size_in=[7,3.5],four_sided_inward_ticks=True,
            trajectory_style='Solid lines without markers; original colors retained.'),
        numerical_cell_hashes=dict(notebook.metadata['editable_plotting']['numerical_cell_hashes']),
        scope='Presentation-only empirical IQR and print-width combined plot. Saved data, selected settings, mean curves, confidence intervals and paired tables remain unchanged.')
    (folder/'figure_manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def _protected_files(root):
    paths=[path for folder in ['results/beamforming_moving/data','results/beamforming_moving/tables']
           for path in (root/folder).rglob('*') if path.is_file()]
    paths += [root/name for name in [
        'results/beamforming/data/paper/performance/manifest.json',
        'results/beamforming/data/paper/performance/selected.json',
        'results/beamforming/data/paper/performance/moving_checkpoints.csv.gz']]
    return {path.relative_to(root).as_posix():sha(path) for path in paths}


def _generate(args):
    from lib.beamforming_moving_evidence import load_moving_evidence
    protected=_protected_files(ROOT)
    evidence=load_moving_evidence(ROOT,CONFIG)
    write_plotting_statistics(ROOT,evidence)
    notebook=build_notebook(ROOT,evidence)
    target=ROOT/NOTEBOOK;target.parent.mkdir(parents=True,exist_ok=True)
    if args.execute:
        for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
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
        notebook.metadata['editable_plotting']['rendered_figure_count']=3
    nbformat.write(notebook,target)
    after=_protected_files(ROOT)
    if protected!=after:raise RuntimeError('A protected numerical/settings file changed during the render window')
    folder=ROOT/'results/beamforming_moving/plotting'
    guard=folder/'run_guard.json'
    guard.write_text(json.dumps(dict(status='passed',lock='/tmp/bayesian-st-notebook-render.lock',
        protected_files=protected,after_hashes=after,all_protected_files_unchanged=True,
        protected_count=len(protected),scope='Original numerical/settings files were unchanged during this exclusive render window; separately authorized notebook/figure updates are outside this data-only guard.'),indent=2)+'\n')
    stats_path=folder/'manifest.json';stats=json.loads(stats_path.read_text())
    stats['artifacts'][guard.relative_to(ROOT).as_posix()]=sha(guard)
    stats['render_guard']='Passed under the shared exclusive notebook-render lock.'
    stats_path.write_text(json.dumps(stats,indent=2,allow_nan=False)+'\n')
    if args.execute:write_figure_manifest(ROOT,notebook,evidence)
    print('Wrote',NOTEBOOK,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Execute only this moving notebook')
    args=parser.parse_args()
    import fcntl
    with Path('/tmp/bayesian-st-notebook-render.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        _generate(args)


if __name__=='__main__':main()
