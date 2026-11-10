"""Partition verified paper PDF pages into main-draft and supplementary files.

This is an editorial export, not a numerical producer or a renderer. Whole
documents are copied byte for byte. When a source contains both main and
supplementary pages, PyMuPDF copies the original PDF page objects without
rasterizing them. The complete original catalogue remains in each application
directory. ``verify_export`` uses only the standard library.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.notebook_pdf_exports import (
    declared_notebook_pdf_routes, verify_notebook_pdf_exports)
EXAMPLES = ('beamforming', 'contact')
EXPECTED_DPI = 300
WIDTH_IN = 7.
MANIFEST = Path('results/draft_figure_manifest.json')
SECTIONS = ('main', 'supplementary')
INDIVIDUAL = (
    'setup', 'task_performance', 'posterior_anisotropy', 'noise_adaptation',
    'information', 'posterior_curvature', 'tracking_bound', 'block_updates',
    'additional_comparisons', 'variance_estimation', 'predictive_calibration',
    'predictive_scores', 'model_mismatch', 'computation_time',
    'numerical_accuracy', 'paper_summary')

MAIN = {
    'beamforming': {
        'setup': 'Defines the array geometry and the measurements available to the task.',
        'task_performance': 'Shows moderate-dimensional task losses and paired same-episode controls.',
        'noise_adaptation': 'Shows the declared noise-change intervention and causal variance learning.',
        'computation_time': 'Shows time to process 64 observations, with task-decision intervals and variance-fitting operations stated explicitly.',
    },
    'contact': {
        'setup': 'Defines the physical contact geometry, applied input and measured velocity.',
        'task_performance': 'Shows response MSE and confidence intervals for the posterior mean and same-data modal, isotropic and regularized-projector controls.',
        'posterior_anisotropy': 'Shows the actual saved normal posterior and its matched-isotropic comparison.',
        'information': 'Shows which physical normal rotations the saved applied inputs distinguish, using the exact Gaussian likelihood divergence.',
    },
}

MAIN_CAPTIONS = {
    ('beamforming', 'setup'): r'Array geometry, available interference-only proper-complex snapshots and a saved distortionless beampattern. The desired steering vector is known to the task rule. The displayed episode is fixed in advance; evaluation truth is not additional training data.',
    ('beamforming', 'task_performance'): r'Held-out disturbance power relative to the known-covariance oracle at $n=32,64,128$ and $L/n=1$, together with paired moving-task episode differences. Snapshot losses are averaged within each independent episode. The mean curve has pointwise 95\% confidence bands; paired boxes describe episode differences, not confidence intervals. Complete training-count, control and motion-condition grids remain supplementary.',
    ('beamforming', 'noise_adaptation'): r'Causal variance learning under the predeclared noise-change intervention. Fits use completed, nonoverlapping past blocks and affect future variances only; every observation is assimilated once. Variance bands describe interquartile repeated-episode spread, not a posterior over the variance. Future-task and excluded-observation score intervals are pointwise 95\% intervals across independent episodes. Empirical-Bayes plug-in scales omit scale uncertainty and are outside the fixed-variance theorem. Motion and signal-power interventions remain supplementary.',
    ('beamforming', 'computation_time'): r'Warmed single-thread time to process exactly 64 observations. Every observation updates the belief; task decisions are computed after every observation or every 16 observations as indicated. The lower panel separates measurement updates, task decisions, uncertainty reports, variance fitting and other execution time for the stated settings. Optional uncertainty reports and variance fits occur every 32 observations, separately from the task-decision interval. EM uses the stated two or four iterations. Sums of per-operation medians need not equal the median total elapsed time. Repeated clocks measure execution variability on one data set, not additional statistical episodes or worst-case deadlines. Memory accounting is retained on the supplementary second page.',
    ('contact', 'setup'): r'Physical contact geometry, maintained bilateral constraint, applied input and directly measured noisy velocity. Known resistance is part of the conditional observation model. Positions and true normals are used to generate and score the experiment; estimated responses and soft commands are evaluated on the same preassigned stream rather than regenerating closed-loop trajectories.',
    ('contact', 'task_performance'): r'Physical contact response MSE on independent episodes and common query streams. Panel (a) uses the stationary concentrated-input condition; its posterior-mean curve carries a pointwise 95\% confidence band. Panel (b) shows absolute episode-average MSE for the moving sphere condition, with pointwise 95\% intervals for every displayed method, including $\overline P_t^+$, $P_{\widehat{\mathcal X}_t}$, the matched isotropic mean $\overline P_t^{\mathrm{iso}}$, and the regularized projector. The isotropic control preserves mode and mean overlap; the regularized projector is $\rho P_{\widehat{\mathcal X}_t}+(1-\rho)(2/3)I_3$ with independently validated $\rho$. Query and time losses are averaged within each episode before intervals are formed. Paired MSE differences are retained on the supplementary second page; differences and absolute MSE both have squared velocity units, whereas relative MSE would be a ratio. GREAT has different input information, stated outside its figure label. Full reference and condition grids remain supplementary.',
    ('contact', 'posterior_anisotropy'): r'The actual normal-line density $q_{-A_t^+}([n])$ at the fixed concentrated-input checkpoint $t=16$, and its fitted isotropic Bingham law with equal mode and mean squared chordal distance. The mean matrices are denoted by $\overline P_t^+$ and $\overline P_t^{\mathrm{iso}}$, following the supplement. Both unit-sphere views retain the saved coordinates, camera and shared logarithmic density normalization; antipodal normals represent the same plane. The blue--purple--pink--yellow gradient changes only the color mapping. For $v(\theta)=\cos\theta e_2+\sin\theta e_3$ and $\theta$ in radians, the lower panel evaluates $v(\theta)^\top[\overline P-(\overline P)^2]v(\theta)$ with each indicated belief and its own mean $\overline P$. These are posterior expected clean-response errors under different beliefs, rather than two estimators evaluated against one common truth. Practical comparisons use the separate repeated-episode task study. The constructed equal-distance-law example is supplementary.',
    ('contact', 'information'): r'Which normal rotations the original saved concentrated-input experiment distinguishes. At the generating plane, $n_0=e_3$, $X_0=(e_1,e_2)$ and $z=\operatorname{vec}(Z)$. The graph chart has normal $n(Z)=(n_0-X_0z)/\sqrt{1+\|z\|^2}$. With the same 16 prescribed inputs and true measurement variance, $p_Z$ is the joint Gaussian velocity law at this plane. Panel (a) shows equal values of its exact divergence $D_{\mathrm{KL}}(p_0\|p_Z)=z^\top\mathcal F_{\mathrm{cond}}z/[2(1+\|z\|^2)]$. Panel (b) uses the physical normal angle $\delta$ along the two native information directions: rotations of $5^\circ$ and $30^\circ$ yield nearly equal divergences, approximately $0.24$. This connects directional excitation to distinguishability of physical rotations. Near zero the formula gives the manuscript Fisher expansion. The generating plane is used for this theoretical analysis, not supplied to the estimator. The contours are likelihood divergences, not posterior credible regions; no new observations, fitting or independent experimental repetitions are introduced.',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_mapping(root, mapping):
    for name, expected in mapping.items():
        digest = expected['sha256'] if isinstance(expected, dict) else expected
        if sha(Path(root)/name) != digest:
            raise RuntimeError('Changed or missing source evidence: ' + name)


def catalogues(root):
    """Verify the complete canonical figure evidence before selecting pages."""
    root = Path(root)
    records = {}
    verified = set()
    for example in EXAMPLES:
        folder = root/'results'/example
        path = folder/'figure_manifest.json'
        meta = json.loads(path.read_text())
        if not (meta['status']=='completed' and meta['stage']=='paper'
                and meta['example']==example and not meta['unresolved']):
            raise RuntimeError('Incomplete canonical figure evidence: ' + example)
        if not (meta['preview_dpi']==meta['rasterized_layer_dpi']==EXPECTED_DPI
                and meta['paper_width_in']==WIDTH_IN and meta['text_usetex'] is True
                and meta['titles'] is False):
            raise RuntimeError('Expected completed 300-dpi, seven-inch LaTeX exports: ' + example)
        verify_mapping(root, {meta['validation_path']:meta['validation_sha256']})
        for mapping in (meta['sources'], meta['input_manifests']):
            for name, expected in mapping.items():
                key = (name, expected if isinstance(expected, str) else expected['sha256'])
                if key not in verified:
                    verify_mapping(root, {name: expected})
                    verified.add(key)
        verify_mapping(folder, meta['files'])
        if len(meta['figures']) != 19:
            raise RuntimeError('The canonical nineteen-PDF catalogue is incomplete')
        for stem in INDIVIDUAL:
            entry = meta['figures'][stem+'.pdf']
            if not entry['pages'] or entry['status']!='completed':
                raise RuntimeError('Unresolved figure pages: ' + stem)
            if any(abs(page['figure_size_in'][0]-WIDTH_IN)>1e-10
                   or page['titles'] is not False for page in entry['pages']):
                raise RuntimeError('Unexpected source page layout: ' + stem)
            # The source manifest records every scientific array/table used.
            for name, expected in entry.get('data_hashes', {}).items():
                key = (name, expected if isinstance(expected, str) else expected['sha256'])
                if key not in verified:
                    verify_mapping(root, {name: expected})
                    verified.add(key)
        records[example] = meta
    return records


def selected_main_pages(example, stem, record):
    if stem not in MAIN[example]:
        return []
    if example=='beamforming' and stem=='noise_adaptation':
        pages = [page['page'] for page in record['pages']
                 if page.get('condition')=='noise_change']
    elif example=='contact' and stem=='posterior_anisotropy':
        pages = [page['page'] for page in record['pages']
                 if page.get('selection', {}).get('episode')==0
                 and page.get('selection', {}).get('t')==16
                 and page.get('selection', {}).get('query_family')==1]
    elif stem in ('computation_time','task_performance'):
        pages = [1]
    else:
        pages = [page['page'] for page in record['pages']]
    if len(pages)!=1:
        raise RuntimeError('Expected exactly one selected main page: ' + example+'/'+stem)
    return pages


def plan(meta):
    files = {}
    for example in EXAMPLES:
        for stem in INDIVIDUAL:
            source = Path('results')/example/(stem+'.pdf')
            record = meta[example]['figures'][stem+'.pdf']
            main_pages = selected_main_pages(example, stem, record)
            for section in SECTIONS:
                pages = [page for page in record['pages']
                         if (page['page'] in main_pages)==(section=='main')]
                if not pages:
                    continue
                output = Path('results')/section/example/(stem+'.pdf')
                rationale = (MAIN[example][stem] if section=='main' else
                    'Retains every remaining source page, including other conditions, '
                    'constructed examples, calibration, limitations and complete diagnostics. '
                    'The overview PDF is an optional composition of evidence already presented elsewhere.')
                files[output.as_posix()] = dict(application=example, section=section,
                    source_pdf=source.as_posix(),
                    source_pdf_sha256=meta[example]['files'][stem+'.pdf'],
                    source_pages=[page['page'] for page in pages], pages=pages,
                    selection_rationale=rationale,
                    proposed_latex_caption=(MAIN_CAPTIONS[example, stem] if section=='main'
                        else record['proposed_latex_caption']),
                    full_source_caption=record['proposed_latex_caption'])
    return files


def _pdf_module():
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError as error:
            raise SystemExit('PDF page extraction requires the optional PyMuPDF package.') from error
    return fitz


def copy_pdf(root, output, record):
    fitz = _pdf_module()
    source = Path(root)/record['source_pdf']
    destination = Path(root)/output
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp.pdf')
    with fitz.open(source) as original:
        selected = record['source_pages']
        whole = selected==list(range(1, len(original)+1))
        if whole:
            shutil.copyfile(source, temporary)
        else:
            with fitz.open() as subset:
                for page in selected:
                    subset.insert_pdf(original, from_page=page-1, to_page=page-1)
                subset.set_metadata(dict(title='', author='', subject='', keywords=''))
                subset.save(temporary, garbage=4, deflate=True)
        with fitz.open(temporary) as exported:
            if len(exported)!=len(selected) or (exported.metadata.get('title') or '').strip():
                raise RuntimeError('Unexpected exported page count or PDF title: '+output)
            for index, source_page in enumerate(selected):
                page = exported[index]
                expected = original[source_page-1]
                if tuple(page.mediabox)!=tuple(expected.mediabox) or tuple(page.cropbox)!=tuple(expected.cropbox):
                    raise RuntimeError('PDF page geometry changed during export: '+output)
                if abs(page.mediabox.width-WIDTH_IN*72)>.01:
                    raise RuntimeError('Exported page is not seven inches wide: '+output)
                if not whole:
                    # This is a visual identity check, not a raster export:
                    # the actual saved PDF still contains the original vector
                    # and image objects. No pixel image is written to disk.
                    if page.get_pixmap(dpi=72, alpha=False).samples!=expected.get_pixmap(dpi=72, alpha=False).samples:
                        raise RuntimeError('Extracted PDF page appearance changed: '+output)
    temporary.replace(destination)
    return dict(sha256=sha(destination), bytes=destination.stat().st_size,
                export_method='byte-exact PDF copy' if whole else 'lossless PDF page-object extraction',
                saved_pdf_rasterized=False, rendered_page_identity_checked=True,
                metadata_title_empty=True, **record)


def readme(section, files, notebook_routes=None):
    description = ('Eight single-page candidates for the main draft: four per application. '
        'They define the applications, compare task decisions, and connect uncertainty '
        'to information or noise learning and computational cost.' if section=='main' else
        'Every remaining page of the sixteen individual source studies per application. '
        'This includes calibration, negative results, model mismatch, numerical accuracy, '
        'other conditions, constructed examples and optional overview compositions.')
    lines = ['# '+('Main draft figures' if section=='main' else 'Supplementary figures'), '',
        '## Current notebook PDFs', '',
        'Running a beamforming or contact figure cell with `SAVE_PDF=True` updates '
        'its local PDF and the copy listed here. The filename matches the visible '
        '`PDF_PATH`; `DRAFT_SECTION` and `DRAFT_PATH` in that figure block control '
        'its manuscript destination. These copies retain the notebook figure size, '
        'labels and styling. `SAVE_PDF=False` leaves both destinations unchanged.', '']
    for output, route in sorted((notebook_routes or {}).items()):
        if route['section'] == section:
            relative = Path(output).relative_to(Path('results')/section)
            lines.append('- ['+route['example']+'/'+Path(output).name+']('+relative.as_posix()+').')
    lines += ['', '## Canonical catalogue exports', '', description, '',
        'These are PDF copies or lossless PDF page extractions from the verified '
        'canonical catalogues. Pages retain seven-inch width, LaTeX/vector text '
        'and lines, and 300-dpi raster layers. Increasing dpi affects raster '
        'layers and previews; PDF vector text has no pixel resolution.', '',
        'No simulation, inference, fitting, benchmark tuning or numerical data '
        'changes occur during this export. Main selections are editorial choices made '
        'while preparing the manuscript; evaluated episodes and methods are unchanged. Every individual '
        'source page appears in exactly one of these two folders.', '',
        'The three original supplementary bundles are not copied because they '
        'repeat individual pages, including main candidates. Complete original '
        'catalogues remain in `results/beamforming/` and `results/contact/` for '
        'the notebooks. Exact selectors, source hashes, captions and output '
        'hashes are in [the export manifest](../draft_figure_manifest.json).', '']
    for example in EXAMPLES:
        lines += ['## '+example.capitalize(), '']
        for name, entry in files.items():
            if entry['section']==section and entry['application']==example:
                relative = Path(name).relative_to(Path('results')/section)
                pages = ', '.join(map(str, entry['source_pages']))
                lines.append('- ['+Path(name).stem.replace('_',' ')+']('+relative.as_posix()+
                    '): source page'+('s ' if len(entry['source_pages'])!=1 else ' ')+pages+'.')
        lines.append('')
    return '\n'.join(lines).rstrip()+'\n'


def verify_export(root=ROOT):
    """Read-only, standard-library validation for the final handoff checker."""
    root = Path(root)
    path = root/MANIFEST
    exported = json.loads(path.read_text())
    if exported['status']!='completed' or exported['producer_sha256']!=sha(root/'scripts/export_draft_figures.py'):
        raise RuntimeError('Stale or incomplete draft figure export')
    meta = catalogues(root)
    expected = plan(meta)
    if set(expected)!=set(exported['files']):
        raise RuntimeError('Draft PDF selection differs from the declared partition')
    if exported['dpi']!=EXPECTED_DPI or exported['paper_width_in']!=WIDTH_IN:
        raise RuntimeError('Draft export has the wrong publication style')
    verify_mapping(root, exported['source_figure_manifests'])
    gates = {record['validation_path']:record['validation_sha256'] for record in meta.values()}
    if exported['source_validation_gates']!=gates:
        raise RuntimeError('Draft export uses a different scientific validation gate')
    verify_mapping(root, exported['source_validation_gates'])
    verify_mapping(root, exported['documentation'])
    actual = {path.relative_to(root).as_posix() for section in SECTIONS
              for path in (root/'results'/section).rglob('*.pdf')}
    notebook_copies = verify_notebook_pdf_exports(root)
    if actual!=set(expected)|set(notebook_copies['files']):
        raise RuntimeError('Unexpected or missing categorized PDF files')
    coverage = Counter()
    for output, planned in expected.items():
        entry = exported['files'][output]
        for field, value in planned.items():
            if entry[field]!=value:
                raise RuntimeError('Changed PDF selector/caption: '+output)
        if sha(root/output)!=entry['sha256'] or (root/output).stat().st_size!=entry['bytes']:
            raise RuntimeError('Changed categorized PDF: '+output)
        if entry['saved_pdf_rasterized'] is not False or entry['metadata_title_empty'] is not True:
            raise RuntimeError('Unverified PDF preservation: '+output)
        for page in entry['source_pages']:
            coverage[entry['source_pdf'], page] += 1
    all_pages = {(f'results/{example}/{stem}.pdf', page['page'])
                 for example in EXAMPLES for stem in INDIVIDUAL
                 for page in meta[example]['figures'][stem+'.pdf']['pages']}
    if set(coverage)!=all_pages or any(count!=1 for count in coverage.values()):
        raise RuntimeError('Main/supplementary source page partition is incomplete or duplicated')
    return dict(status='verified', pdf_files=len(expected), source_pages=len(all_pages),
        main_pdf_files=sum(entry['section']=='main' for entry in expected.values()),
        supplementary_pdf_files=sum(entry['section']=='supplementary' for entry in expected.values()),
        dpi=EXPECTED_DPI, paper_width_in=WIDTH_IN,
        notebook_pdf_copies=len(notebook_copies['files']))


def export(root=ROOT):
    root = Path(root)
    meta = catalogues(root)
    selected = plan(meta)
    existing = {path.relative_to(root).as_posix() for section in SECTIONS
                for path in (root/'results'/section).rglob('*.pdf')}
    notebook_copies = verify_notebook_pdf_exports(root)
    if existing-set(selected)-set(notebook_copies['files']):
        raise RuntimeError('Unrecognized existing export PDFs; preserve them before changing the selection')
    sources = {f'results/{example}/figure_manifest.json':sha(root/'results'/example/'figure_manifest.json')
               for example in EXAMPLES}
    # Keep byte-identical catalogue copies when only notebook destinations or
    # editorial documentation changed. Every retained file is checked again.
    previous = (json.loads((root/MANIFEST).read_text()).get('files', {})
                if (root/MANIFEST).is_file() else {})
    files = {}
    for output, record in selected.items():
        old = previous.get(output)
        destination = root/output
        if (old and destination.is_file()
                and all(old.get(key)==value for key,value in record.items())
                and sha(destination)==old.get('sha256')
                and destination.stat().st_size==old.get('bytes')):
            files[output] = old
        else:
            files[output] = copy_pdf(root, output, record)
    documentation = {}
    for section in SECTIONS:
        path = root/'results'/section/'README.md'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(readme(section, files, declared_notebook_pdf_routes(root)))
        documentation[path.relative_to(root).as_posix()] = sha(path)
    manifest = dict(schema=1, status='completed', producer='scripts/export_draft_figures.py',
        producer_sha256=sha(root/'scripts/export_draft_figures.py'),
        source_figure_manifests=sources, dpi=EXPECTED_DPI, paper_width_in=WIDTH_IN,
        source_validation_gates={record['validation_path']:record['validation_sha256']
                                 for record in meta.values()},
        section_order=list(SECTIONS), applications=list(EXAMPLES),
        page_partition_scope='Every page of the sixteen individual canonical PDFs per application appears exactly once; the three duplicate bundles remain only in canonical directories.',
        selection_scope='Explicit editorial selection by scientific role; all source pages retained, no benchmark or episode selection.',
        main_policy=MAIN, documentation=documentation, files=files)
    (root/MANIFEST).write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    verify_mapping(root, sources)
    return verify_export(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify_export() if args.check_only else export(), indent=2))


if __name__=='__main__':
    main()
