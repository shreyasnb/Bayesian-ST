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


ROOT = Path(__file__).resolve().parents[1]
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
        'computation_time': 'Shows the computational cost at explicitly declared action and learning cadences.',
    },
    'contact': {
        'setup': 'Defines the physical contact geometry, applied input and measured velocity.',
        'task_performance': 'Shows paired task comparisons against same-data modal and isotropic controls.',
        'posterior_anisotropy': 'Shows the actual saved normal posterior and its matched-isotropic comparison.',
        'information': 'Connects the conditional observation law and directional excitation to Fisher information.',
    },
}

MAIN_CAPTIONS = {
    ('beamforming', 'setup'): r'Array geometry, available interference-only proper-complex snapshots and a saved distortionless beampattern. The desired steering vector is known to the task rule. The displayed episode is fixed in advance; evaluation truth is not additional training data.',
    ('beamforming', 'task_performance'): r'Held-out disturbance power relative to the known-covariance oracle at $n=32,64,128$ and $L/n=1$, together with paired moving-task episode differences. Snapshot losses are averaged within each independent episode. The mean curve has pointwise 95\% confidence bands; paired boxes describe episode differences, not confidence intervals. Complete training-count, control and motion-condition grids remain supplementary.',
    ('beamforming', 'noise_adaptation'): r'Causal variance learning under the predeclared noise-change intervention. Fits use completed, nonoverlapping past blocks and affect future variances only; every observation is assimilated once. Variance bands describe interquartile repeated-episode spread, not a posterior over the variance. Future-task and excluded-observation score intervals are pointwise 95\% intervals across independent episodes. Empirical-Bayes plug-in scales omit scale uncertainty and are outside the fixed-variance theorem. Motion and signal-power interventions remain supplementary.',
    ('beamforming', 'computation_time'): r'Warmed single-thread costs at the stated action, uncertainty-reporting and variance-learning cadences. Full-cycle timings, nonoverlapping phase costs, and memory accounting have separate scopes. Action cadence differs from the fixed block of 32 observations per scale fit. Repeated clocks quantify host timing variation; they are not additional statistical episodes or worst-case deadlines.',
    ('contact', 'setup'): r'Physical contact geometry, maintained bilateral constraint, applied input and directly measured noisy velocity. Known resistance is part of the conditional observation model. Positions and true normals are used to generate and score the experiment; estimated responses and soft commands are evaluated on the same preassigned stream rather than regenerating closed-loop trajectories.',
    ('contact', 'task_performance'): r'Physical contact task losses on independent episodes and common query streams. Paired same-data comparisons against modal, matched-isotropic and regularized spectral controls isolate different uses of the posterior. Within-episode snapshots and queries are averaged before forming pointwise 95\% confidence intervals. GREAT has different input information; complete reference and condition grids remain supplementary. Expected soft-command penalties do not imply hard constraint feasibility.',
    ('contact', 'posterior_anisotropy'): r'The actual data-generated normal-line density $q_{-A_t^+}([n])$ at the predeclared concentrated-input checkpoint $t=16$, and a fitted isotropic Bingham law with the same mode and mean squared chordal distance. Both unit-sphere views share the camera and density normalization; antipodal normals describe one contact plane. For unit queries $v(\theta)=\cos\theta e_2+\sin\theta e_3$, the risk curves are $v^T(M-M^2)v$ under each belief and its own mean. They are not two estimators evaluated against a common truth. Repeated-episode task comparisons provide the separate evidence for practical benefit. The constructed equal-distance-law example is supplementary.',
    ('contact', 'information'): r'Gaussian Fisher-information checks in aligned tangent coordinates and the information eigenvalues produced by balanced and directionally concentrated contact inputs. Analytical information is compared with independently estimated score covariance. Expected Fisher information is distinct from finite-data posterior curvature. Monte Carlo data sets are distinct from observations per data set.',
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


def readme(section, files):
    description = ('Eight single-page candidates for the main draft: four per application. '
        'They define the applications, compare task decisions, and connect uncertainty '
        'to information or noise learning and computational cost.' if section=='main' else
        'Every remaining page of the sixteen individual source studies per application. '
        'This includes calibration, negative results, model mismatch, numerical accuracy, '
        'other conditions, constructed examples and optional overview compositions.')
    lines = ['# '+('Main draft figures' if section=='main' else 'Supplementary figures'), '',
        description, '',
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
    if actual!=set(expected):
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
        dpi=EXPECTED_DPI, paper_width_in=WIDTH_IN)


def export(root=ROOT):
    root = Path(root)
    meta = catalogues(root)
    selected = plan(meta)
    existing = {path.relative_to(root).as_posix() for section in SECTIONS
                for path in (root/'results'/section).rglob('*.pdf')}
    if existing-set(selected):
        raise RuntimeError('Unrecognized existing export PDFs; preserve them before changing the selection')
    sources = {f'results/{example}/figure_manifest.json':sha(root/'results'/example/'figure_manifest.json')
               for example in EXAMPLES}
    files = {output:copy_pdf(root, output, record) for output, record in selected.items()}
    documentation = {}
    for section in SECTIONS:
        path = root/'results'/section/'README.md'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(readme(section, files))
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
