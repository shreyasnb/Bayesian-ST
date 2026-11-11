"""Retain canonical PDFs alongside the six revised main notebook figures.

This editorial export copies complete canonical documents byte for byte into
supplementary/. The main/ PDFs come from their editable notebook cells. An old
manifest-owned main catalogue copy is removed only after hash verification and
retention of its complete source document in supplementary/. The original
catalogue remains in each application directory. ``verify_export`` uses only
the standard library and does not execute notebook code.
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

# The revised numerical section includes only these six notebook PDFs.
# Canonical catalogue copies are retained in supplementary/ in their entirety.
MAIN = {
    'beamforming': {
        'task_performance_output_power': 'Stationary dimension comparison with a fixed total of T=200 measurements; the n=64 table is available separately.',
        'noise_adaptation_page1': 'Variance estimation and prediction before each measurement.',
    },
    'contact': {
        'information_rotation': 'Likelihood divergence along two physical normal rotations.',
        'posterior_anisotropy_spheres': 'Full and matched isotropic normal densities.',
        'posterior_anisotropy_risk': 'Normalized posterior response variance.',
        'task_performance_page1': 'Mean squared velocity prediction error.',
    },
}
MAIN_NOTEBOOK_PDFS = frozenset(
    (Path('results')/'main'/example/(stem+'.pdf')).as_posix()
    for example, figures in MAIN.items() for stem in figures)

MAIN_CAPTIONS = {('beamforming', 'task_performance_output_power'): 'Stationary beamforming: '
                                                   '$\\ell_t=10\\log_{10}[(w_t^{\\mathrm '
                                                   'H}R_tw_t)/p_t^\\star]$ '
                                                   'from~\\eqref{eq:beam-power-comparison}, after $T=200$ '
                                                   'measurements for each $n\\in\\{32,64,128\\}$. Blue, '
                                                   'orange, green and dark gray solid curves denote the '
                                                   'posterior mean $\\overline P_T^+$, mode '
                                                   '$P_{\\widehat{\\mathcal X}_T}$, GREAT (sliding window) '
                                                   'and MVDR with regularized sample covariance, '
                                                   'respectively. Each curve joins three arithmetic means '
                                                   'over 256 Monte Carlo simulations; blue shading gives '
                                                   'pointwise Student 95\\% confidence intervals for the '
                                                   'posterior-mean averages.',
 ('beamforming', 'noise_adaptation_page1'): 'Causal variance estimation with $n=32$ and $T=256$. Black, gray '
                                            'and blue curves use supplied true variances, fixed estimates '
                                            'and EM estimates, respectively. The purple dashed line at '
                                            '$t=129$ marks the noise-variance change from $\\sigma_t^2=0.2$ '
                                            'to $0.8$; $\\tau_t^2=1$. The top-left and top-right panels show '
                                            'the variances available before $u_t$: solid estimate curves '
                                            'join medians at the eight saved indices $t=32,64,\\ldots,256$, '
                                            'and the blue bands contain the pointwise empirical 25th--75th '
                                            'percentiles across 256 Monte Carlo simulations. Black '
                                            'references give the true schedules. The bottom-left and '
                                            'bottom-right panels show arithmetic means over these '
                                            'simulations of $\\ell_t$ in~\\eqref{eq:beam-power-comparison} '
                                            'and $\\log\\pi_t(u_t)$ in~\\eqref{eq:beam-predictive-density}, '
                                            'evaluated from $q_{A_t^-}$ before correction at those same '
                                            'indices; all three methods have pointwise Student 95\\% '
                                            'confidence bands. EM uses blocks of $L=32$ measurements and two '
                                            'iterations; a fit completed at $t$ is first used at $t+1$.',
 ('contact', 'information_rotation'): 'Anisotropy of the conditional measurement information for the '
                                      'stationary plane with normal $e_3$. Orange and blue solid curves show '
                                      'the exact $D_{\\mathrm{KL}}(p_0\\Vert p_z)$ for '
                                      '$z=(\\tan\\delta,0)^\\top$ and $z=(0,\\tan\\delta)^\\top$, '
                                      'respectively, where $n(z)=(e_3-z_1e_1-z_2e_2)/\\sqrt{1+\\|z\\|^2}$ '
                                      'and $p_z$ is the joint conditional likelihood of the 16 measurements '
                                      'from~\\eqref{eq:contact-model}. Curves evaluate 201 angles from '
                                      '$0.25^\\circ$ to $35^\\circ$; the orange and blue points mark '
                                      '$5^\\circ$ and $30^\\circ$. The thin neutral segment joins the '
                                      'highlighted values $D_{\\mathrm{KL}}=0.23957$ at $5^\\circ$ and '
                                      '$0.23077$ at $30^\\circ$: nearly equal divergences arise at very '
                                      'different rotation angles. Their actual heights are retained. These '
                                      'are deterministic likelihood comparisons, with no Monte Carlo '
                                      'averaging or confidence band.',
 ('contact', 'posterior_anisotropy_spheres'): 'Posterior normal densities for one saved stationary-plane '
                                              'simulation at corrected time $t=16$. Left: $q_{-A_t^+}([n])$; '
                                              'right: the isotropic Bingham law with the same modal line and '
                                              'expected squared chordal distance, whose plane mean '
                                              'is~\\eqref{eq:isotropic-control}. The '
                                              'blue--purple--pink--yellow colors show normalized density '
                                              'relative to invariant measure on one common logarithmic '
                                              'scale, from lower to higher density. Orange points mark the '
                                              'common modal line; gray curves are coordinate great circles. '
                                              'Antipodal normals represent the same plane. Each surface '
                                              'evaluates one saved distribution; it is not a Monte Carlo '
                                              'average or a confidence region.',
 ('contact', 'posterior_anisotropy_risk'): 'Posterior response variance $r_t(v(\\theta))/v_0^2$ '
                                           'from~\\eqref{eq:contact-response-variance}, for '
                                           '$v(\\theta)=v_0(\\cos\\theta\\,e_1+\\sin\\theta\\,e_2)$ and '
                                           '$v_0=0.02\\,\\mathrm{m/s}$ at $t=16$. Blue and pink solid curves '
                                           'use the full and matched isotropic laws, respectively, each with '
                                           'its own posterior mean. The curves evaluate the same saved '
                                           'posterior pair at 181 angles over $[0,2\\pi]$; they are '
                                           'deterministic posterior variances, with no Monte Carlo averaging '
                                           'or confidence band.',
 ('contact', 'task_performance_page1'): 'Velocity prediction error $\\mathcal E_t(D_t)$ '
                                        'in~\\eqref{eq:contact-prediction-error}, in '
                                        '$(\\mathrm{mm}/\\mathrm{s})^2$. In the stationary-plane panel '
                                        '(left), blue, orange, brown and green solid curves denote the '
                                        'posterior mean $\\overline P_t^+$, mode $P_{\\widehat{\\mathcal '
                                        'X}_t}$, regularized matrix $\\widetilde M_t$ and GREAT (sliding '
                                        'window). Each joins arithmetic means over 256 Monte Carlo '
                                        'simulations at $t=1,4,16,32$; blue shading gives pointwise Student '
                                        '95\\% confidence intervals for the posterior-mean averages. In the '
                                        'sphere panel (right), blue, orange, pink and brown points denote '
                                        'the posterior mean, mode, matched isotropic mean and regularized '
                                        'matrix $\\widetilde M_t$. Each point first averages the error over '
                                        'all 48 indices $t=4,8,\\ldots,192$ within each simulation, then '
                                        'averages the 256 resulting values; horizontal bars are Student '
                                        '95\\% confidence intervals for every displayed method. Here '
                                        '$\\widetilde M_t=\\rho P_{\\widehat{\\mathcal '
                                        'X}_t}+(1-\\rho)(2/3)I_3$.'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def optional_moving_copies(root):
    """Verify independent candidate copies without changing the main policy."""
    root = Path(root)
    if not (root/'notebooks/beamforming_moving_interference.ipynb').is_file():
        return {}
    from scripts.check_moving_beamforming import optional_moving_routes
    records = {}
    for destination, route in optional_moving_routes(root).items():
        path, local = root/destination, root/route['local_pdf']
        if not path.is_file() or not local.is_file():
            raise RuntimeError('Run the independent moving notebook to export: '+destination)
        if path.read_bytes()[:5] != b'%PDF-' or sha(path) != sha(local):
            raise RuntimeError('Moving candidate differs from its source export: '+destination)
        records[destination] = dict(route, sha256=sha(path), bytes=path.stat().st_size)
    return records


def contact_diagnostic_copies(root):
    """Read independent new exports without replacing the retained catalogue."""
    root = Path(root)
    if not (root/'notebooks/contact_prediction_diagnostics.ipynb').is_file():
        return {}
    from scripts.check_contact_prediction_diagnostics import optional_contact_diagnostic_routes
    records = {}
    for destination, route in optional_contact_diagnostic_routes(root).items():
        path, local = root/destination, root/route['local_pdf']
        if not path.is_file() or not local.is_file():
            raise RuntimeError('Run the contact diagnostics notebook to export: '+destination)
        if path.read_bytes()[:5] != b'%PDF-' or sha(path) != sha(local):
            raise RuntimeError('Contact diagnostic differs from its source export: '+destination)
        records[destination] = dict(route, sha256=sha(path), bytes=path.stat().st_size)
    return records


def verify_export_producer(root, exported):
    """Verify the exact retained export producer while keeping old outputs intact."""
    if exported['status'] != 'completed':
        raise RuntimeError('Incomplete draft figure export')
    expected = exported['producer_sha256']
    if expected == sha(root/'scripts/export_draft_figures.py'):
        return
    # Only this exact pre-diagnostics manifest may retain its historical
    # producer. Its data, captions, selectors and PDF hashes remain strict.
    archive = root/'archive/dba55ba/presentation_sources/scripts/export_draft_figures.py'
    if (sha(root/MANIFEST) == '58ffecda530f0dd3f4386b3ed015e827819d83cb5b376dc00dfa5cb54bf6c788'
            and expected == 'a9b5e2482b063d8f6f947d655e55952426fa2583c1785df0c8b775b2239dd3d7'
            and archive.is_file() and sha(archive) == expected):
        return
    raise RuntimeError('Stale or unverified draft figure export producer')


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
    """Canonical copies are supplementary; the six main figures are notebooks."""
    return []


def plan(meta):
    """Retain every canonical individual PDF, without splitting off main pages."""
    files = {}
    for example in EXAMPLES:
        for stem in INDIVIDUAL:
            source = Path('results')/example/(stem+'.pdf')
            record = meta[example]['figures'][stem+'.pdf']
            output = Path('results')/'supplementary'/example/(stem+'.pdf')
            files[output.as_posix()] = dict(application=example, section='supplementary',
                source_pdf=source.as_posix(),
                source_pdf_sha256=meta[example]['files'][stem+'.pdf'],
                source_pages=[page['page'] for page in record['pages']],
                pages=record['pages'],
                selection_rationale='Retains the complete verified canonical PDF. '
                    'The revised main section uses only its six declared notebook PDFs.',
                proposed_latex_caption=record['proposed_latex_caption'],
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
    lines = ['# '+('Main draft figures' if section=='main' else 'Supplementary figures'), '',
        ('The revised main numerical section uses exactly six PDFs: two for the '
         'array and four for constrained motion. The contact sphere pair and '
         'response-variance plot are combined in TeX.' if section=='main' else
         'These figures contain the additional notebook plots and the complete '
         'verified canonical catalogues. Tracking-bound checks, setup drawings '
         'and numerical diagnostics remain supplementary.'), '',
        '## Current notebook PDFs', '',
        'Running a figure cell with `SAVE_PDF=True` updates its local PDF and '
        'the same-named manuscript copy. The visible `DRAFT_SECTION` and '
        '`DRAFT_PATH` set the destination; `SAVE_PDF=False` updates neither copy.', '']
    for output, route in sorted((notebook_routes or {}).items()):
        if route['section'] == section:
            relative = Path(output).relative_to(Path('results')/section)
            lines.append('- ['+route['example']+'/'+Path(output).name+']('+relative.as_posix()+').')
    if section == 'main':
        lines += ['', '## Caption notes', '']
        for example, figures in MAIN.items():
            for stem in figures:
                lines += ['### '+example+'/'+stem+'.pdf', '',
                          MAIN_CAPTIONS[example, stem], '']
    else:
        lines += ['', '## Canonical catalogue exports', '',
            'Each of the sixteen individual canonical PDFs per application is '
            'copied in full. Every source page appears exactly once in this '
            'canonical-copy list; notebook figures may reproduce the same saved '
            'evidence. The three duplicate bundles remain in the original '
            'application directories.', '',
            'Canonical copies retain their declared page geometry, LaTeX/vector '
            'text and 300-dpi raster layers. Exporting these PDFs does not rerun '
            'simulation, inference, fitting, or parameter selection. Sources, '
            'captions and output hashes are in [the export manifest]'
            '(../draft_figure_manifest.json).', '']
        for example in EXAMPLES:
            lines += ['### '+example.capitalize(), '']
            for name, entry in files.items():
                if entry['application'] == example:
                    relative = Path(name).relative_to(Path('results')/section)
                    pages = ', '.join(map(str, entry['source_pages']))
                    lines.append('- ['+Path(name).stem.replace('_', ' ')+']('
                        +relative.as_posix()+'): source pages '+pages+'.')
            lines.append('')
    return '\n'.join(lines).rstrip()+'\n'


def _check_main_notebook_routes(routes):
    declared = {name for name, route in routes.items() if route['section']=='main'}
    if declared!=MAIN_NOTEBOOK_PDFS:
        raise RuntimeError('The revised main section requires exactly its six declared notebook PDFs')


def verify_export(root=ROOT):
    """Read-only, standard-library validation for the final handoff checker."""
    root = Path(root)
    path = root/MANIFEST
    exported = json.loads(path.read_text())
    verify_export_producer(root, exported)
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
    _check_main_notebook_routes(notebook_copies['routes'])
    moving_copies = optional_moving_copies(root)
    contact_copies = contact_diagnostic_copies(root)
    if actual!=set(expected)|set(notebook_copies['files'])|set(moving_copies)|set(contact_copies):
        raise RuntimeError('Unexpected or missing categorized PDF files')
    actual_main = {name for name in actual if name.startswith('results/main/')}
    contact_main = {name for name, record in contact_copies.items() if record['section']=='main'}
    if actual_main!=MAIN_NOTEBOOK_PDFS|set(moving_copies)|contact_main:
        raise RuntimeError('Unexpected main figure outside the six retained figures and optional moving candidates')
    if exported.get('optional_moving_candidates', {}) != moving_copies:
        raise RuntimeError('Stale independent moving candidate export records')
    if (exported.get('main_notebook_files')!=sorted(MAIN_NOTEBOOK_PDFS)
            or exported.get('main_policy')!=MAIN):
        raise RuntimeError('Draft export uses a different main-figure policy')
    caption_notes = {name: MAIN_CAPTIONS[Path(name).parent.name, Path(name).stem]
                     for name in sorted(MAIN_NOTEBOOK_PDFS)}
    if exported.get('main_caption_notes')!=caption_notes:
        raise RuntimeError('Draft export uses different main-figure definitions')
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
    return dict(status='verified', pdf_files=len(actual), canonical_pdf_files=len(expected),
        source_pages=len(all_pages), main_pdf_files=len(actual_main),
        supplementary_pdf_files=len(actual-actual_main),
        dpi=EXPECTED_DPI, paper_width_in=WIDTH_IN,
        notebook_pdf_copies=len(notebook_copies['files']),
        optional_moving_candidates=len(moving_copies),
        contact_diagnostic_exports=len(contact_copies),
        retained_main_figures=len(MAIN_NOTEBOOK_PDFS))


def export(root=ROOT):
    root = Path(root)
    meta = catalogues(root)
    selected = plan(meta)
    existing = {path.relative_to(root).as_posix() for section in SECTIONS
                for path in (root/'results'/section).rglob('*.pdf')}
    notebook_copies = verify_notebook_pdf_exports(root)
    _check_main_notebook_routes(notebook_copies['routes'])
    moving_copies = optional_moving_copies(root)
    contact_copies = contact_diagnostic_copies(root)
    live_copies = set(notebook_copies['files']) | set(moving_copies) | set(contact_copies)
    if not MAIN_NOTEBOOK_PDFS<=live_copies:
        raise RuntimeError('Save the six revised main notebook figures before exporting')
    if set(selected)&set(notebook_copies['routes']):
        raise RuntimeError('A notebook destination overlaps a canonical supplementary copy')
    previous = (json.loads((root/MANIFEST).read_text()).get('files', {})
                if (root/MANIFEST).is_file() else {})
    if existing-set(selected)-live_copies-set(previous):
        raise RuntimeError('Unrecognized existing export PDFs; preserve them before changing the selection')
    # Only the previous manifest establishes ownership of obsolete canonical
    # copies. Check every such file before copying or deleting anything.
    for output in sorted((existing&set(previous))-live_copies):
        old, path = previous[output], root/output
        if sha(path)!=old.get('sha256') or path.stat().st_size!=old.get('bytes'):
            raise RuntimeError('Modified previous export PDF; preserve it before migration: '+output)
    obsolete = (existing&set(previous))-set(selected)-live_copies
    for output in sorted((existing&set(selected))-set(previous)):
        if sha(root/output)!=selected[output]['source_pdf_sha256']:
            raise RuntimeError('Refusing to overwrite an unowned PDF: '+output)
    sources = {f'results/{example}/figure_manifest.json':sha(root/'results'/example/'figure_manifest.json')
               for example in EXAMPLES}
    # Keep byte-identical catalogue copies when only notebook destinations or
    # editorial documentation changed. Every retained file is checked again.
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
    # Every replacement catalogue document is retained in full before an old
    # main-page copy is removed. Never remove unknown or modified files.
    for output, record in files.items():
        if (sha(root/output)!=record['source_pdf_sha256']
                or (root/output).stat().st_size!=record['bytes']):
            raise RuntimeError('Canonical supplementary copy is not byte exact: '+output)
    for output in sorted(obsolete):
        old, path = previous[output], root/output
        if sha(path)!=old['sha256'] or path.stat().st_size!=old['bytes']:
            raise RuntimeError('Previous export changed during migration: '+output)
        path.unlink()
    documentation = {}
    for section in SECTIONS:
        path = root/'results'/section/'README.md'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(readme(section, files, declared_notebook_pdf_routes(root)))
        if section == 'main' and moving_copies:
            with path.open('a') as stream:
                stream.write('\n## Optional moving-interference candidates\n\n'
                    'These independent exports are candidates for review. They do not '
                    'replace the stationary presentation or select a manuscript block. '
                    'Their editable source and summaries are in '
                    '[the moving notebook](../../notebooks/beamforming_moving_interference.ipynb).\n\n')
                for name in sorted(moving_copies):
                    stream.write('- ['+Path(name).name+']('
                        +Path(name).relative_to('results/main').as_posix()+').\n')
        documentation[path.relative_to(root).as_posix()] = sha(path)
    manifest = dict(schema=1, status='completed', producer='scripts/export_draft_figures.py',
        producer_sha256=sha(root/'scripts/export_draft_figures.py'),
        source_figure_manifests=sources, dpi=EXPECTED_DPI, paper_width_in=WIDTH_IN,
        source_validation_gates={record['validation_path']:record['validation_sha256']
                                 for record in meta.values()},
        section_order=list(SECTIONS), applications=list(EXAMPLES),
        page_partition_scope='Every page of the sixteen individual canonical PDFs per application appears exactly once in a complete supplementary canonical copy; the three duplicate bundles remain in canonical directories.',
        selection_scope='The revised main section uses six notebook PDFs. All canonical evidence is retained in supplementary; no simulations, fitting, settings, benchmarks or episodes are changed.',
        main_policy=MAIN, main_notebook_files=sorted(MAIN_NOTEBOOK_PDFS),
        optional_moving_candidates=moving_copies,
        optional_moving_scope='Independent candidates for review only; the six retained main figures and stationary tables are unchanged by this notebook.',
        main_caption_notes={name: MAIN_CAPTIONS[Path(name).parent.name, Path(name).stem]
                            for name in sorted(MAIN_NOTEBOOK_PDFS)},
        documentation=documentation, files=files)
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
