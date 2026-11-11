"""Render title-free final paper figures from saved numerical artifacts only."""
from pathlib import Path
from datetime import datetime, timezone
import os
import argparse
import json
import sys
import traceback
import copy
import shutil
import subprocess
import tempfile
os.environ.setdefault('MPLCONFIGDIR', '/tmp/bayesian-st-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/bayesian-st-cache')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from matplotlib.backends.backend_pdf import PdfPages
from lib.final_paper_visuals import (
    CATALOGUE, BUNDLES, CAPTIONS, PLACEMENT, CAMERA, FIGURE_WIDTH, DPI,
    load_evidence, paper_style, prepare_paper_axes, verify_title_free, sha, plt)
from lib.final_validation import require_gate, gate_path, current_gate_path, verify_mapping
from lib.rendering_provenance import resolve_source_mapping, archive_manifest_provenance

DEPENDENCIES = {
    'setup': {'performance'},
    'task_performance': {'performance'},
    'posterior_anisotropy': {'performance'},
    'noise_adaptation': {'prediction'},
    'information': {'theory', 'performance'},
    'posterior_curvature': {'theory'},
    'tracking_bound': {'theory'},
    'block_updates': {'theory', 'performance', 'computation'},
    'additional_comparisons': {'performance'},
    'variance_estimation': set(),
    'predictive_calibration': {'prediction'},
    'predictive_scores': {'prediction'},
    'model_mismatch': {'prediction'},
    'computation_time': {'computation'},
    'numerical_accuracy': {'theory', 'computation'},
    'paper_summary': {'performance', 'prediction', 'computation'},
}
SOURCES = ('lib/final_paper_visuals.py', 'scripts/plot_final_simulations.py',
           'lib/plotting.py', 'lib/distribution_geometry_visuals.py',
           'lib/final_information_visuals.py', 'lib/rendering_provenance.py')


def json_safe(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def counts(evidence):
    """Record configured budgets and actual repetition identifiers, not pixels."""
    result = {}
    for component, tables in evidence.tables.items():
        result[component] = {}
        for name, frame in tables.items():
            entry = {'rows': len(frame)}
            for column in ('trial', 'episode', 'repetition', 'repeat'):
                if column in frame:
                    entry['distinct_' + column] = int(frame[column].nunique())
            for column in ('numerical_ok', 'mean_ok', 'tail_ok', 'quantile_ok', 'fit_ok'):
                if column in frame:
                    entry['unresolved_' + column] = int((~frame[column].map(lambda value: value is True or str(value).lower() == 'true')).sum())
            if 'repetitions' in frame:
                entry['recorded_clock_repetitions'] = sorted(int(value) for value in frame.repetitions.dropna().unique())
            result[component][name] = entry
    return result


def page_notes(figures):
    records = []
    for i, fig in enumerate(figures, 1):
        record = {'page': i, 'figure_size_in': list(fig.get_size_inches()),
                  'axes': len(fig.axes), 'titles': False,
                  'title_audit': dict(figure_title=False,axes_with_title=0,
                                     preview_dpi=fig.dpi,latex=True),
                  'font_sizes_pt': sorted(set(float(text.get_fontsize())
                     for ax in fig.axes for text in [ax.xaxis.label,ax.yaxis.label,*ax.texts]))}
        if hasattr(fig, '_final_condition'):
            record['condition'] = fig._final_condition
        if hasattr(fig, '_final_conditions'):
            record['conditions'] = list(fig._final_conditions)
        if hasattr(fig,'_final_selection'):record['selection']=fig._final_selection
        if hasattr(fig,'_final_repetition_unit'):record['repetition_unit']=fig._final_repetition_unit
        record['panel_selections']=[getattr(ax,'_final_selection') for ax in fig.axes if hasattr(ax,'_final_selection')]
        record['sphere_displays']=[getattr(ax,'_sphere_display') for ax in fig.axes if hasattr(ax,'_sphere_display')]
        records.append(record)
    return records


def configurations(evidence):
    return {key:meta.get('config',{}) for key,meta in {**evidence.components,**evidence.references}.items()}


def selected_fields(value, terms, path=''):
    result={}
    if isinstance(value,dict):
        for name,item in value.items():
            location=path+'.'+name if path else name
            if any(term in name.lower() for term in terms):result[location]=item
            else:result.update(selected_fields(item,terms,location))
    elif isinstance(value,list):
        for index,item in enumerate(value):result.update(selected_fields(item,terms,path+f'[{index}]'))
    return result


def scientific_metadata(evidence):
    configs=configurations(evidence)
    return dict(configurations=configs,
        seed_sets={name:selected_fields(config,('seed','stage_offset','generator_offset')) for name,config in configs.items()},
        repetition_unit='Independent episode; within-episode snapshots, queries and reporting times are not independent repetitions. Repeated clocks are separately labelled timing repetitions.',
        repetition_counts={**counts(evidence),**{name:meta.get('tables',{}) for name,meta in evidence.references.items()}},
        numerical_tolerances={name:selected_fields(config,('tolerance','tol','floor','precision','order','threshold','draws','cap','digits','limit','resolution')) for name,config in configs.items()},
        numerical_status={name:meta.get('checks',{}) for name,meta in {**evidence.components,**evidence.references}.items()})


def selection_metadata(name,evidence,figures):
    inventory={}
    for component,tables in evidence.tables.items():
        inventory[component]={}
        for table,frame in tables.items():
            inventory[component][table]={column:frame[column].drop_duplicates().tolist()
                for column in ('condition','regime','stress','n','beta','design','nonuniform_prior','zero_signal') if column in frame}
    explicit=[getattr(fig,'_final_selection',{}) for fig in figures]
    return dict(display_selection=dict(figure=name,pages=explicit,
        rule='Predeclared representative selectors and full-page condition order are documented in pages and display_rules; no selection uses observed method advantage.'),
        omitted_conditions=dict(scope='Conditions outside each displayed selector are retained in the manifest-declared raw tables; representative main panels do not claim to display every condition.',
            complete_inventory=inventory,
            locations=['additional_comparisons.pdf','supp_comparisons.pdf',
                       *evidence.inputs.keys()]))


def extend_reference_sources(sources,evidence):
    for meta in evidence.references.values():
        for name,digest in meta['sources'].items():
            resolved=resolve_source_mapping(evidence.root,{name:digest})
            for path,expected in resolved.items():
                if path in sources and sources[path]!=expected:
                    raise RuntimeError('Incompatible source claims for historical reference: '+path)
                sources[path]=expected


def caption_for(name,evidence):
    caption=CAPTIONS[name]
    if name=='information' and evidence.example=='contact':
        from lib.final_information_visuals import CONTACT_INFORMATION_CAPTION
        caption=CONTACT_INFORMATION_CAPTION
    if name=='posterior_anisotropy':
        if 'Beamforming:' in caption:
            contact, beam = caption.split('Beamforming:', 1)
            caption = contact.removeprefix('Contact:').strip() if evidence.example == 'contact' else beam.strip()
        if evidence.example=='contact':
            caption+=' The actual posterior is the fixed concentrated-input simulation 0 at t=16. Directional density and repeated-simulation task performance are separate comparisons.'
        else:
            caption+=' Complementary mean eigenvalues compare the full and isotropic means. The lower panel is the empirical distribution of paired isotropic-minus-full exact-covariance power differences at n=64, t=T=200, in dB; it is not a confidence interval.'
    if name=='posterior_anisotropy' or (evidence.example=='contact' and name in ('paper_summary','noise_adaptation')):
        caption+=' Spheres retain the original saved physical/coefficient coordinates and shared oblique camera. The blue-purple-pink-yellow palette uses the unchanged shared logarithmic density scale; its four-decade display floor does not mean zero density. Color positions are not probability quantiles. Thin gray curves are front-visible coordinate great circles, not density contours or credible regions. No mode-facing rotation, density deformation or additional posterior fitting is used. In the directional-risk panel, theta is in radians and the indicated mean is either the full posterior mean or the manuscript-defined isotropic mean; each curve is its own belief risk.'
    if name=='predictive_calibration':
        target=(r'$q^H Pq$ for a known unit complex query $q$' if evidence.example=='beamforming'
            else r'$e_1^T Pq$ for a known unit real query $q$')
        caption+=' Both applications use n=3 reliability fixtures. On the inclusion pages, panel (a) checks latent-subspace inclusion in its HPD set; panel (b) reports the clean scalar '+target+r'; panel (c) reports the noisy scalar $\operatorname{Re}(u_{t,1})$. Contact observations are real. On the appended size pages, panels (a), (b), (c) show the state-ball chordal radius, clean scalar width and noisy scalar width, respectively.'
        caption+=' These are '+str(evidence.config('prediction')['calibration_test_episodes'])+' independent test episodes per fixed checkpoint and regime.'
    if name=='task_performance':
        count=evidence.config('performance')['test_episodes']
        caption+=' This cohort has '+str(count)+' independent test episodes per condition; snapshots and queries are averaged within each episode.'
    return caption


def render(root, example, stage, selected, out, engineering=False, *, fixed_stationary=False):
    components = sorted(set().union(*(DEPENDENCIES[name] for name in selected)))
    # An EM reference page alone has no new numerical dependency, but still
    # carries the current theory component when making a published figure set.
    if not components:
        components = ['theory']
    evidence = load_evidence(root, example, stage, components=components,
                             require_general=not engineering)
    source_hashes = resolve_source_mapping(root,{name:sha(root/name) for name in SOURCES})
    for component_meta in evidence.components.values():
        source_hashes.update(resolve_source_mapping(root,component_meta['sources']))
    if 'stationary_T200' in evidence.components:
        certificate=current_gate_path(root)
        evidence.inputs[str(certificate.relative_to(root))]=sha(certificate)
    initial_inputs = dict(evidence.inputs)
    out.mkdir(parents=True, exist_ok=True)
    preview = out/'preview'
    preview.mkdir(exist_ok=True)
    files, entries, figures_by_name, failures = {}, {}, {}, {}
    with paper_style():
        for name in selected:
            try:
                figures = CATALOGUE[name](evidence)
                if fixed_stationary and name=='additional_comparisons':
                    for unchanged in figures[3:]:plt.close(unchanged)
                    figures=figures[:3]
                if fixed_stationary and name=='noise_adaptation':
                    main=[fig for fig in figures if getattr(fig,'_final_condition',None)=='noise_change']
                    if len(main)!=1:raise RuntimeError('One saved noise-change page is required')
                    for unchanged in figures:
                        if unchanged not in main:plt.close(unchanged)
                    figures=main
                if not figures:
                    raise RuntimeError('A required figure has no completed numerical panels')
                for fig in figures:
                    prepare_paper_axes(fig)
                    if len([ax for ax in fig.axes if ax.get_label() != '<colorbar>']) > 1:
                        for ax in fig.axes:
                            for text in list(ax.texts):
                                if text.get_text() in ('(a)', '(b)', '(c)', '(d)', '(e)', '(f)'):
                                    text.remove()
                    verify_title_free(fig)
                    if not abs(float(fig.get_figwidth())-FIGURE_WIDTH)<1e-12:
                        raise RuntimeError('Every final paper page must be seven inches wide')
                extend_reference_sources(source_hashes,evidence)
                path = out/(name+'.pdf')
                with PdfPages(path, metadata={'Title': '', 'Author': '',
                                             'Subject': '', 'Keywords': ''}) as pdf:
                    for fig in figures:
                        pdf.savefig(fig, dpi=DPI, bbox_inches=None)
                outputs = {path.name: sha(path)}
                for index, fig in enumerate(figures):
                    suffix = '' if index == 0 else f'_page{index+1}'
                    png = preview/(name+suffix+'.png')
                    fig.savefig(png, dpi=DPI, bbox_inches=None)
                    outputs[str(png.relative_to(out))] = sha(png)
                files.update(outputs)
                figures_by_name[name] = figures
                entries[name+'.pdf'] = dict(
                    status='completed', experiment=example,
                    formula_or_claim=caption_for(name,evidence),
                    producer='scripts/plot_final_simulations.py',
                    routine='lib.final_paper_visuals.'+name,
                    **scientific_metadata(evidence),
                    source_hashes=source_hashes, data_manifest_hashes=dict(evidence.inputs),
                    data_hashes={path:expected for meta in evidence.components.values() for path,expected in meta['artifacts'].items()},
                    data_hash_scope='data_manifest_hashes closes over each component manifest, whose artifacts map verifies every raw array/table. Historical source/table manifests are verified independently of obsolete global gates.',
                    **selection_metadata(name,evidence,figures),
                    proposed_latex_caption=caption_for(name,evidence),
                    placement=('supplementary' if example=='beamforming' and name=='task_performance'
                               else PLACEMENT.get(name,'supplementary')),
                    pages=page_notes(figures), output_hashes=outputs,
                    interval_scope='Pointwise confidence intervals across independent episodes or explicitly identified posterior/repeated-episode bands; see caption and raw tables.',
                    density_scope=('Real antipodally symmetric normal-line density'
                                   if example == 'contact' else 'Complex density sections are explicitly restrictions, never full or normalized sphere marginals'),
                    camera=CAMERA if name in ('posterior_anisotropy', 'paper_summary', 'setup', 'noise_adaptation') else None)
                print('Rendered', example, name, len(figures), 'page(s)', flush=True)
            except Exception as error:
                failures[name] = dict(status='unresolved', error=type(error).__name__+': '+str(error),
                                      traceback=traceback.format_exc())
                print('Unresolved', example, name, type(error).__name__, str(error), flush=True)
        full_catalogue = set(selected) == set(CATALOGUE)
        if full_catalogue and not failures:
            for name, parts in BUNDLES.items():
                pages = [fig for part in parts for fig in figures_by_name[part]]
                path = out/(name+'.pdf')
                with PdfPages(path, metadata={'Title': '', 'Subject': ''}) as pdf:
                    for fig in pages:
                        verify_title_free(fig)
                        pdf.savefig(fig, dpi=DPI, bbox_inches=None)
                outputs = {path.name: sha(path)}
                files.update(outputs)
                entries[name+'.pdf'] = dict(status='completed', experiment=example,
                    producer='scripts/plot_final_simulations.py',
                    formula_or_claim='Readable multi-page bundle of the individually documented panels.',
                    constituent_outputs=[part+'.pdf' for part in parts],
                    proposed_latex_caption=' '.join(caption_for(part,evidence) for part in parts),
                    placement='supplementary', pages=page_notes(pages),
                    **scientific_metadata(evidence),
                    **selection_metadata(name,evidence,pages),
                    source_hashes=source_hashes, data_manifest_hashes=dict(evidence.inputs),
                    data_hashes={path:expected for meta in evidence.components.values() for path,expected in meta['artifacts'].items()},
                    output_hashes=outputs, alias_scope='Reuses the same saved arrays and Matplotlib figures, not additional independent experiments.')
        for figures in figures_by_name.values():
            for fig in figures:
                plt.close(fig)
    # Reference reads may have added verified historical input manifests.
    verify_mapping(root, source_hashes)
    verify_mapping(root, evidence.inputs)
    validation = None if engineering else gate_path(root)
    meta = dict(
        schema=1, status='engineering_preview' if engineering else
            'completed' if full_catalogue and not failures else 'incomplete',
        example=example, stage=stage, created_utc=datetime.now(timezone.utc).isoformat(),
        component_stages={key:meta['stage'] for key,meta in evidence.components.items()},
        sources=source_hashes, input_manifests=evidence.inputs,
        validation_sha256=sha(validation) if validation is not None else None,
        validation_path=str(validation.relative_to(root)) if validation is not None else None,
        figures=entries, files=files, unresolved=failures,
        text_usetex=True, preview_dpi=DPI, rasterized_layer_dpi=DPI,
        pdf_scope='Vector text, axes and lines; density surfaces and heatmap layers may be rasterized. PDF has no global dpi.',
        titles=False, shared_camera=CAMERA, paper_width_in=FIGURE_WIDTH,
        display_rules=dict(setup='Predeclared trial0 and saved setup arrays',
            stationary='Fixed T=200 at n=32,64,128; actual measurement indices t=25,50,100,200 in additional comparisons',
            paired_moving='Predeclared smooth condition when available, contact sphere; all conditions retained in supplementary',
            contact_anisotropy='Predeclared stationary concentrated-input posterior; query family1 in saved coordinates',
            beam_anisotropy='Predeclared n64, t200, real-coefficient density section; no density inferred from isotropic mean',
            prediction='One page per declared condition or probability regime, fixed configuration order',
            numerical_errors='Numerical zero discrepancies use a 1e-17 display floor; unresolved targets are counted separately'),
        scope='Rendering reads verified saved arrays and tables only. No simulation, posterior integration or resampling, fitting, hyperparameter tuning, or favorable trajectory selection occurs here.')
    (out/'figure_manifest.json').write_text(json.dumps(json_safe(meta), indent=2, allow_nan=False)+'\n')
    notes = ['# Proposed figure captions', '',
             'These notes are external to the title-free PDFs. Page and panel conditions are in figure_manifest.json.', '']
    for name, entry in entries.items():
        notes.extend(['## '+name, '', entry['proposed_latex_caption'], ''])
        for page in entry.get('pages', []):
            if page.get('condition') or page.get('conditions'):
                notes.append('Page '+str(page['page'])+': '+str(page.get('condition', page.get('conditions')))+'.')
        notes.append('')
    (out/'caption_notes.md').write_text('\n'.join(notes).rstrip()+'\n')
    if failures:
        raise RuntimeError(f'{example}: unresolved required figures: '+', '.join(failures))
    return meta



FIXED_STATIONARY_FIGURES = ('setup','task_performance','posterior_anisotropy',
                           'additional_comparisons','paper_summary')


def _pdf_objects(destination, sections):
    """Copy native PDF page objects; never rasterize retained pages."""
    code = """import fitz,json,sys
request=json.loads(sys.argv[1]);result=fitz.open()
for part in request['sections']:
    with fitz.open(part['path']) as source:
        first=part.get('first',0);last=part.get('last',len(source)-1)
        if first<=last:result.insert_pdf(source,from_page=first,to_page=last)
result.set_metadata({'title':'','author':'','subject':'','keywords':''})
result.save(request['destination'],garbage=0,deflate=False)
result.close()
"""
    candidates=list(dict.fromkeys([sys.executable,shutil.which('python'),'/usr/bin/python3']))
    interpreter=None
    for candidate in candidates:
        if candidate and subprocess.run([candidate,'-c','import fitz'],capture_output=True).returncode==0:
            interpreter=candidate;break
    if interpreter is None:raise RuntimeError('Lossless PDF composition requires PyMuPDF in an available Python interpreter')
    request={'destination':str(destination),'sections':[{'path':str(part[0]),**part[1]} for part in sections]}
    subprocess.run([interpreter,'-c',code,json.dumps(request)],check=True)


def _mapping_union(records,key):
    result={}
    for record in records:
        for path,digest in record.get(key,{}).items():
            if path in result and result[path]!=digest:
                raise RuntimeError('Conflicting preserved provenance: '+key+'/'+path)
            result[path]=digest
    return result


def _write_caption_notes(out,entries):
    notes=['# Proposed figure captions','',
        'These notes are external to the title-free PDFs. Page and panel conditions are in figure_manifest.json.','']
    for name,entry in entries.items():
        notes.extend(['## '+name,'',entry['proposed_latex_caption'],''])
        for page in entry.get('pages',[]):
            if page.get('condition') or page.get('conditions'):
                notes.append('Page '+str(page['page'])+': '+str(page.get('condition',page.get('conditions')))+'.')
        notes.append('')
    (out/'caption_notes.md').write_text('\n'.join(notes).rstrip()+'\n')


def refresh_fixed_stationary(root,stage='paper',out=None):
    """Refresh changed stationary data and the first noise-change page only."""
    from lib.final_stationary_evidence import stationary_component_path
    root=Path(root).resolve();out=root/'results/beamforming' if out is None else Path(out).resolve()
    if stage!='paper':raise ValueError('A partial publication refresh requires the paper stage')
    manifest=out/'figure_manifest.json'
    original=json.loads(manifest.read_text())
    if original.get('status')!='completed' or len(original.get('figures',{}))!=19:
        raise RuntimeError('Partial refresh requires the complete existing nineteen-PDF catalogue')
    verify_mapping(out,original['files'])
    preserved=archive_manifest_provenance(root,original)
    for mapping in (preserved['sources'],preserved['input_manifests']):verify_mapping(root,mapping)
    stationary_manifest=stationary_component_path(root,stage)
    stationary_name=str(stationary_manifest.relative_to(root))
    stationary_entry=original['figures']['task_performance.pdf']
    stationary_current=(stationary_entry.get('data_manifest_hashes',{}).get(stationary_name)==sha(stationary_manifest)
        and stationary_entry.get('configurations',{}).get('performance',{}).get('stationary_T')==200)
    selected=([] if stationary_current else list(FIXED_STATIONARY_FIGURES))+['noise_adaptation']
    bundles=[name for name,parts in BUNDLES.items() if set(parts).intersection(selected)]
    # Complete rendering/composition in staging before changing publication files.
    with tempfile.TemporaryDirectory(prefix='bayesian-st-fixed-stationary-') as temporary:
        staged=Path(temporary)
        revised=render(root,'beamforming',stage,selected,staged,fixed_stationary=True)
        entries=copy.deepcopy(preserved['figures']);files=dict(preserved['files'])
        partial_pages={'additional_comparisons':(3,8),'noise_adaptation':(1,3)}
        for name in selected:
            key=name+'.pdf';entry=copy.deepcopy(revised['figures'][key])
            if name in partial_pages:
                replace_count,total_count=partial_pages[name];old_entry=preserved['figures'][key]
                if len(old_entry['pages'])!=total_count or len(entry['pages'])!=replace_count:
                    raise RuntimeError('Unexpected partial page inventory for '+name)
                composed=staged/(name+'-composed.pdf')
                _pdf_objects(composed,[(staged/key,{}),(out/key,{'first':replace_count})])
                composed.replace(staged/key)
                entry['pages'].extend(copy.deepcopy(old_entry['pages'][replace_count:]))
                for field in ('source_hashes','data_manifest_hashes','data_hashes'):
                    entry[field]=_mapping_union([old_entry,entry],field)
                entry['output_hashes']={**old_entry['output_hashes'],**entry['output_hashes'],key:sha(staged/key)}
                entry['page_provenance']=[dict(pages=list(range(1,replace_count+1)),action='rendered',
                    source_hashes=revised['figures'][key]['source_hashes'],
                    data_manifest_hashes=revised['figures'][key]['data_manifest_hashes']),
                    dict(pages=list(range(replace_count+1,total_count+1)),action='retained native PDF page objects',
                    original_pdf_sha256=original['files'][key],source_hashes=old_entry['source_hashes'],
                    data_manifest_hashes=old_entry['data_manifest_hashes'])]
            entries[key]=entry;files.update(entry['output_hashes'])
        for bundle in bundles:
            key=bundle+'.pdf';parts=[part+'.pdf' for part in BUNDLES[bundle]]
            _pdf_objects(staged/key,[(staged/part if (staged/part).exists() else out/part,{}) for part in parts])
            entry=copy.deepcopy(preserved['figures'][key]);part_entries=[entries[part] for part in parts]
            for field in ('source_hashes','data_manifest_hashes','data_hashes'):
                entry[field]=_mapping_union(part_entries,field)
            for field in ('configurations','seed_sets','repetition_counts','numerical_tolerances','numerical_status'):
                entry[field]={key:value for part in part_entries for key,value in part.get(field,{}).items()}
            entry['pages']=[]
            for part,part_entry in zip(parts,part_entries):
                for page in part_entry['pages']:
                    page=copy.deepcopy(page);page['constituent_page']=page['page'];page['constituent_output']=part
                    page['page']=len(entry['pages'])+1;entry['pages'].append(page)
            entry['proposed_latex_caption']=' '.join(entries[part]['proposed_latex_caption'] for part in parts)
            entry['producer']='scripts/plot_final_simulations.py'
            entry['composition']='Native PDF objects copied from constituent documents; retained figures were not rerendered'
            entry['constituent_provenance']={part:{field:entries[part][field]
                for field in ('source_hashes','data_manifest_hashes','output_hashes')} for part in parts}
            entry['output_hashes']={key:sha(staged/key)};entries[key]=entry;files[key]=sha(staged/key)
        entries['task_performance.pdf'].update(placement='supplementary',
            replacement_main_table='results/main/beamforming/stationary_T200_table.tex')
        merged=copy.deepcopy(preserved)
        display_rules=dict(preserved.get('display_rules',{}))
        if not stationary_current:
            display_rules.update({key:revised['display_rules'][key] for key in ('stationary','beam_anisotropy')})
        merged.update(status='completed',created_utc=datetime.now(timezone.utc).isoformat(),
            figures=entries,files=files,unresolved={},
            sources=_mapping_union([preserved,revised,*[{'sources':item['source_hashes']} for item in entries.values()]],'sources'),
            input_manifests=_mapping_union([preserved,revised],'input_manifests'),
            component_stages={**preserved.get('component_stages',{}),**revised['component_stages']},
            display_rules=display_rules,
            partial_refresh=dict(study='stationary_T200',presentation='noise_change_at_129',
                stationary_already_current=stationary_current,rendered_figures=selected,
                recomposed_bundles=bundles,retained_noise_adaptation_pages=[2,3],
                retained_additional_comparisons_pages=[4,5,6,7,8],
                original_manifest_sha256=sha(manifest),
                preserved_outputs={name:digest for name,digest in original['files'].items()
                    if name not in revised['files'] and name not in {bundle+'.pdf' for bundle in bundles}}))
        certificate=current_gate_path(root)
        merged['stationary_validation_path']=str(certificate.relative_to(root))
        merged['stationary_validation_sha256']=sha(certificate)
        changed={name for name,digest in files.items() if original['files'].get(name)!=digest}
        if len(entries)!=19 or sum(name.endswith('.png') for name in files)!=39:
            raise RuntimeError('Partial refresh changed the complete catalogue or preview inventory')
        for name in changed:
            source=staged/name
            if not source.is_file() or sha(source)!=files[name]:raise RuntimeError('Missing staged replacement: '+name)
        for name in changed:
            destination=out/name;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(staged/name,destination)
        verify_mapping(out,files)
        for name,digest in merged['partial_refresh']['preserved_outputs'].items():
            if sha(out/name)!=digest:raise RuntimeError('Unrelated publication output changed: '+name)
        for mapping in (merged['sources'],merged['input_manifests']):verify_mapping(root,mapping)
        manifest.write_text(json.dumps(json_safe(merged),indent=2,allow_nan=False)+'\n')
        _write_caption_notes(out,entries)
    contact=root/'results/contact/figure_manifest.json'
    contact_meta=json.loads(contact.read_text());resolved=archive_manifest_provenance(root,contact_meta)
    verify_mapping(contact.parent,resolved['files']);verify_mapping(root,resolved['sources'])
    if resolved!=contact_meta:contact.write_text(json.dumps(json_safe(resolved),indent=2,allow_nan=False)+'\n')
    print('Refreshed requested publication pages; all unrelated PDF/PNG hashes retained',flush=True)
    return merged

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['verify', 'pilot', 'paper'], default='paper')
    parser.add_argument('--example', choices=['beamforming', 'contact', 'both'], default='both')
    parser.add_argument('--only', nargs='+', choices=list(CATALOGUE))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--engineering-preview', action='store_true')
    parser.add_argument('--replace-fixed-stationary',action='store_true',
        help='Refresh changed T200 figures and noise-change page1 while preserving all unrelated publication pages')
    args = parser.parse_args()
    if args.engineering_preview and (args.output is None or not str(args.output.resolve()).startswith('/tmp/')):
        parser.error('Engineering previews must use an explicit /tmp output folder')
    if not args.engineering_preview:
        require_gate(ROOT)
    if args.replace_fixed_stationary:
        if args.engineering_preview or args.only or args.example=='contact':
            parser.error('--replace-fixed-stationary requires publication beamforming output without --only')
        refresh_fixed_stationary(ROOT,args.stage,args.output)
        return
    selected = args.only or list(CATALOGUE)
    examples = ['beamforming', 'contact'] if args.example == 'both' else [args.example]
    for example in examples:
        if args.output:
            out = args.output/example if len(examples) > 1 else args.output
        elif args.stage == 'paper':
            out = ROOT/'results'/example
        else:
            out = Path('/tmp/final_simulation_preview')/args.stage/example
        render(ROOT, example, args.stage, selected, out, args.engineering_preview)


if __name__ == '__main__':
    main()
