"""Update only beamforming cells affected by the fixed-T stationary cohort.

--prepare changes sources, explanatory notes and provenance, without executing
cells. --execute runs only the affected cells in an isolated temporary notebook
and merges their outputs. With neither flag both operations run. The moving
comparison in the two-figure task cell is never executed or replaced.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
NOTEBOOK='notebooks/01_array.ipynb'
TARGET_IDS={'e796892d','f2772ed9','stationary-dimensions-T200'}
BASENAMES={'task_performance_output_power.pdf','noise_adaptation_page1.pdf',
           'stationary_dimensions_T200.pdf'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(cell):
    value=cell['source']
    return ''.join(value) if isinstance(value,list) else value


def set_source(cell,value):
    cell['source']=value.splitlines(keepends=True)


def first_figure(source_text):
    """Split after the first top-level plt.show(), retaining its exact suffix."""
    lines=source_text.splitlines(keepends=True)
    for node in ast.parse(source_text).body:
        call=node.value if isinstance(node,ast.Expr) else None
        if (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
                and isinstance(call.func.value,ast.Name) and call.func.value.id=='plt'
                and call.func.attr=='show'):
            return ''.join(lines[:node.end_lineno]),''.join(lines[node.end_lineno:])
    raise RuntimeError('Task plotting cell has no top-level plt.show()')


def figure_key(cell):
    info=cell.get('metadata',{})
    return info.get('figure_id'),int(info.get('source_page',1))


def target_indices(notebook):
    selected=[i for i,c in enumerate(notebook['cells']) if c.get('id') in TARGET_IDS]
    if {notebook['cells'][i]['id'] for i in selected}!=TARGET_IDS:
        raise RuntimeError('Missing a power, variance or stationary-dimension cell')
    return selected


def pdf_hashes(root):
    folders=['results/notebook_exports/01_array','results/main/beamforming',
             'results/supplementary/beamforming']
    return {str(path.relative_to(root)):sha(path) for name in folders
            for path in (root/name).glob('*.pdf')}


def unaffected_pdf_changes(root,before):
    after=pdf_hashes(root)
    return [name for name,digest in before.items()
            if Path(name).name not in BASENAMES and after.get(name)!=digest]


def write_notebook(path,notebook):
    temporary=path.with_suffix('.fixed-stationary.tmp')
    temporary.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n')
    temporary.replace(path)


def refresh_metadata(root,notebook):
    from lib.notebook_inputs import verify_saved_inputs
    from lib.final_stationary_evidence import verify_stationary_component,stationary_component_path
    verify_stationary_component(root,stage='paper')
    info=notebook['metadata']['editable_plotting']
    info['input_hashes']=verify_saved_inputs(root,'beamforming')
    for name in list(info.get('loader_sources',{})):
        info['loader_sources'][name]=sha(root/name)
    info.setdefault('loader_sources',{})['lib/final_stationary_evidence.py']=sha(root/'lib/final_stationary_evidence.py')
    final=notebook['metadata']['final_study']
    final['figure_manifest_sha256']=sha(root/'results/beamforming/figure_manifest.json')
    final['producer_sha256']=sha(root/'scripts/make_final_notebooks.py')
    for name in list(final.get('reference_previews',{})):
        final['reference_previews'][name]=sha(root/name)
    supplement=notebook['metadata'].get('supplementary_appendix',{})
    supplement['source_figure_manifest_sha256']=final['figure_manifest_sha256']
    for name in list(supplement.get('generation_sources',{})):
        supplement['generation_sources'][name]=sha(root/name)
    manifest_path=stationary_component_path(root,stage='paper')
    notebook['metadata']['fixed_stationary_update'].update(
        stationary_T=200,stationary_primary_t=200,
        evidence_manifest=str(manifest_path.relative_to(root)),evidence_manifest_sha256=sha(manifest_path),
        producer='scripts/update_fixed_stationary_notebook.py',producer_sha256=sha(__file__))


def prepare(root):
    from scripts.notebook_plot_cells_beamforming import get_plot_cell
    from scripts.make_final_notebooks import MAIN_PLOT_NOTES
    path=root/NOTEBOOK
    notebook=json.loads(path.read_text());before=deepcopy(notebook)
    previous_task=next(c for c in notebook['cells'] if c.get('id')=='e796892d')
    if previous_task.get('metadata',{}).get('figure_id')=='abrupt_tracking':
        for extension in ('pdf','png'):
            current=root/'results/main/beamforming'/('task_performance_output_power.'+extension)
            archived=root/'results/beamforming_moving/figures'/('abrupt_previous_main.'+extension)
            if not archived.is_file() or not current.is_file() or sha(archived)!=sha(current):
                raise RuntimeError('Preserve the prior abrupt figure before restoring the stationary plot: '+str(archived))
    handle,baseline_name=tempfile.mkstemp(prefix='beamforming-plot-update-baseline-',suffix='.json',dir='/tmp')
    os.close(handle);baseline=Path(baseline_name)
    other_notebooks={str(p.relative_to(root)):sha(p) for p in (root/'notebooks').glob('*.ipynb') if p!=path}
    baseline.write_text(json.dumps(dict(notebook=before,pdfs=pdf_hashes(root),other_notebooks=other_notebooks),indent=1)+'\n')
    notebook['metadata']['fixed_stationary_update']={'baseline_path':str(baseline)}
    task=next(c for c in notebook['cells'] if c.get('id')=='e796892d')
    _,original_moving=first_figure(source(task))
    new_first,_=first_figure(get_plot_cell('task_performance',1))
    set_source(task,new_first+original_moving)
    task['metadata'].update(figure_id='task_performance',source_page=1,source_catalogue_page=True)
    noise=next(c for c in notebook['cells'] if c.get('id')=='f2772ed9')
    set_source(noise,get_plot_cell('noise_adaptation',1))
    dimension_id='stationary-dimensions-T200'
    existing=next((c for c in notebook['cells'] if c.get('id')==dimension_id),None)
    if existing is None:
        dimension=dict(cell_type='code',id=dimension_id,metadata={
            'editable_plotting_role':'plot','figure_id':'task_performance','source_page':1,
            'source_catalogue_page':False},execution_count=None,outputs=[],
            source=get_plot_cell('stationary_dimensions',1).splitlines(keepends=True))
        after=next(i for i,c in enumerate(notebook['cells']) if c.get('id')=='dc1fa354')
        notebook['cells'].insert(after+1,dimension)
        appendix=notebook['metadata']['supplementary_appendix']
        appendix['original_cell_count']+=1
    else:
        set_source(existing,get_plot_cell('stationary_dimensions',1))
        existing['metadata']['source_catalogue_page']=False
    changed_ids=set(TARGET_IDS)
    note=next(c for c in notebook['cells'] if c.get('id')=='484a19bd')
    value=source(note)
    value=value.replace(
        'The plotted output power is measured on independent held-out snapshots and normalized by this reference power before conversion to dB.',
        'The stationary curve uses covariance-based relative power after correction at measurement 200. Independent held-out snapshots are used only for parameter selection and the separately retained historical comparisons.')
    value=value.replace(
        'Panel (a) uses stationary trials with the observation count equal to the number of sensors. Panel (b) uses moving interference:',
        'The first figure uses stationary simulations with exactly 200 observations at every sensor count. The separate boxplot retains the existing moving-interference comparison:')
    starts=[position for phrase in ('The stationary curves use','The main tracking plot')
            if (position:=value.find(phrase))>=0]
    if starts:
        value=value[:min(starts)]+MAIN_PLOT_NOTES['beamforming_stationary'].strip()+'\n'
    else:
        value+='\n\n'+MAIN_PLOT_NOTES['beamforming_stationary'].strip()+'\n'
    set_source(note,value);changed_ids.add(note['id'])
    selected=target_indices(notebook)
    notebook['metadata']['fixed_stationary_update']['affected_code_cell_ids']=[notebook['cells'][i]['id'] for i in selected]
    info=notebook['metadata']['editable_plotting']
    info['figure_count']=sum(c.get('metadata',{}).get('editable_plotting_role')=='plot' for c in notebook['cells'])
    info['rendered_figure_count']=sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and isinstance(n.func.value,ast.Name) and n.func.value.id=='plt' and n.func.attr in ('figure','subplots')
        for c in notebook['cells'] if c['cell_type']=='code' for n in ast.walk(ast.parse(source(c))))
    refresh_metadata(root,notebook)
    before_by_id={c['id']:c for c in before['cells']}
    if [c['id'] for c in notebook['cells'] if c['id'] in before_by_id]!=[c['id'] for c in before['cells']]:
        raise RuntimeError('Existing notebook IDs or ordering changed')
    preserved=[c['id'] for c in notebook['cells'] if c['id'] in before_by_id and c['id'] not in changed_ids]
    if any(c!=before_by_id[c['id']] for c in notebook['cells'] if c['id'] in preserved):
        raise RuntimeError('An unrelated cell changed during preparation')
    if first_figure(source(task))[1]!=original_moving:
        raise RuntimeError('Moving comparison source changed')
    write_notebook(path,notebook)
    return dict(stage='prepared',baseline=str(baseline),changed_cell_ids=sorted(changed_ids),
                preserved_cell_objects=len(preserved),affected_code_cell_indices=selected)


def image_outputs(cell):
    return [output for output in cell.get('outputs',[]) if 'image/png' in output.get('data',{})]


def execute(root,timeout):
    import nbformat
    from nbclient import NotebookClient
    path=root/NOTEBOOK
    initial=path.read_bytes();notebook=json.loads(initial)
    baseline_path=Path(notebook['metadata']['fixed_stationary_update']['baseline_path'])
    baseline=json.loads(baseline_path.read_text())
    selected=target_indices(notebook)
    old=deepcopy(notebook)
    pdfs_before=pdf_hashes(root)
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='1'
    os.environ.setdefault('MPLCONFIGDIR','/tmp/bayesian-st-matplotlib')
    os.environ.setdefault('JUPYTER_RUNTIME_DIR','/tmp/bayesian-st-jupyter')
    os.environ.setdefault('IPYTHONDIR','/tmp/bayesian-st-ipython')
    setup=next(i for i,c in enumerate(notebook['cells']) if c.get('id')=='9bd0f660')
    supplementary_setup=next(i for i,c in enumerate(notebook['cells']) if c.get('metadata',{}).get('supplementary_setup'))
    work=[];mapping=[]
    for index in [setup,supplementary_setup,*selected]:
        cell=deepcopy(notebook['cells'][index])
        if cell.get('id')=='e796892d':
            set_source(cell,first_figure(source(cell))[0])
        cell['outputs']=[];cell['execution_count']=None
        work.append(cell);mapping.append(index)
    mini=nbformat.reads(json.dumps(dict(nbformat=4,nbformat_minor=5,cells=work,
        metadata={'kernelspec':notebook['metadata']['kernelspec']})),as_version=4)
    NotebookClient(mini,timeout=timeout,kernel_name='python3',
        resources={'metadata':{'path':str(root)}}).execute()
    if path.read_bytes()!=initial:
        raise RuntimeError('Notebook changed during execution; refusing to overwrite it')
    for index,executed in zip(mapping,mini.cells):
        if index not in selected:
            continue
        cell=notebook['cells'][index]
        if cell.get('id')=='e796892d':
            new_images=image_outputs(executed)
            if len(new_images)!=1 or len(image_outputs(cell))!=2:
                raise RuntimeError('Task-cell output count is inconsistent with two independent figures')
            first=next(i for i,o in enumerate(cell['outputs']) if 'image/png' in o.get('data',{}))
            cell['outputs'][first:first+1]=json.loads(json.dumps(executed['outputs']))
            if image_outputs(cell)[1]!=image_outputs(old['cells'][index])[1]:
                raise RuntimeError('The moving comparison inline image changed')
        else:
            cell['outputs']=json.loads(json.dumps(executed['outputs']))
        cell['execution_count']=executed['execution_count']
        if 'execution' in executed.get('metadata',{}):
            cell['metadata']['execution']=json.loads(json.dumps(executed['metadata']['execution']))
    refresh_metadata(root,notebook)
    preserved=[i for i in range(len(notebook['cells'])) if i not in selected]
    if any(notebook['cells'][i]!=old['cells'][i] for i in preserved):
        raise RuntimeError('An unselected cell object changed during execution')
    if unaffected_pdf_changes(root,baseline['pdfs']):
        raise RuntimeError('An unrelated PDF changed: '+str(unaffected_pdf_changes(root,baseline['pdfs'])))
    for name,digest in baseline['other_notebooks'].items():
        if sha(root/name)!=digest:
            raise RuntimeError('An unrelated notebook changed: '+name)
    task_index=next(i for i in selected if notebook['cells'][i].get('id')=='e796892d')
    baseline_task=next(c for c in baseline['notebook']['cells'] if c.get('id')=='e796892d')
    if first_figure(source(notebook['cells'][task_index]))[1]!=first_figure(source(baseline_task))[1]:
        raise RuntimeError('Moving comparison source is not byte-identical to baseline')
    if image_outputs(notebook['cells'][task_index])[1]!=image_outputs(baseline_task)[1]:
        raise RuntimeError('Moving comparison inline output is not baseline-identical')
    moving_pdfs={name:digest for name,digest in baseline['pdfs'].items()
                 if Path(name).name=='task_performance_paired_difference.pdf'}
    if len(moving_pdfs)<2 or any(sha(root/name)!=digest for name,digest in moving_pdfs.items()):
        raise RuntimeError('The moving comparison local/manuscript PDFs changed')
    moving_image=image_outputs(notebook['cells'][task_index])[1]['data']['image/png']
    if isinstance(moving_image,list):
        moving_image=''.join(moving_image)
    current_by_id={c['id']:c for c in notebook['cells']}
    original_preserved=[c['id'] for c in baseline['notebook']['cells'] if c==current_by_id[c['id']]]
    write_notebook(path,notebook)
    return dict(stage='executed',baseline=str(baseline_path),executed_code_cell_indices=selected,
        preserved_prepared_cell_objects=len(preserved),baseline_identical_cell_objects=len(original_preserved),
        unchanged_other_notebooks=len(baseline['other_notebooks']),
        unrelated_pdfs_unchanged=len([name for name in pdfs_before if Path(name).name not in BASENAMES]),
        moving_inline_png_preserved=True,moving_source_suffix_preserved=True,
        moving_inline_png_sha256=hashlib.sha256(moving_image.encode()).hexdigest(),
        moving_pdf_preserved=True,moving_pdf_sha256=moving_pdfs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--timeout',type=int,default=900)
    parser.add_argument('--output',type=Path,help='Optional JSON preservation report')
    args=parser.parse_args();root=args.root.resolve();reports=[]
    if args.prepare or not args.execute:
        reports.append(prepare(root))
    if args.execute or not args.prepare:
        reports.append(execute(root,args.timeout))
    report={'status':'passed','notebook':NOTEBOOK,'operations':reports}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
