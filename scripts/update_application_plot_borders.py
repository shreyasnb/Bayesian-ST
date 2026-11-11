"""Patch and reexecute only ordinary 2D plots in the two application notebooks.

Run the entire command under the shared rendering lock, for example::

    flock /tmp/bayesian-st-notebook-render.lock .venv/bin/python \
        scripts/update_application_plot_borders.py --execute

Pure 3D figure blocks and their embedded outputs/files are retained exactly.
Mixed figures keep their 3D drawing source and apply borders to ordinary axes
only. Figure size, DPI, fonts, arrays, method settings and numerical cells are
never edited by this script. The default is a read-only inventory.
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
from tempfile import TemporaryDirectory

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.notebook_publication_style import (
    BORDER_BEGIN,BORDER_END,apply_publication_borders,figure_blocks,figure_kind,
)
BOOKS={'01_array.ipynb':'beamforming','03_contact.ipynb':'contact'}


def source(cell):
    value=cell.get('source','')
    return ''.join(value) if isinstance(value,list) else value


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def value_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                     ensure_ascii=False).encode()).hexdigest()


def without_borders(text):
    return re.sub(re.escape(BORDER_BEGIN)+r'.*?'+re.escape(BORDER_END)+r'\n?',
                  '',text,flags=re.DOTALL)


class _RemoveSpineVisibility(ast.NodeTransformer):
    def visit_Expr(self,node):
        call=node.value
        if (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
                and call.func.attr=='set_visible' and isinstance(call.func.value,ast.Subscript)
                and isinstance(call.func.value.value,ast.Attribute)
                and call.func.value.value.attr=='spines'):
            return None
        return self.generic_visit(node)


def numerical_ast_sha(text):
    tree=_RemoveSpineVisibility().visit(ast.parse(without_borders(text)))
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


def setup_spine_defaults(text):
    return re.sub(r"(['\"]axes\.spines\.(?:top|right)['\"]\s*:\s*)False",
                  lambda match:match.group(1)+'True',text)


def controls(text):
    keep={'DPI','FIGSIZE','LABEL_FONT_SIZE','TICK_FONT_SIZE','LEGEND_FONT_SIZE'}
    return {target.id:ast.dump(node.value,include_attributes=False)
        for node in ast.parse(text).body if isinstance(node,ast.Assign)
        for target in node.targets if isinstance(target,ast.Name) and target.id in keep}


def declaration(text,directory,example):
    assignments={target.id:node.value for node in ast.parse(text).body if isinstance(node,ast.Assign)
        for target in node.targets if isinstance(target,ast.Name)}
    pdf=assignments['PDF_PATH']
    if not isinstance(pdf,ast.BinOp) or not isinstance(pdf.op,ast.Div):
        raise ValueError('Expected an explicit local PDF basename')
    filename=ast.literal_eval(pdf.right)
    section=ast.literal_eval(assignments['DRAFT_SECTION'])
    if Path(filename).name!=filename or section not in ('main','supplementary'):
        raise ValueError('Unexpected plotting export declaration')
    paths=[Path(directory)/filename,Path('results')/section/example/filename]
    if 'PNG_PATH' in text or 'DRAFT_PNG_PATH' in text:
        paths.extend(path.with_suffix('.png') for path in paths[:2])
    return dict(filename=filename,section=section,paths=[p.as_posix() for p in paths])


def plan(root,names):
    notebooks={};rows=[]
    for name in names:
        notebook=json.loads((root/'notebooks'/name).read_text())
        notebooks[name]=notebook
        directory=notebook['metadata']['editable_plotting']['output_directory']
        for index,cell in enumerate(notebook['cells']):
            if cell.get('cell_type')!='code':continue
            old=source(cell)
            if 'plt.figure(' not in old and 'plt.subplots(' not in old:continue
            patched=apply_publication_borders(old)
            before_blocks=figure_blocks(old);after_blocks=figure_blocks(patched)
            assert len(before_blocks)==len(after_blocks)
            for block_index,(before,after) in enumerate(zip(before_blocks,after_blocks)):
                kind=figure_kind(before)
                if not any(kind.values()):continue
                route=declaration(before,directory,BOOKS[name])
                changed=before!=after
                if kind['2d']==0:assert not changed
                assert controls(before)==controls(after)
                assert numerical_ast_sha(before)==numerical_ast_sha(after)
                rows.append(dict(notebook=name,cell=index,cell_id=cell.get('id'),
                    block=block_index,changed=changed,pure3d=kind['3d']>0 and kind['2d']==0,
                    mixed=kind['3d']>0 and kind['2d']>0,axes_constructors=kind,
                    source_before_sha256=hashlib.sha256(before.encode()).hexdigest(),
                    source_after_sha256=hashlib.sha256(after.encode()).hexdigest(),
                    numerical_ast_sha256=numerical_ast_sha(before),
                    controls=controls(before),**route))
    return notebooks,rows


def protection(root,names,rows):
    targets={path for row in rows if row['changed'] for path in row['paths']}
    folders=[]
    for name in names:
        example=BOOKS[name]
        folders.extend(['results/'+example,'results/main/'+example,
                        'results/supplementary/'+example,'results/notebook_exports/'+Path(name).stem])
    paths=set()
    for folder in folders:
        paths.update(p for p in (root/folder).rglob('*') if p.is_file())
    mappings={}
    for name in names:
        notebook=json.loads((root/'notebooks'/name).read_text())
        meta=notebook['metadata']['editable_plotting']
        for key in ('input_hashes','data_hashes','loader_sources'):
            mappings.update(meta.get(key,{}))
    paths.update(root/name for name in mappings)
    return {str(path.relative_to(root)):sha(path) for path in sorted(paths)
            if str(path.relative_to(root)) not in targets}


AUDIT_SOURCE = '''
# Independent verification of the visible border statements; not stored in the notebook.
_border_ordinary=[]
for _check_ax in fig.axes:
    if (_check_ax.name=='3d' or _check_ax.get_label()=='<colorbar>'
            or getattr(_check_ax,'_colorbar',None) is not None):
        continue
    _spines={_side:dict(visible=bool(_check_ax.spines[_side].get_visible()),
                       linewidth=float(_check_ax.spines[_side].get_linewidth()))
             for _side in ('left','right','bottom','top')}
    assert all(_value['visible'] and abs(_value['linewidth']-.9)<1e-12 for _value in _spines.values())
    assert _check_ax.xaxis.get_label_position()=='bottom'
    assert _check_ax.yaxis.get_label_position()=='left'
    for _axis in (_check_ax.xaxis,_check_ax.yaxis):
        for _tick in _axis.get_major_ticks()+_axis.get_minor_ticks():
            assert _tick._tickdir=='in'
            assert _tick.tick1line.get_visible() and _tick.tick2line.get_visible()
            assert _tick.label1.get_visible() and not _tick.label2.get_visible()
    _border_ordinary.append(_spines)
_BORDER_AUDIT.append(dict(notebook=BOOK_NAME,cell=CELL_INDEX,block=BLOCK_INDEX,axes=_border_ordinary))
'''


def execute(root,name,before,rows,timeout,directory):
    import nbformat
    from nbclient import NotebookClient
    selected=[row for row in rows if row['notebook']==name and row['changed']]
    if not selected:return before,[]
    cells=[nbformat.v4.new_code_cell(setup_spine_defaults(source(before['cells'][1])))]
    setup=next(cell for cell in before['cells'] if cell.get('metadata',{}).get('supplementary_setup'))
    cells.append(nbformat.v4.new_code_cell(setup_spine_defaults(source(setup))))
    cells.append(nbformat.v4.new_code_cell('_BORDER_AUDIT=[]'))
    mapping=[]
    for row in selected:
        index,block=row['cell'],row['block']
        text=figure_blocks(apply_publication_borders(source(before['cells'][index])))[block]
        check=AUDIT_SOURCE.replace('BOOK_NAME',repr(name)).replace('CELL_INDEX',str(index)).replace('BLOCK_INDEX',str(block))
        mapping.append((len(cells),row))
        cells.append(nbformat.v4.new_code_cell(text+'\n'+check))
    cells.append(nbformat.v4.new_code_cell('import json\nprint(json.dumps(_BORDER_AUDIT))'))
    mini=nbformat.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    for key,value in {'MPLCONFIGDIR':'/tmp/bayesian-st-matplotlib',
        'JUPYTER_CONFIG_DIR':'/tmp/bayesian-st-jupyter-config',
        'JUPYTER_DATA_DIR':'/tmp/bayesian-st-jupyter-data',
        'JUPYTER_RUNTIME_DIR':'/tmp/bayesian-st-jupyter',
        'IPYTHONDIR':'/tmp/bayesian-st-ipython'}.items():os.environ.setdefault(key,value)
    try:
        NotebookClient(mini,timeout=timeout,kernel_name='python3',
            resources={'metadata':{'path':str(root)}}).execute()
    finally:
        nbformat.write(mini,Path(directory)/(Path(name).stem+'-border-execution.ipynb'))
    after=deepcopy(before);new_images={}
    for execution_index,row in mapping:
        outputs=json.loads(json.dumps(mini.cells[execution_index].outputs))
        images=[out for out in outputs if 'image/png' in out.get('data',{})]
        assert len(images)==1 and not any(out.get('output_type')=='error' for out in outputs)
        new_images[row['cell'],row['block']]=images[0]
    setup_indices={1,next(i for i,cell in enumerate(before['cells'])
                          if cell.get('metadata',{}).get('supplementary_setup'))}
    for index in setup_indices:
        after['cells'][index]['source']=setup_spine_defaults(source(before['cells'][index])).splitlines(keepends=True)
    changed_cells={row['cell'] for row in selected}
    for index in changed_cells:
        old=before['cells'][index];updated=after['cells'][index]
        updated['source']=apply_publication_borders(source(old)).splitlines(keepends=True)
        image_indices=[i for i,out in enumerate(old['outputs']) if 'image/png' in out.get('data',{})]
        assert len(image_indices)==len(figure_blocks(source(old)))
        for block,output_index in enumerate(image_indices):
            if (index,block) in new_images:
                updated['outputs'][output_index]=new_images[index,block]
        assert old['metadata']==updated['metadata']
    for index,(old,new) in enumerate(zip(before['cells'],after['cells'])):
        if index not in changed_cells|setup_indices:assert old==new
        if index in setup_indices:
            assert source(new)==setup_spine_defaults(source(old))
            assert old['outputs']==new['outputs'] and old['metadata']==new['metadata']
            continue
        if old.get('cell_type')=='code':
            for block,(a,b) in enumerate(zip(figure_blocks(source(old)),figure_blocks(source(new)))):
                if (index,block) not in new_images:assert a==b
    output=''.join(item.get('text','') for item in mini.cells[-1].outputs if item.output_type=='stream')
    return after,json.loads(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--notebook',choices=tuple(BOOKS),action='append')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--timeout',type=int,default=900)
    parser.add_argument('--baseline',type=Path,default=Path('/tmp/application-plot-borders-baseline.json'))
    parser.add_argument('--report',type=Path,default=Path('/tmp/application-plot-borders-report.json'))
    args=parser.parse_args();root=args.root.resolve();names=args.notebook or list(BOOKS)
    notebooks,rows=plan(root,names)
    if not args.execute:
        print(json.dumps(dict(status='planned',figures=rows,
            changed_figures=sum(row['changed'] for row in rows),
            pure_3d_preserved=sum(row['pure3d'] for row in rows)),indent=2));return
    if args.baseline.exists():raise RuntimeError('Preserve the previous border baseline; pass a fresh --baseline path')
    protected=protection(root,names,rows)
    before_files={path:sha(root/path) for row in rows for path in row['paths'] if (root/path).is_file()}
    baseline=dict(notebooks=notebooks,protected_files=protected,figures=rows,output_hashes=before_files)
    args.baseline.write_text(json.dumps(baseline,indent=1)+'\n')
    execution=[]
    with TemporaryDirectory(prefix='application-border-render-') as directory:
        for name in names:
            print('Rendering ordinary 2D blocks:',name,flush=True)
            after,audit=execute(root,name,notebooks[name],rows,args.timeout,directory)
            current=json.loads((root/'notebooks'/name).read_text())
            assert current==notebooks[name],'Notebook changed during isolated rendering: '+name
            (root/'notebooks'/name).write_text(json.dumps(after,indent=1,ensure_ascii=False)+'\n')
            execution.extend(audit)
    changed=[name for name,digest in protected.items() if not (root/name).is_file() or sha(root/name)!=digest]
    assert not changed,('An unrelated source/data/figure changed',changed)
    outputs={path:dict(before=before_files.get(path),after=sha(root/path))
        for row in rows if row['changed'] for path in row['paths'] if (root/path).is_file()}
    result=dict(status='passed',notebooks=names,border_width_points=.9,
        tick_direction='in',ticks_all_four_sides=True,labels_bottom_left=True,
        changed_figures=sum(row['changed'] for row in rows),
        pure_3d_figures_preserved=sum(row['pure3d'] for row in rows),
        protected_files_verified=len(protected),all_protected_files_unchanged=True,
        numerical_ast_and_existing_controls_unchanged=True,
        setup_spine_defaults_only_updated=True,original_numerical_cells_unchanged=True,
        figures=rows,axes_checks=execution,output_hashes=outputs,
        baseline_sha256=sha(args.baseline),
        sources={str(Path(__file__).relative_to(root)):sha(__file__),
                 'scripts/notebook_publication_style.py':sha(root/'scripts/notebook_publication_style.py'),
                 'scripts/make_final_notebooks.py':sha(root/'scripts/make_final_notebooks.py'),
                 'scripts/append_supplementary_notebooks.py':sha(root/'scripts/append_supplementary_notebooks.py')})
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key not in
        ('figures','axes_checks','output_hashes')},indent=2))


if __name__=='__main__':main()
