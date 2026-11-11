"""Apply solid, marker-free trajectories to the application notebooks only.

The entire execution must be run under the shared notebook-render lock::

    flock /tmp/bayesian-st-notebook-render.lock .venv/bin/python \
        scripts/update_application_trajectory_style.py --execute

Pure 3D source blocks, outputs and files remain exact. Every changed 2D block
is checked against its previous artists before its replacement output is merged.
No simulation, parameter selection, library procedure or numerical cell changes.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import mkdtemp
from textwrap import indent

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.update_application_plot_borders import (
    BOOKS,source,sha,value_sha,controls,declaration,protection)
from scripts.notebook_publication_style import figure_blocks,figure_kind

OUTPUT=Path('results/application_trajectory_style_update')

KL_CONNECTOR_SOURCE=r'''# Evaluate the two endpoints with the same exact KL function and saved-input Fisher matrix.
from lib.final_information_visuals import exact_contact_kl
connector_angles = np.asarray(data["marked_angles"], dtype=float)
connector_kl = exact_contact_kl(np.asarray(data["marked_points"]), {"F": data["fisher_matrix"]})
np.testing.assert_allclose(connector_kl, data["marked_kl"], rtol=0., atol=1e-14)
ax.plot(connector_angles, connector_kl, "-", color="#8A8A8A", linewidth=1.,
        zorder=8, label="_contact_kl_connector")
for angle, divergence, label in zip(connector_angles, connector_kl, [r"$5^\circ$", r"$30^\circ$"]):
    ax.annotate(label, xy=(angle, divergence), xytext=(0, 9), textcoords="offset points",
                ha="center", va="bottom", fontsize=TICK_FONT_SIZE, zorder=9)
'''


def contact_kl_update(text):
    blocks=figure_blocks(text)
    for i,block in enumerate(blocks):
        if "'information_rotation.pdf'" not in block:
            continue
        old='["#D55E00", "#0072B2"], ["-", "--"], data["marked_points"],'
        new='["#D55E00", "#0072B2"], ["-", "-"], data["marked_points"],'
        if old in block:block=block.replace(old,new,1)
        assert new in block
        if '_contact_kl_connector' not in block:
            needle='ax.set(xlabel=r"$\\delta\\ ({}^\\circ)$"'
            assert block.count(needle)==1
            block=block.replace(needle,KL_CONNECTOR_SOURCE+needle,1)
        blocks[i]=block
    return ''.join(blocks)


def plan(root,names,originals=None):
    from scripts.notebook_trajectory_style import (
        apply_method_trajectory_style,trajectory_style_report,assert_trajectory_array_parity)
    notebooks={};patched={};rows=[]
    for name in names:
        before=deepcopy(originals[name]) if originals is not None else json.loads((root/'notebooks'/name).read_text())
        notebooks[name]=before;after=deepcopy(before)
        directory=before['metadata']['editable_plotting']['output_directory']
        for index,cell in enumerate(before['cells']):
            if cell.get('cell_type')!='code':
                continue
            text=source(cell)
            if 'plt.figure(' not in text and 'plt.subplots(' not in text:
                continue
            intended=contact_kl_update(text) if name=='03_contact.ipynb' else text
            new=apply_method_trajectory_style(intended)
            assert apply_method_trajectory_style(new)==new,'Style transform is not idempotent'
            assert_trajectory_array_parity(intended,new)
            old_blocks=figure_blocks(text);new_blocks=figure_blocks(new)
            assert len(old_blocks)==len(new_blocks)
            for block,(a,b) in enumerate(zip(old_blocks,new_blocks)):
                kind=figure_kind(a)
                if not any(kind.values()):
                    continue
                pure3d=kind['3d']>0 and kind['2d']==0
                if pure3d:
                    assert a==b,'A pure 3D source block changed'
                assert controls(a)==controls(b),'Canvas/font controls changed'
                old_route=declaration(a,directory,BOOKS[name])
                assert old_route==declaration(b,directory,BOOKS[name]),'Export routing changed'
                rows.append(dict(notebook=name,cell=index,cell_id=cell.get('id'),block=block,
                    changed=a!=b,pure3d=pure3d,mixed=kind['3d']>0 and kind['2d']>0,
                    source_before_sha256=hashlib.sha256(a.encode()).hexdigest(),
                    source_after_sha256=hashlib.sha256(b.encode()).hexdigest(),
                    controls=controls(a),decisions=trajectory_style_report(a),**old_route))
            if text!=new:
                after['cells'][index]['source']=new.splitlines(keepends=True)
        for cell in after['cells']:
            if cell.get('id')=='supp-noise-adaptation-2-note':
                cell['source']=source(cell).replace('Markers are estimates used before receiving',
                    'Stored values are estimates used before receiving').splitlines(keepends=True)
        patched[name]=after
    return notebooks,patched,rows


CAPTURE_SOURCE=r'''
import hashlib as _hashlib
import json as _json
import numpy as _np
from matplotlib import colors as _audit_colors

def _audit_array(value):
    if value is None:
        return None
    a=_np.asarray(_np.ma.getdata(value))
    if a.dtype.kind in 'OU':
        payload=_json.dumps(a.tolist(),sort_keys=True,default=str).encode()
    else:
        payload=_np.ascontiguousarray(a).tobytes()
    record=dict(shape=list(a.shape),dtype=str(a.dtype),sha256=_hashlib.sha256(payload).hexdigest())
    if _np.ma.isMaskedArray(value):
        record['mask_sha256']=_hashlib.sha256(_np.ascontiguousarray(_np.ma.getmaskarray(value)).tobytes()).hexdigest()
    return record

def _audit_value(value):
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    if isinstance(value,(_np.integer,_np.floating)):
        return value.item()
    if isinstance(value,(tuple,list)):
        return [_audit_value(x) for x in value]
    return _audit_array(value)

def _audit_rgba(value):
    return list(_audit_colors.to_rgba(value))

def _capture_artists(figure):
    data=dict(size_in=figure.get_size_inches().tolist(),dpi=float(figure.dpi),axes=[],legends=[])
    styles=[]
    for ai,ax in enumerate(figure.axes):
        ordinary=ax.name!='3d'
        item=dict(name=ax.name,xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),
            xscale=ax.get_xscale(),yscale=ax.get_yscale(),
            labels=[ax.get_xlabel(),ax.get_ylabel(),ax.get_title()],
            label_fonts=[float(ax.xaxis.label.get_fontsize()),float(ax.yaxis.label.get_fontsize()),float(ax.title.get_fontsize())],
            lines=[],collections=[],patches=[],images=[],texts=[],
            spines={side:[bool(spine.get_visible()),float(spine.get_linewidth())]
                    for side,spine in ax.spines.items()})
        if not ordinary:
            item['zlim']=list(ax.get_zlim())
        for li,line in enumerate(ax.lines):
            if hasattr(line,'get_data_3d'):
                coordinates=[_audit_array(x) for x in line.get_data_3d()]
            else:
                coordinates=[_audit_array(line.get_xdata(orig=False)),_audit_array(line.get_ydata(orig=False))]
            item['lines'].append(dict(coordinates=coordinates,color=_audit_rgba(line.get_color()),
                linewidth=float(line.get_linewidth()),alpha=_audit_value(line.get_alpha()),
                drawstyle=line.get_drawstyle(),zorder=float(line.get_zorder()),label=line.get_label()))
            marker=str(line.get_marker());linestyle=str(line.get_linestyle())
            count=len(line.get_xdata(orig=False))
            styles.append(dict(axes=ai,line=li,label=line.get_label(),points=count,
                marker=marker,linestyle=linestyle,markevery=_audit_value(line.get_markevery()),
                scientific_point=(count<=1 or (linestyle in ('None','none','',' ') and marker not in ('None','none','',' '))),
                reference_transform=line.get_transform() is not ax.transData))
        for collection in ax.collections:
            c=dict(type=type(collection).__name__,alpha=_audit_value(collection.get_alpha()),
                zorder=float(collection.get_zorder()),clim=_audit_value(collection.get_clim()),
                values=_audit_array(collection.get_array()),linewidths=_audit_array(collection.get_linewidths()))
            if hasattr(collection,'_vec'):
                c['vertices_3d']=_audit_array(collection._vec)
            elif hasattr(collection,'_offsets3d'):
                c['offsets_3d']=[_audit_array(x) for x in collection._offsets3d]
            elif hasattr(collection,'_segments3d'):
                c['segments_3d']=[_audit_array(x) for x in collection._segments3d]
            elif hasattr(collection,'U') and hasattr(collection,'V'):
                c['U']=_audit_array(collection.U);c['V']=_audit_array(collection.V)
                c['offsets']=_audit_array(collection.get_offsets())
            else:
                c['offsets']=_audit_array(collection.get_offsets())
                if hasattr(collection,'get_segments'):
                    c['segments']=[_audit_array(x) for x in collection.get_segments()]
                c['paths']=[_audit_array(p.vertices) for p in collection.get_paths()]
            if hasattr(collection,'get_sizes'):
                c['sizes']=_audit_array(collection.get_sizes())
            if not ordinary:
                for key in ('_facecolor3d','_edgecolor3d'):
                    if hasattr(collection,key):c[key]=_audit_array(getattr(collection,key))
            else:
                collection.update_scalarmappable()
                c['facecolors']=_audit_array(collection.get_facecolors())
                c['edgecolors']=_audit_array(collection.get_edgecolors())
            item['collections'].append(c)
        for patch in ax.patches:
            p=dict(type=type(patch).__name__,color=_audit_rgba(patch.get_facecolor()),
                edgecolor=_audit_rgba(patch.get_edgecolor()),linewidth=float(patch.get_linewidth()),
                alpha=_audit_value(patch.get_alpha()),zorder=float(patch.get_zorder()),hatch=patch.get_hatch())
            if hasattr(patch,'_posA_posB'):
                p['endpoints']=_audit_value(patch._posA_posB)
            else:
                p['path']=_audit_array(patch.get_path().vertices)
            for field in ('get_x','get_y','get_width','get_height','get_angle','get_xy','get_center'):
                if hasattr(patch,field):p[field]=_audit_value(getattr(patch,field)())
            item['patches'].append(p)
        for image in ax.images:
            item['images'].append(dict(values=_audit_array(image.get_array()),
                extent=_audit_value(image.get_extent()),clim=_audit_value(image.get_clim()),
                alpha=_audit_value(image.get_alpha()),cmap=image.get_cmap().name,
                interpolation=image.get_interpolation()))
        item['texts']=[dict(text=t.get_text(),size=float(t.get_fontsize()),
                            weight=str(t.get_fontweight())) for t in ax.texts]
        legend=ax.get_legend()
        if legend is not None:
            item['legend']=[dict(text=t.get_text(),size=float(t.get_fontsize())) for t in legend.get_texts()]
        data['axes'].append(item)
    for legend in figure.legends:
        data['legends'].append([dict(text=t.get_text(),size=float(t.get_fontsize())) for t in legend.get_texts()])
    return dict(data=data,styles=styles)

def _compare_artists(before,after,identifier):
    geometry=None
    if identifier=='03_contact.ipynb:5:1':
        # This explicitly requested connector is additional scientific
        # geometry; the original four lines and all their arrays stay exact.
        import copy as _copy
        after=_copy.deepcopy(after)
        ax=after['data']['axes'][0]
        indices=[i for i,line in enumerate(ax['lines']) if line['label']=='_contact_kl_connector']
        assert indices==[len(ax['lines'])-1]
        connector=ax['lines'].pop(indices[0])
        expected_angles=_np.asarray(data['marked_angles'],float)
        expected_kl=exact_contact_kl(_np.asarray(data['marked_points']),{'F':data['fisher_matrix']})
        _np.testing.assert_allclose(expected_kl,data['marked_kl'],rtol=0.,atol=1e-14)
        assert connector['coordinates']==[_audit_array(expected_angles),_audit_array(expected_kl)]
        assert connector['color']==_audit_rgba('#8A8A8A') and connector['linewidth']==1. and connector['zorder']==8.
        assert expected_angles.tolist()==[5.,30.] and abs(expected_kl[0]-expected_kl[1])>1e-4
        labels=[t for t in ax['texts'] if t['text'] in (r'$5^\circ$',r'$30^\circ$')]
        assert len(labels)==2
        ax['texts']=[t for t in ax['texts'] if t['text'] not in (r'$5^\circ$',r'$30^\circ$')]
        after['styles']=[s for s in after['styles'] if s['label']!='_contact_kl_connector']
        geometry=dict(angles_deg=expected_angles.tolist(),divergences=expected_kl.tolist(),
            exact_original_function_and_saved_inputs=True,artificial_equal_height=False,
            linewidth_points=1.,color='#8A8A8A',zorder=8,endpoint_labels=[t['text'] for t in labels])
    assert before['data']==after['data'], 'Numerical/artist properties changed: '+identifier
    assert len(before['styles'])==len(after['styles']),identifier
    changed=[]
    for a,b in zip(before['styles'],after['styles']):
        assert [a[k] for k in ('axes','line','label','points')]==[b[k] for k in ('axes','line','label','points')]
        if a['scientific_point'] or a['reference_transform']:
            assert (a['marker'],a['linestyle'],a['markevery'])==(b['marker'],b['linestyle'],b['markevery']), 'Scientific point/reference changed: '+identifier
        if (a['marker'],a['linestyle'],a['markevery'])!=(b['marker'],b['linestyle'],b['markevery']):
            if a['marker']!=b['marker']:
                assert b['marker'] in ('None','none','',' '),'A new marker was introduced'
            if a['linestyle']!=b['linestyle']:
                assert b['linestyle']=='-','A method trajectory is not solid'
            if a['markevery']!=b['markevery']:
                assert b['markevery'] is None,'A marker stagger remains'
            changed.append(dict(before=a,after=b))
    return dict(identifier=identifier,data_sha256=_hashlib.sha256(_json.dumps(before['data'],sort_keys=True).encode()).hexdigest(),
        axes=len(before['data']['axes']),line_artists=len(before['styles']),
        collection_artists=sum(len(x['collections']) for x in before['data']['axes']),
        patch_artists=sum(len(x['patches']) for x in before['data']['axes']),
        image_artists=sum(len(x['images']) for x in before['data']['axes']),
        unchanged_numerical_arrays_colors_widths_shading_fonts_limits=True,style_changes=changed,
        additional_requested_kl_geometry=geometry)
_TRAJECTORY_AUDIT=[]
'''


def baseline_function(text,identifier):
    import ast
    lines=text.splitlines(keepends=True)
    edits=[]
    for node in ast.parse(text).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SAVE_PDF' for t in node.targets):
            assert isinstance(node.value,ast.Constant) and node.value.value is True
            edits.append((node.value.lineno-1,node.value.col_offset,node.value.end_col_offset,'False'))
        if (isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
                and isinstance(node.value.func,ast.Attribute)
                and isinstance(node.value.func.value,ast.Name)
                and node.value.func.value.id=='plt' and node.value.func.attr=='show'):
            edits.append((node.lineno-1,node.col_offset,node.end_col_offset,'plt.close(fig)'))
    for row,start,end,value in sorted(edits,reverse=True):
        lines[row]=lines[row][:start]+value+lines[row][end:]
    return ('def _original_plot_for_parity():\n'+indent(''.join(lines),'    ')
        +'\n    return _capture_artists(fig)\n'
        +'_original_artists = _original_plot_for_parity()\n')


def execute(root,name,before,after,rows,timeout,directory,cache=None):
    import nbformat
    from nbclient import NotebookClient
    selected=[row for row in rows if row['notebook']==name and row['changed']]
    if not selected:
        return after,[]
    setup=next(cell for cell in before['cells'] if cell.get('metadata',{}).get('supplementary_setup'))
    cells=[nbformat.v4.new_code_cell(source(before['cells'][1])),
           nbformat.v4.new_code_cell(source(setup)),nbformat.v4.new_code_cell(CAPTURE_SOURCE)]
    mapping=[];replacement={};cached_audits=[]
    for row in selected:
        index,block=row['cell'],row['block']
        key=(name,index,block)
        if cache and key in cache:
            replacement[index,block]=cache[key]['image']
            cached_audits.append(cache[key]['audit'])
            continue
        old=figure_blocks(source(before['cells'][index]))[block]
        new=figure_blocks(source(after['cells'][index]))[block]
        identifier=f"{name}:{index}:{block}"
        cells.append(nbformat.v4.new_code_cell(baseline_function(old,identifier)))
        mapping.append((len(cells),row))
        cells.append(nbformat.v4.new_code_cell(new+'\n_TRAJECTORY_AUDIT.append(_compare_artists('
            '_original_artists,_capture_artists(fig),'+repr(identifier)+'))\n'
            +'print("TRAJECTORY_PARITY_RECORD "+_json.dumps(_TRAJECTORY_AUDIT[-1]))\n'))
    cells.append(nbformat.v4.new_code_cell('import json\nprint(json.dumps(_TRAJECTORY_AUDIT))'))
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
        nbformat.write(mini,Path(directory)/(Path(name).stem+'-trajectory-parity.ipynb'))
    for position,row in mapping:
        outputs=json.loads(json.dumps(mini.cells[position].outputs))
        images=[x for x in outputs if 'image/png' in x.get('data',{})]
        assert len(images)==1 and not any(x.get('output_type')=='error' for x in outputs)
        replacement[row['cell'],row['block']]=images[0]
    for index in {row['cell'] for row in selected}:
        old=before['cells'][index];new=after['cells'][index]
        indices=[i for i,out in enumerate(old['outputs']) if 'image/png' in out.get('data',{})]
        assert len(indices)==len(figure_blocks(source(old)))
        for block,output in enumerate(indices):
            if (index,block) in replacement:
                new['outputs'][output]=replacement[index,block]
        assert old['metadata']==new['metadata']
    log=''.join(x.get('text','') for x in mini.cells[-1].outputs if x.output_type=='stream')
    return after,cached_audits+json.loads(log)


def completed_render_cache(root,baseline,books,patched,rows,paths):
    """Reuse only fully asserted renders with exact old/new source and assets."""
    from datetime import datetime
    cached={}
    old_script='scripts/update_application_trajectory_style.py'
    digest=baseline['source_hashes'][old_script]
    archive=root/OUTPUT/'source_archive'/digest/old_script
    assert sha(archive)==digest
    old_tree=ast.parse(archive.read_text())
    old_capture=next(ast.literal_eval(node.value) for node in old_tree.body
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CAPTURE_SOURCE' for t in node.targets))
    for path in paths:
        mini=json.loads(Path(path).read_text())
        proof_bytes=Path(path).read_bytes()
        proof=root/OUTPUT/'execution_proofs'/(hashlib.sha256(proof_bytes).hexdigest()+'.ipynb.gz')
        proof.parent.mkdir(exist_ok=True)
        if not proof.exists():proof.write_bytes(gzip.compress(proof_bytes,mtime=0))
        name=Path(path).name.split('-trajectory-parity.ipynb')[0]+'.ipynb'
        assert name in books
        assert source(mini['cells'][2])==old_capture,'The cached numerical guard differs'
        assert source(mini['cells'][0])==source(books[name]['cells'][1])
        old_rows=[r for r in baseline['figures'] if r['notebook']==name and r['changed']]
        for j,row in enumerate(old_rows):
            pos=4+2*j
            if pos>=len(mini['cells']):break
            cell=mini['cells'][pos]
            if cell.get('execution_count') is None or any(x.get('output_type')=='error' for x in cell.get('outputs',[])):
                continue
            images=[x for x in cell.get('outputs',[]) if 'image/png' in x.get('data',{})]
            if len(images)!=1:continue
            key=(name,row['cell'],row['block'])
            current=next((x for x in rows if (x['notebook'],x['cell'],x['block'])==key and x['changed']),None)
            if current is None:continue
            new=figure_blocks(source(patched[name]['cells'][row['cell']]))[row['block']]
            old=figure_blocks(source(books[name]['cells'][row['cell']]))[row['block']]
            rendered=source(cell).split('\n_TRAJECTORY_AUDIT.append')[0]
            if rendered!=new or source(mini['cells'][pos-1])!=baseline_function(old,''):
                continue
            execution=cell.get('metadata',{}).get('execution',{})
            start=execution.get('iopub.status.busy');end=execution.get('iopub.status.idle')
            if not start or not end:continue
            lower=datetime.fromisoformat(start.replace('Z','+00:00')).timestamp()-.2
            upper=datetime.fromisoformat(end.replace('Z','+00:00')).timestamp()+.2
            if not all((root/p).is_file() and lower<=(root/p).stat().st_mtime<=upper for p in row['paths']):
                continue
            cached[key]=dict(image=images[0],audit=dict(identifier=f'{name}:{row["cell"]}:{row["block"]}',
                reused_completed_render=True,unchanged_numerical_arrays_colors_widths_shading_fonts_limits=True,
                exact_before_after_source_and_controls=True,
                original_guard_sha256=hashlib.sha256(old_capture.encode()).hexdigest(),
                execution_notebook=str(proof.relative_to(root)),execution_notebook_sha256=sha(path),
                compressed_execution_proof_sha256=sha(proof),
                asserted_execution_cell_sha256=value_sha(cell),
                output_hashes={p:sha(root/p) for p in row['paths']}))
    return cached


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--notebook',choices=tuple(BOOKS),action='append')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--timeout',type=int,default=900)
    parser.add_argument('--baseline-commit',default='2e2767a')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--resume-execution',type=Path,action='append',default=[])
    args=parser.parse_args();root=args.root.resolve();names=args.notebook or list(BOOKS)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    assert head.startswith(args.baseline_commit),'The preserved user baseline changed'
    out=root/OUTPUT
    initial=None;originals=None
    if args.resume:
        initial=json.loads((out/'baseline.json').read_text())
        assert initial['commit']==head
        originals={}
        for name in names:
            raw=subprocess.check_output(['git','show',head+':notebooks/'+name],cwd=root)
            assert hashlib.sha256(raw).hexdigest()==initial['notebook_hashes'][name]
            originals[name]=json.loads(raw)
    before,after,rows=plan(root,names,originals)
    if not args.execute:
        print(json.dumps(dict(status='planned',changed_blocks=sum(x['changed'] for x in rows),
            preserved_pure3d=sum(x['pure3d'] for x in rows),figures=rows),indent=2));return
    out.mkdir(parents=True,exist_ok=True)
    if (out/'baseline.json').exists() and not args.resume:
        raise RuntimeError('Keep the preceding baseline; use a fresh output root for another update')
    protected=protection(root,names,rows)
    for folder in ('results/beamforming_comparison_audit/moving/mean_validation',
                   'results/beamforming_comparison_audit/stationary/mean_validation'):
        for path in (root/folder).rglob('*'):
            if path.is_file():protected[str(path.relative_to(root))]=sha(path)
    for path in (root/'notebooks').glob('*.ipynb'):
        if path.name not in names:protected[str(path.relative_to(root))]=sha(path)
    if initial:
        for name,digest in initial['protected_files'].items():
            if ('/data/' in name or '/tables/' in name or name.startswith('lib/')
                    or name.endswith(('.npz','.npy','.csv','.csv.gz'))):
                assert sha(root/name)==digest,'Numerical evidence changed between attempts: '+name
    before_files=(initial['output_hashes'] if initial else
        {path:sha(root/path) for row in rows for path in row['paths'] if (root/path).is_file()})
    bindings={name:sha(root/name) for name in ('scripts/update_application_trajectory_style.py',
        'scripts/notebook_trajectory_style.py','scripts/notebook_publication_style.py',
        'scripts/notebook_plot_cells_contact.py')}
    for name,digest in bindings.items():
        archive=out/'source_archive'/digest/name
        archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists():archive.write_bytes((root/name).read_bytes())
        assert sha(archive)==digest
    baseline=dict(commit=head,notebook_hashes={name:sha(root/'notebooks'/name) for name in names},
        cell_hashes={name:[value_sha(cell) for cell in before[name]['cells']] for name in names},
        protected_files=protected,output_hashes=before_files,source_hashes=bindings,figures=rows)
    if not args.resume:
        (out/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
    else:
        (out/'resume_preservation.json').write_text(json.dumps(baseline,indent=2)+'\n')
    cache=completed_render_cache(root,initial,before,after,rows,args.resume_execution) if args.resume else {}
    print('Verified completed render cache:',len(cache),flush=True)
    audits=[]
    # Failed mini notebooks remain in /tmp for diagnosis; successful runs keep
    # only the compact artist-array fingerprints in the presentation report.
    directory=Path(mkdtemp(prefix='application-trajectory-parity-'))
    print('Temporary execution notebooks:',directory,flush=True)
    for name in names:
        print('Checking and rendering changed 2D blocks:',name,flush=True)
        live=json.loads((root/'notebooks'/name).read_text())
        assert live['metadata']==before[name]['metadata'],'Notebook metadata changed outside this update'
        for old,new,current in zip(before[name]['cells'],after[name]['cells'],live['cells']):
            assert current['metadata']==old['metadata']
            assert source(current) in (source(old),source(new)),'Unrelated source changed before resume'
        updated,audit=execute(root,name,before[name],after[name],rows,args.timeout,directory,cache)
        assert json.loads((root/'notebooks'/name).read_text())==live, 'Notebook changed during rendering'
        allowed={row['cell'] for row in rows if row['notebook']==name and row['changed']}
        for i,(old,new) in enumerate(zip(before[name]['cells'],updated['cells'])):
            if i not in allowed and old.get('id')!='supp-noise-adaptation-2-note':
                assert old==new,'Unrelated notebook cell changed'
            if old.get('cell_type')=='code':
                images_old=[x for x in old.get('outputs',[]) if 'image/png' in x.get('data',{})]
                images_new=[x for x in new.get('outputs',[]) if 'image/png' in x.get('data',{})]
                for block,(a,b) in enumerate(zip(figure_blocks(source(old)),figure_blocks(source(new)))):
                    if figure_kind(a)['2d']==0 and figure_kind(a)['3d']>0:
                        assert a==b and images_old[block]==images_new[block], 'Pure3D changed'
        assert before[name]['metadata']==updated['metadata']
        (root/'notebooks'/name).write_text(json.dumps(updated,indent=1,ensure_ascii=False)+'\n')
        audits.extend(audit)
    changed=[name for name,digest in protected.items() if not (root/name).is_file() or sha(root/name)!=digest]
    assert not changed,('Unrelated data/source/figure changed',changed)
    assert all(sha(root/name)==digest for name,digest in bindings.items()),'A producer source changed during execution'
    outputs={path:dict(before=before_files.get(path),after=sha(root/path))
        for row in rows if row['changed'] for path in row['paths'] if (root/path).is_file()}
    report=dict(status='passed',baseline_commit=head,notebooks=names,
        notebook_hashes={name:sha(root/'notebooks'/name) for name in names},
        changed_blocks=sum(x['changed'] for x in rows),
        preserved_pure3d=sum(x['pure3d'] for x in rows),
        protected_files_verified=len(protected),all_protected_files_unchanged=True,
        figures=rows,artist_array_parity=audits,output_hashes=outputs,source_hashes=bindings,
        baseline_sha256=sha(out/'baseline.json'),new_simulations=0,new_parameter_selections=0,
        reused_completed_renders=len(cache),
        historical_baseline_sources={name:dict(sha256=digest,archive=str(
            OUTPUT/'source_archive'/digest/name)) for name,digest in
            (initial or baseline)['source_hashes'].items()},
        scope='Only method trajectory marker/linestyle controls and two matching '
            'marker descriptions changed. Standalone scientific points, references, '
            'all numerical arrays, widths/colors/bands/fonts/layouts, input evidence, '
            'pure3D blocks and export routes remain unchanged.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    shutil.rmtree(directory)
    print(json.dumps({k:v for k,v in report.items() if k not in
        ('figures','artist_array_parity','output_hashes')},indent=2))


if __name__=='__main__':
    main()
