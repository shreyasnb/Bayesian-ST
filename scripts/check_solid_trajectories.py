"""Read-only source and PDF checks of solid, marker-free method trajectories.

No notebook, plotting code, simulation, parameter selection or timing is run.
Standalone scientific points, confidence-interval point estimates, boxplot
fliers, contour levels and mathematical/event reference lines are distinct
from connected method trajectories. Numerical preservation is checked by the
separate scientific audit scripts and exact presentation-update proofs.
"""
from __future__ import annotations

import argparse
import ast
import csv
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
NOTEBOOKS=tuple('notebooks/'+name+'.ipynb' for name in (
    '01_array','03_contact','beamforming_moving_interference',
    'contact_prediction_diagnostics','beamforming_moving_comparison_audit',
    'beamforming_stationary_comparison_audit'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(cell):
    value=cell.get('source','')
    return ''.join(value) if isinstance(value,list) else value


def qualified(node):
    if isinstance(node,ast.Name):return node.id
    if isinstance(node,ast.Attribute):return qualified(node.value)+'.'+node.attr
    return ''


def literal(node,default=None):
    try:return ast.literal_eval(node)
    except (ValueError,TypeError):return default


def _formats(call):
    # A format can follow a star-unpacked point, two coordinate arrays, or
    # three coordinate arrays. Literal formats contain no numerical data.
    return [arg.value for arg in call.args if isinstance(arg,ast.Constant)
            and isinstance(arg.value,str)]


def _point_format(value):
    # A color-only format such as 'k' still draws a solid trajectory. A
    # standalone marker format may additionally contain a one-letter color.
    markers=set('.,ov^<>12348spP*hH+xXDd|_')
    return (bool(value) and not any(style in value for style in ('-',':'))
            and any(character in markers for character in value)
            and all(character in markers or character in 'bgrcmykw' for character in value))


def _solid_format(value):
    if any(style in value for style in ('--','-.',':')):return False
    remainder=value.replace('-','')
    if remainder in ('','solid') or re.fullmatch(r'C\d+',remainder):return True
    return len(remainder)==1 and remainder in 'bgrcmykw'


def _reference(call):
    label=next((literal(item.value) for item in call.keywords if item.arg=='label'),None)
    if label in ('Distance ceiling','Theorem bound'):
        return True
    if (qualified(call.func).rsplit('.',1)[-1]=='step' and len(call.args)>=2
            and any(term in ast.unparse(call.args[1]).lower() for term in
                    ('truth','true_schedule','true_noise'))):
        return True
    if len(call.args)>=2 and ast.dump(call.args[0])==ast.dump(call.args[1]):
        return True
    first=literal(call.args[0]) if call.args else None
    second=literal(call.args[1]) if len(call.args)>1 else None
    return isinstance(first,(list,tuple)) and first==second


def _mathematical_context(call,parents,text):
    """Identify the notebook's explicit formula/constructed-law comparisons."""
    node=call
    while node in parents:
        node=parents[node]
        if not isinstance(node,ast.For):continue
        iterator=ast.unparse(node.iter)
        fields={item.id for item in ast.walk(node.target) if isinstance(item,ast.Name)}
        if 'kl_first' in iterator and 'kl_second' in iterator:
            return 'controlled_KL_rotation',node
        if 'load_constructed_geometry' in text and {'case','shade','style'}<=fields:
            return 'constructed_distribution',node
        if 'operatorname{tr}' in iterator and 'mathcal F' in iterator:
            return 'inverse_Fisher_reference',node
    return None,None


def _solid_expression(node,assigned):
    value=literal(node,object())
    if isinstance(value,str):return value in ('-','solid')
    if isinstance(node,ast.IfExp):
        return _solid_expression(node.body,assigned) and _solid_expression(node.orelse,assigned)
    if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name):
        values=literal(assigned.get(node.value.id))
        return isinstance(values,dict) and bool(values) and all(v in ('-','solid') for v in values.values())
    if isinstance(node,ast.Name) and node.id in assigned:
        return _solid_expression(assigned[node.id],assigned)
    if isinstance(node,ast.Name) and node.id=='linestyle_':
        # The moving notebook exposes (label,color,linestyle) triples.
        values=literal(assigned.get('STYLE'))
        return isinstance(values,dict) and bool(values) and all(
            isinstance(v,(tuple,list)) and len(v)==3 and v[-1]=='-' for v in values.values())
    return False


def inspect_cell(cell,index,endpoint_point=False):
    text=source(cell);tree=ast.parse(text)
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)]
    parents={child:node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    assigned={target.id:node.value for node in tree.body if isinstance(node,ast.Assign)
              for target in node.targets if isinstance(target,ast.Name)}
    trajectories=[];points=[];references=[];errors=[]
    for call in calls:
        name=qualified(call.func);kind=name.rsplit('.',1)[-1]
        if kind=='scatter':points.append(dict(line=call.lineno,kind='scatter'));continue
        if kind in ('axhline','axvline','hlines','vlines','contour','contourf'):
            references.append(dict(line=call.lineno,kind=kind));continue
        if kind=='boxplot':points.append(dict(line=call.lineno,kind='boxplot'));continue
        if kind not in ('plot','step','semilogx','semilogy','loglog','errorbar'):continue
        formats=_formats(call)
        kwargs={item.arg:item.value for item in call.keywords}
        label=literal(kwargs.get('label'))
        if (endpoint_point and kind=='plot' and qualified(call.func)=='tracking_ax.plot'
                and label in ('Tracking error','Theorem bound')):
            points.append(dict(line=call.lineno,kind='verified_single_checkpoint_comparison'));continue
        if kind=='errorbar':
            fmt=literal(kwargs.get('fmt'))
            if fmt in (None,'none','None','') or (isinstance(fmt,str) and _point_format(fmt)):
                points.append(dict(line=call.lineno,kind='interval_point_or_bars'));continue
            # Symbolic per-method errorbar formats in these notebooks describe
            # forest-plot points, not connected arrays. There is no line style.
            if not isinstance(fmt,str) and 'linestyle' not in kwargs:
                points.append(dict(line=call.lineno,kind='interval_point'));continue
        if formats and all(_point_format(value) for value in formats):
            points.append(dict(line=call.lineno,kind='standalone_plot_points'));continue
        if _reference(call):
            references.append(dict(line=call.lineno,kind='mathematical_line'));continue
        family,reference_loop=_mathematical_context(call,parents,text)
        if family:
            references.append(dict(line=call.lineno,kind='mathematical_curve',family=family))
            if family=='inverse_Fisher_reference' and {'marker','markevery'}&set(kwargs):
                errors.append(dict(cell=index,line=call.lineno,error='mixed empirical/reference curve retains sampling markers'))
            if family=='controlled_KL_rotation':
                styles=[node.value for node in ast.walk(reference_loop.iter)
                        if isinstance(node,ast.Constant) and node.value in ('--','-.',':')]
                if styles:
                    errors.append(dict(cell=index,line=call.lineno,error='controlled KL rotation curves must both be solid'))
            continue
        issues=[]
        for key in ('marker','markersize','markevery','ms','markeredgecolor','mec',
                    'markerfacecolor','mfc','markeredgewidth','mew','fillstyle'):
            if key in kwargs and literal(kwargs[key],'not a literal') not in (None,'None','none',''):
                issues.append('trajectory retains '+key)
        if 'markevery' in kwargs:
            issues.append('trajectory retains marker-stagger configuration')
        for value in formats:
            if not _solid_format(value):
                issues.append('trajectory retains positional format '+repr(value))
        for key in ('linestyle','ls'):
            if key in kwargs and not _solid_expression(kwargs[key],assigned):
                issues.append('trajectory linestyle is not visibly solid: '+ast.unparse(kwargs[key]))
        # The plotting defaults are solid and marker-free; an explicit source
        # override of those defaults must agree with that declaration.
        for node in ast.walk(tree):
            if isinstance(node,ast.Dict):
                for key,value in zip(node.keys,node.values):
                    if literal(key)=='lines.linestyle' and literal(value) not in ('-','solid'):
                        issues.append('non-solid lines.linestyle default')
                    if literal(key)=='lines.marker' and literal(value) not in (None,'None','none',''):
                        issues.append('nonempty lines.marker default')
        record=dict(line=call.lineno,primitive=kind,style='solid',markers=False)
        trajectories.append(record)
        errors.extend(dict(cell=index,line=call.lineno,error=issue) for issue in issues)
    if any(isinstance(node,ast.Name) and re.search(r'(MARK.*OFFSET|MARKER_OFFSET|STAGGER)',node.id)
           for node in ast.walk(tree)):
        errors.append(dict(cell=index,error='obsolete marker offset/stagger control'))
    return dict(cell=index,id=cell.get('id'),source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        trajectories=trajectories,scientific_points=points,references=references,errors=errors)


@lru_cache(maxsize=4)
def _special_tracking_groups(root,example):
    path=Path(root)/f'results/{example}/tables/paper/theory/tracking.csv'
    with path.open(newline='') as stream:
        reader=csv.DictReader(stream);rows=list(reader);columns=reader.fieldnames
    keys=[key for key in ('condition','beta','nonuniform_prior','zero_signal','design') if key in columns]
    groups={}
    for row in rows:
        special=(any(row.get(key,'').strip().lower() in ('true','1') for key in ('nonuniform_prior','zero_signal'))
                 or ('design' in row and row['design']!='balanced'))
        if special:groups.setdefault(tuple(row[key] for key in keys),[]).append(row)
    return path,sha(path),keys,list(groups.items())


def _endpoint_evidence(root,notebook,cell):
    identifier=cell.get('id')
    if notebook not in ('notebooks/01_array.ipynb','notebooks/03_contact.ipynb'):
        return None
    if identifier not in ('supp-tracking-bound-3','supp-tracking-bound-4'):
        return None
    selected=0 if identifier.endswith('-3') else 1
    tree=ast.parse(source(cell));indices=[]
    for node in ast.walk(tree):
        if (isinstance(node,ast.Assign) and isinstance(node.value,ast.Subscript)
                and isinstance(node.value.value,ast.Name) and node.value.value.id=='special_groups'):
            indices.append(literal(node.value.slice))
    assert indices==[selected],'Changed single-checkpoint cohort selection'
    example='beamforming' if '01_array' in notebook else 'contact'
    path,digest,keys,groups=_special_tracking_groups(str(Path(root).resolve()),example)
    assert sha(path)==digest,'Saved endpoint evidence changed during the audit'
    values,rows=groups[selected];times=sorted({int(float(row['t'])) for row in rows})
    assert len(times)==1,'A multi-time curve cannot use the endpoint-point exception'
    return dict(path=str(path.relative_to(root)),sha256=digest,
        selection=dict(zip(keys,values)),measurements=times,raw_rows=len(rows),
        plotted_coordinates_per_method=1)


def check_source_styles(root=ROOT,notebooks=NOTEBOOKS):
    root=Path(root).resolve();reports={};errors=[]
    for name in notebooks:
        path=root/name;notebook=json.loads(path.read_text());cells=[]
        for index,cell in enumerate(notebook['cells']):
            if cell['cell_type']!='code' or not source(cell).strip():continue
            endpoint=_endpoint_evidence(root,name,cell)
            record=inspect_cell(cell,index,endpoint_point=endpoint is not None)
            if endpoint:record['single_checkpoint_evidence']=endpoint
            if record['trajectories'] or record['scientific_points'] or record['references']:
                cells.append(record)
                errors.extend(dict(notebook=name,**item) for item in record['errors'])
        reports[name]=dict(sha256=sha(path),cells=cells,
            trajectories=sum(len(cell['trajectories']) for cell in cells),
            point_primitives=sum(len(cell['scientific_points']) for cell in cells))
    return dict(status='failed' if errors else 'passed',notebooks=reports,errors=errors,
        excluded_notebooks=['notebooks/00_theory.ipynb','notebooks/02_field.ipynb',
                            'notebooks/04_implementation_audit.ipynb'])


def _routes(root,notebooks=NOTEBOOKS):
    from lib.notebook_pdf_exports import declared_notebook_pdf_routes
    from scripts.check_moving_beamforming import optional_moving_routes
    from scripts.check_contact_prediction_diagnostics import optional_contact_diagnostic_routes
    routes=declared_notebook_pdf_routes(root)
    for extra in [optional_moving_routes(root),optional_contact_diagnostic_routes(root)]:
        assert not set(routes)&set(extra)
        routes.update(extra)
    for scope in ('moving','stationary'):
        meta=json.loads((root/f'results/beamforming_comparison_audit/{scope}/figure_manifest.json').read_text())
        notebook=json.loads((root/meta['notebook']).read_text())
        cells={cell.get('metadata',{}).get('figure_id'):index for index,cell in enumerate(notebook['cells'])
               if cell.get('metadata',{}).get('editable_plotting_role')=='plot'}
        for filename,item in meta['figures'].items():
            assert item['pdf'] not in routes
            routes[item['pdf']]=dict(notebook=meta['notebook'],cell=cells[Path(filename).stem],
                                    filename=filename,local_pdf=item['pdf'])
    return {path:record for path,record in routes.items() if record['notebook'] in notebooks}


def _dashed(value):
    match=re.match(r'\s*\[([^]]*)\]',value or '')
    return bool(match and match.group(1).strip())


def _figure_source(cell,filename):
    """Keep the PDF's own plot when a cell sequentially displays several."""
    text=source(cell);lines=text.splitlines(keepends=True);start=0;blocks=[]
    for node in ast.parse(text).body:
        if (isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
                and qualified(node.value.func)=='plt.show'):
            blocks.append(''.join(lines[start:node.end_lineno]));start=node.end_lineno
    if not blocks:return text
    if start<len(lines):blocks[-1]+=''.join(lines[start:])
    for block in blocks:
        for node in ast.parse(block).body:
            if not isinstance(node,ast.Assign):continue
            if not any(isinstance(target,ast.Name) and target.id=='PDF_PATH' for target in node.targets):continue
            if isinstance(node.value,ast.BinOp) and literal(node.value.right)==filename:
                return block
    raise AssertionError('No individual figure block for '+filename)


def _small_filled_marker(drawing):
    rect=drawing['rect'];items=drawing['items']
    if drawing.get('fill') is None or not (2<=rect.width<=12 and 2<=rect.height<=12):return False
    if any(item[0]=='re' for item in items):return True
    if len(items)<3:return False
    if drawing.get('closePath'):return True
    if all(item[0] in ('l','c') for item in items):
        first=items[0][1];last=items[-1][-1]
        return abs(first.x-last.x)<.002 and abs(first.y-last.y)<.002
    return False


def _reference_colours(text,contours=False):
    colours=[]
    for call in ast.walk(ast.parse(text)):
        if not isinstance(call,ast.Call):continue
        kind=qualified(call.func).rsplit('.',1)[-1]
        eligible=kind in ('contour','contourf') if contours else (
            kind in ('axhline','axvline','hlines','vlines') or (kind=='plot' and _reference(call)))
        if not eligible:continue
        for item in call.keywords:
            if item.arg not in ('color','colors'):continue
            for value in ast.walk(item.value):
                if isinstance(value,ast.Constant) and isinstance(value.value,str) and re.fullmatch(r'#[0-9A-Fa-f]{6}',value.value):
                    colours.append(tuple(int(value.value[index:index+2],16)/255 for index in (1,3,5)))
    return colours


def _curved_reference_colours(text):
    colours=[];tree=ast.parse(text)
    parents={child:node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    assigned={target.id:literal(node.value) for node in tree.body if isinstance(node,ast.Assign)
              for target in node.targets if isinstance(target,ast.Name)}
    def resolve(node):
        if isinstance(node,ast.Name):return assigned.get(node.id)
        if isinstance(node,ast.Subscript):
            container=resolve(node.value);key=literal(node.slice)
            try:return container[key]
            except (KeyError,TypeError,IndexError):return None
        if isinstance(node,ast.Call) and qualified(node.func)=='color' and node.args:
            styles=assigned.get('STYLE',{});method=literal(node.args[0])
            value=styles.get(method) if isinstance(styles,dict) else None
            if isinstance(value,(tuple,list)) and len(value)>1:return value[1]
            colours=assigned.get('COLORS',{})
            return colours.get(method) if isinstance(colours,dict) else None
        return literal(node)
    for call in ast.walk(tree):
        if not isinstance(call,ast.Call) or qualified(call.func).rsplit('.',1)[-1] not in ('plot','step'):continue
        family,loop=_mathematical_context(call,parents,text)
        kind=qualified(call.func).rsplit('.',1)[-1]
        label=next((literal(item.value) for item in call.keywords if item.arg=='label'),None)
        direct_reference=((kind=='step' and _reference(call)) or label in ('Theorem bound','Distance ceiling'))
        if not family and not direct_reference:continue
        candidates=list(ast.walk(loop.iter)) if family else [item.value for item in call.keywords if item.arg=='color']
        for value in candidates:
            colour=resolve(value)
            if isinstance(colour,str) and re.fullmatch(r'#[0-9A-Fa-f]{6}',colour):
                colours.append(tuple(int(colour[index:index+2],16)/255 for index in (1,3,5)))
    return colours


def audit_pdf_styles(root=ROOT,notebooks=NOTEBOOKS):
    try:import pymupdf as fitz
    except ImportError:import fitz
    root=Path(root).resolve();sources=check_source_styles(root,notebooks);errors=list(sources['errors']);files={}
    for name,route in _routes(root,notebooks).items():
        notebook=json.loads((root/route['notebook']).read_text());cell=notebook['cells'][route['cell']]
        text=_figure_source(cell,route['filename'])
        endpoint=_endpoint_evidence(root,route['notebook'],cell)
        record=inspect_cell(dict(source=text,id=cell.get('id')),route['cell'],endpoint_point=endpoint is not None)
        path=root/name;before=sha(path)
        assert before==sha(root/route['local_pdf']),'Local and curated PDF differ: '+name
        references=record['references'];point_exceptions=record['scientific_points']
        reference_colours=_reference_colours(text);contour_colours=_reference_colours(text,contours=True)
        curved_reference_colours=_curved_reference_colours(text)
        def colour_matches(colour,choices):
            return colour is not None and len(colour)==3 and any(
                max(abs(float(a)-b) for a,b in zip(colour,expected))<.002 for expected in choices)
        curves=[];events=[];allowed=[];unexpected_markers=[]
        with fitz.open(path) as document:
            assert len(document)==1
            page=document[0];page_text=' '.join(page.get_text().split())
            for drawing in page.get_drawings():
                rect=drawing['rect'];width=float(drawing.get('width') or 0.)
                dashed=_dashed(drawing.get('dashes'));items=drawing['items']
                color=drawing.get('color');fill=drawing.get('fill')
                if color is not None and len(color)==3:
                    is_event=max(abs(float(a)-b) for a,b in zip(color,(123/255,50/255,148/255)))<.002
                else:is_event=False
                if is_event and abs(width-1.3)<.005 and rect.height>36 and rect.width<.02:
                    events.append(dict(dashed=dashed,width_points=width,bounds=list(rect)))
                    if not dashed:errors.append(dict(pdf=name,error='event line is not dashed'))
                # PDF dash properties are attached to complete paths. Long
                # non-straight dashed paths cannot be event/identity lines.
                if dashed and rect.width>18 and rect.height>2:
                    is_straight=len(items)==1 and items[0][0]=='l'
                    contour=colour_matches(color,contour_colours)
                    mathematical=is_straight and colour_matches(color,reference_colours)
                    constructed=colour_matches(color,curved_reference_colours)
                    if not (contour or mathematical or constructed):
                        errors.append(dict(pdf=name,error='non-reference connected PDF path is dashed',
                                           bounds=list(rect),dashes=drawing.get('dashes')))
                    else:allowed.append(dict(kind='contour' if contour else 'mathematical_reference',bounds=list(rect)))
                if drawing.get('type') in ('s','fs') and len(items)>=2 and rect.width>18 and rect.height>2:
                    curves.append(dict(solid=not dashed,width_points=width))
                # In a figure containing only trajectories, filled tiny closed
                # paths are unwanted line/legend markers. Scientific point and
                # boxplot figures are deliberately excluded from this test.
                if record['trajectories'] and not point_exceptions and _small_filled_marker(drawing):
                    unexpected_markers.append(dict(bounds=list(rect),fill=list(fill)))
            if events:
                expected='Noise variance change' if 'Noise variance change' in text else 'Direction change'
                count=page_text.count(expected)
                if count!=1:errors.append(dict(pdf=name,error='event needs one common legend entry',label=expected,count=count))
                if any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                       and node.func.attr=='annotate' for node in ast.walk(ast.parse(text))):
                    errors.append(dict(pdf=name,error='event figure has an annotation'))
            if unexpected_markers:
                errors.append(dict(pdf=name,error='trajectory-only PDF contains small filled marker paths',
                                   markers=unexpected_markers))
        assert sha(path)==before,'PDF changed during the style audit: '+name
        files[name]=dict(sha256=before,notebook=route['notebook'],cell=route['cell'],
            source_sha256=record['source_sha256'],connected_paths=len(curves),
            events=events,allowed_reference_paths=len(allowed),
            standalone_point_exception=bool(point_exceptions),unwanted_marker_paths=len(unexpected_markers))
    return dict(status='failed' if errors else 'passed',files=files,errors=errors,
        scope='AST and actual exported PDF paths only; no plotting, simulation or timing execution.',
        method_trajectories='Solid lines without markers, markevery or marker offsets.',
        exceptions='Standalone scientific/interval points, scatter, boxplot fliers, '
                   'mathematical reference lines, contour levels and event lines.')


def check(root=ROOT,source_only=False,pdf_only=False):
    root=Path(root).resolve()
    if pdf_only:return audit_pdf_styles(root)
    sources=check_source_styles(root)
    if source_only:return sources
    if importlib.util.find_spec('pymupdf') or importlib.util.find_spec('fitz'):
        pdfs=audit_pdf_styles(root)
    else:
        candidates=list(dict.fromkeys(p for p in [shutil.which('python'),shutil.which('python3'),'/usr/bin/python3']
                                      if p and Path(p).absolute()!=Path(sys.executable).absolute()))
        failures=[]
        for interpreter in candidates:
            process=subprocess.run([interpreter,str(Path(__file__).absolute()),'--root',str(root),'--pdf-only'],
                text=True,capture_output=True,check=False)
            if process.returncode in (0,1) and process.stdout.lstrip().startswith('{'):
                pdfs=json.loads(process.stdout);break
            failures.append(process.stderr[-2000:])
        else:raise RuntimeError('PyMuPDF is needed for read-only path inspection: '+str(failures))
    errors=sources['errors']+pdfs['errors']
    return dict(status='failed' if errors else 'passed',sources=sources,pdfs=pdfs,errors=errors)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--source-only',action='store_true')
    parser.add_argument('--pdf-only',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args();report=check(args.root,args.source_only,args.pdf_only)
    report.update(checker=str(Path(__file__).relative_to(ROOT)),checker_sha256=sha(__file__))
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))
    raise SystemExit(0 if report['status']=='passed' else 1)


if __name__=='__main__':main()
