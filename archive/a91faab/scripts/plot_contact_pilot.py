"""Render the physical contact pilot after verifying its gate and raw tables."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from lib.contact_validation import require_contact_gate
from lib.plotting import plt
from lib.contact_visuals import (physical_setup,paired_losses,coverage_width,
    response_intervals,risk_calibration,benchmark_losses,runtime,mechanical_diagnostics,validation_curves)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_contact(root=ROOT,directory=None):
    root=Path(root).resolve();directory=Path(directory or root/'results/contact_pilot').resolve()
    gate=require_contact_gate(root)
    meta=json.loads((directory/'config.json').read_text())
    if meta.get('development_only'):raise RuntimeError('Development smoke results are not the contact pilot')
    for name,digest in meta['sources'].items():
        if sha(root/name)!=digest:raise RuntimeError(f'Stale numerical source: {name}')
    if meta['gate']['contact_hash']!=gate['contact_hash'] or meta['gate']['core_hash']!=gate['core_hash']:
        raise RuntimeError('Contact results were produced under a different validation certificate')
    frames={'config':meta['config'],'metadata':meta}
    for name,digest in meta['tables'].items():
        path=directory/name
        if sha(path)!=digest:raise RuntimeError(f'Changed contact table: {name}')
        if path.suffix=='.csv':frames[path.stem]=pd.read_csv(path)
    return frames


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=ROOT/'results/contact_pilot')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();directory=args.results_dir.resolve()
    frames=load_contact(ROOT,directory);gate=require_contact_gate(ROOT)
    out=(args.output_dir or directory/'figures').resolve();out.mkdir(parents=True,exist_ok=True)
    products=[]
    def render(function,name,**kwargs):
        fig=function(frames,path=out/name,**kwargs);plt.close(fig)
        products.extend([out/f'{name}.pdf',out/f'{name}.png'])
        print('Rendered',name,flush=True)
    render(physical_setup,'physical_setup')
    render(paired_losses,'paired_losses')
    render(coverage_width,'coverage_width')
    render(response_intervals,'response_intervals',condition='weak')
    render(risk_calibration,'risk_calibration')
    render(benchmark_losses,'benchmark_losses')
    render(runtime,'runtime')
    render(mechanical_diagnostics,'mechanical_diagnostics')
    render(validation_curves,'validation_curves')
    render(paired_losses,'paired_stress',include_stress=True)
    render(coverage_width,'coverage_stress',include_stress=True)
    record=dict(core_hash=gate['core_hash'],contact_hash=gate['contact_hash'],
        sources={p:sha(ROOT/p) for p in ['lib/contact_visuals.py','scripts/plot_contact_pilot.py','lib/plotting.py']},
        metadata_sha256=sha(directory/'config.json'),
        figures={p.name:sha(p) for p in products},dpi=200,text_usetex=True,
        main_figures=['physical_setup','paired_losses','coverage_width'],
        interpretation='Geometry is plant/scoring only. Independent trajectories, maintained-posterior uncertainty and numerical resolution have distinct labels.')
    (out/'manifest.json').write_text(json.dumps(record,indent=2))


if __name__=='__main__':main()
