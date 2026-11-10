"""Render the unified beamforming figures from one verified frozen profile.

No simulations, fitting, posterior sampling, or tuning occur in this command.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from lib.beamforming_visuals import (load_unified,setup,information,tracking,
    decisions,uncertainty,prediction_and_risk,parameter_inference,process_and_cost,FIGURE_CAPTIONS)
from lib.plotting import plt, DPI


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['quick','paper'],default='paper')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    data=load_unified(ROOT,args.profile)
    producer_sources=['lib/beamforming_visuals.py','scripts/plot_beamforming_unified.py','lib/plotting.py']
    sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in producer_sources}
    input_paths=[next(p for p in [data['base']/component/'manifest.json',data['base']/component/'config.json']
        if p.exists()) for component in ('information','decisions','tracking','process','inference','numerics')]
    input_manifests={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}
    out=args.output or data['base']/'figures';out.mkdir(parents=True,exist_ok=True)
    d=data['decisions'];tr=data['tracking'];p=data['process'];num=data['numerics']
    moment=num.get('matrixcomparison',num.get('matrix_comparison'))
    if moment is None:
        moment=num['transition_checks']
    process_data=dict(transition_checks=moment,reference_checks=p['shape_checks'],timings=num['timings'])
    figures=[('01_setup',setup,d['story']),
        ('02_information',information,data['information']),
        ('03_tracking',lambda x,**kw:tracking(x['bounds'],x['trajectory'],**kw),tr),
        ('04_decisions',decisions,d),('05_uncertainty',uncertainty,d),
        ('06_prediction_and_risk',prediction_and_risk,d),
        ('07_parameter_inference',parameter_inference,data['inference']),
        ('08_process_and_cost',process_and_cost,process_data)]
    for name,function,frame in figures:
        figure=function(frame,path=out/name);plt.close(figure)
        print('Rendered',name,flush=True)
    # Complete gain panels remain available for every predeclared regime;
    # the saved illustrative beampattern belongs only to the fixed ULA record.
    from lib.beamforming_visuals import moving_parameter_inference,complete_decision_comparison,state_coverage
    figure=state_coverage(tr,path=out/'05_state_coverage')
    plt.close(figure)
    figure=complete_decision_comparison(d,path=out/'04_complete_comparison')
    plt.close(figure)
    figure=moving_parameter_inference(data['inference'],path=out/'07_process_parameters')
    plt.close(figure)
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*'))
           if p.suffix in ('.pdf','.png')}
    for name,digest in {**sources,**input_manifests}.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Plot producer or input changed during rendering: {name}')
    (out/'manifest.json').write_text(json.dumps(dict(profile=args.profile,
        figure_dpi=DPI,savefig_dpi=DPI,text_usetex=True,
        display_choices=dict(n=8,s=.2,q=1.,trial=0,posterior_N=8,parameter_N=32),
        captions=FIGURE_CAPTIONS,
        sources=sources,input_manifests=input_manifests,
        files=files),indent=2)+'\n')


if __name__=='__main__':main()
