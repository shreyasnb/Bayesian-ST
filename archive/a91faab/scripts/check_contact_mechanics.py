"""Compare dt and dt/2 on preassigned final trajectories, without retuning."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lib.contact_mechanics import generate_contact


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out=ROOT/'results/contact_pilot';meta=json.loads((out/'config.json').read_text())
    config=meta['config'];rows=[]
    for condition in config['main_conditions']+config['stress_conditions']:
        for trial in range(4):
            parameters={key:config[key] for key in ['T','dt','d','v0','sigma']}
            coarse=generate_contact(config['test_seed'],trial,condition,substeps=1,**parameters)
            fine=generate_contact(config['test_seed'],trial,condition,substeps=2,**parameters)
            assert np.array_equal(coarse['times'],fine['times'])
            assert np.array_equal(coarse['V'],fine['V'])
            coarse_noise=coarse['U']-coarse['velocities'];fine_noise=fine['U']-fine['velocities']
            assert np.max(abs(coarse_noise-fine_noise))<1e-16
            velocity_error=float(np.max(np.linalg.norm(coarse['velocities']-fine['velocities'],axis=1)))
            position_error=float(np.max(np.linalg.norm(coarse['positions']-fine['positions'],axis=1)))
            assert velocity_error<.01*config['sigma'],'Time discretization is not negligible relative to measurement noise'
            rows.append(dict(condition=condition,trial=trial,observation_dt=config['dt'],
                fine_integration_dt=config['dt']/2,observations=config['T'],
                maximum_position_difference=position_error,maximum_velocity_difference=velocity_error,
                velocity_difference_over_sigma=velocity_error/config['sigma'],
                maximum_surface_error=max(coarse['diagnostics']['maximum_surface_error'],fine['diagnostics']['maximum_surface_error']),
                matched_scalar_model=condition!='anisotropic',checked=True))
    directory=out/'numerics';directory.mkdir(parents=True,exist_ok=True)
    path=directory/'mechanics_checks.csv';pd.DataFrame(rows).to_csv(path,index=False)
    record=dict(status='passed',scope='Integration correctness on first four preassigned test trajectories per condition; no inference or tuning changed',
        maximum_position_difference=max(r['maximum_position_difference'] for r in rows),
        maximum_velocity_difference=max(r['maximum_velocity_difference'] for r in rows),
        maximum_velocity_difference_over_sigma=max(r['velocity_difference_over_sigma'] for r in rows),
        table_sha256=sha(path),config_sha256=sha(out/'config.json'),
        sources={name:sha(ROOT/name) for name in ['lib/contact_mechanics.py','scripts/check_contact_mechanics.py']})
    (directory/'mechanics_checks.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))


if __name__=='__main__':main()
