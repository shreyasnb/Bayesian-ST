"""Paired, trajectory-level endpoint analysis of the frozen matched experiment.

No target is removed because its local approximation or reference is unresolved.
Undefined or absent paired endpoints mark the corresponding comparison unresolved.
This script changes no filtering, tuning, integration or test-source declarations.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_analysis(directory):
    directory=Path(directory)
    declaration=json.loads((directory/'config.json').read_text())
    config=declaration['config']
    metrics=pd.read_csv(directory/'metrics.csv')
    terminal=metrics[metrics.t==config['T']]
    rows=[]
    for kappa,g in terminal.groupby('kappa'):
        for comparator in ['Same-parameter Laplace','Recursive Gaussian',
                           'Fixed forgetting diagnostic','Full-density reference']:
            count=(config['reference_trials'] if comparator=='Full-density reference'
                   else config['trials'])
            identities=np.arange(count)
            selected=g[g.method.isin(['Process Bingham',comparator])]
            flags=selected.pivot(index='trial',columns='method',values='numerical_ok').reindex(identities)
            resolved=(flags.get('Process Bingham',False)&flags.get(comparator,False)).fillna(False)
            for endpoint in ['clean_mse','interval_score','clean_coverage','clean_width','prior_log_score']:
                pivot=selected.pivot(index='trial',columns='method',values=endpoint).reindex(identities)
                delta=(pivot[comparator]-pivot['Process Bingham']).to_numpy()
                finite=np.isfinite(delta)
                valid=bool(finite.all())
                difference=float(delta.mean()) if valid else np.nan
                half=(float(student_t.ppf(.975,count-1)*np.std(delta,ddof=1)/np.sqrt(count))
                      if valid and count>1 else np.nan)
                rows.append(dict(kappa=kappa,comparator=comparator,endpoint=endpoint,
                    trajectories=count,finite_pairs=int(finite.sum()),resolved_pairs=int(resolved.sum()),
                    comparator_minus_process=difference,low=difference-half,high=difference+half,
                    endpoint_status='finite' if valid else 'unresolved missing/nonfinite endpoint',
                    inference='Exploratory two-sided 95% paired Student t interval across independent trajectories',
                    scope='First predeclared sixteen trajectories' if comparator=='Full-density reference' else 'All predeclared trajectories',
                    interpretation='Positive favors process for loss/width; positive log-score favors comparator; coverage alone is not a score'))
    frame=pd.DataFrame(rows)
    frame.to_csv(directory/'paired_effects.csv',index=False)
    metadata=dict(source_hash=digest(Path(__file__)),
        input_hashes={name:digest(directory/name) for name in ['config.json','metrics.csv','reference_checks.csv','shape_checks.csv']},
        output_hash=digest(directory/'paired_effects.csv'),
        experiment_source_hash=declaration['source_hash'],
        policy='No filtering on numerical/local-approximation success; undefined paired endpoints remain unresolved',
        endpoint_time=config['T'],confidence=.95,
        multiplicity='Exploratory intervals without multiplicity correction; no superiority claim for unresolved calculations')
    (directory/'analysis.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,
        default=Path(__file__).resolve().parents[1]/'results/contact_uncertainty/matched')
    args=parser.parse_args()
    frame=paired_analysis(args.directory)
    print(frame[frame.endpoint.isin(['clean_mse','interval_score'])].to_string(index=False))


if __name__=='__main__':
    main()
