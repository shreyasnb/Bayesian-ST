"""Independent native-complex numerical checks for the unified array study."""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,hashlib,json,platform,sys
from time import perf_counter,process_time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,scipy
from numpy.polynomial.legendre import leggauss
from lib.beamforming_protocol import profile,output_directory,ula,proper_complex
from lib.beamforming_numerics import (complex_moments,sample_complex_bingham,
    sample_complex_transition,sample_complex_transitions,transition_coefficients_complex,
    transition_log_normalizer,invert_complex_mean,posterior_report,cp1_power_cdf)
from lib.geometry import center


SOURCES=['lib/beamforming_numerics.py','lib/beamforming_protocol.py',
    'lib/spectral_series.py','lib/geometry.py','lib/process_prediction.py',
    'scripts/check_beamforming_numerics.py','tests/test_beamforming_numerics.py']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def portable(x):
    if isinstance(x,dict):return {str(k):portable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [portable(v) for v in x]
    if isinstance(x,np.generic):return portable(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def write_json(path,value):path.write_text(json.dumps(portable(value),indent=2,allow_nan=False)+'\n')


def matrix_rows(case,n,kappa,analytic,vectors):
    P=vectors[:,:,None]*vectors[:,None,:].conj();empirical=P.mean(axis=0)
    real_se=P.real.std(axis=0,ddof=1)/np.sqrt(len(P));imag_se=P.imag.std(axis=0,ddof=1)/np.sqrt(len(P))
    rows=[]
    for i in range(n):
        for j in range(n):
            rows.append(dict(case=case,n=n,kappa=kappa,row=i,column=j,
                analytic_real=float(analytic[i,j].real),analytic_imag=float(analytic[i,j].imag),
                empirical_real=float(empirical[i,j].real),empirical_imag=float(empirical[i,j].imag),
                mcse_real=float(real_se[i,j]),mcse_imag=float(imag_se[i,j])))
    error=float(np.linalg.norm(empirical-analytic,'fro'))
    se=float(np.sqrt(np.sum(real_se**2+imag_se**2)))
    return rows,error,se


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['quick','paper'],default='quick')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--development',action='store_true')
    args=parser.parse_args();config=profile(args.profile)
    config.update(component='numerics',seed=6575011 if args.profile=='quick' else 6575012,
        transition_draws=8192 if args.profile=='quick' else 65536,
        tail_draws_per_batch=2048 if args.profile=='quick' else 16384,
        timing_repetitions=1 if args.profile=='quick' else 3,
        dimensions=[2,8,16],transition_kappas=[0.,4.,20.,100.],
        mean_tolerance=1e-10,series_max_order=4096,
        profiles_scope='Native n=8, separate n=16 feasibility, independent exact CP1 reference',
        negative_control='Span20000 exceeds declared series budget; no surrogate mean is substituted',
        sampling='Exact iid rejection, independent threshold/check batches; estimated MCSE is not deterministic tail accuracy')
    sources={p:sha(ROOT/p) for p in SOURCES};key=sha(ROOT/'lib/beamforming_numerics.py')
    out=args.output or output_directory(ROOT,args.profile,'numerics');out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():
        old=json.loads((out/'config.json').read_text())
        if old['sources']==sources and old['config']==config and all(sha(out/p)==v for p,v in old['tables'].items()):
            print('Verified cached native-complex numerical evidence');return
        raise RuntimeError('Preserve existing numerical evidence; choose a new output directory')
    gate=None
    if args.profile=='paper' and not args.development:
        from lib.beamforming_validation import require_beamforming_gate
        gate=require_beamforming_gate(ROOT)
    write_json(out/'preregistered.json',dict(config=config,sources=sources,gate=gate,
        development_only=args.development,source_hash=key))
    rng=np.random.default_rng(config['seed']);means=[];timings=[];transition=[];entries=[];references=[];reports=[];risks=[]
    parameters={};W_by_n={};start_all=perf_counter()
    def timed(operation,n,function,repetition,scope):
        start=perf_counter();cpu=process_time();result=function()
        wall=perf_counter()-start;cpu=process_time()-cpu
        timings.append(dict(operation=operation,n=n,repetition=repetition,seconds=wall,
            cpu_seconds=cpu,cpu_wall_ratio=cpu/wall,scope=scope))
        return result
    for n in [2,8,16]:
        h=ula(n,np.deg2rad(20.));U=proper_complex(rng,(64,1))*h+proper_complex(rng,(64,n),.2)
        scatter=np.zeros((n,n),complex)
        for t,u in enumerate(U,1):
            scatter+=np.outer(u,u.conj())/.24
            if t not in (1,8,32,64):continue
            A=center(scatter);parameters[f'n{n}_N{t}']=A
            try:
                result=complex_moments(A,tol=1e-10,max_order=4096)
                means.append(dict(n=n,N=t,span=float(np.ptp(np.linalg.eigvalsh(A))),
                    numerical_ok=result.numerical_ok,order=result.order,
                    mean_error_bound=result.moment_error_bound,trace_error=abs(np.trace(result.mean).real-1),
                    seconds=result.seconds,failure='',negative_control=False))
            except ArithmeticError as error:
                means.append(dict(n=n,N=t,numerical_ok=False,failure=str(error),negative_control=False))
        A=parameters[f'n{n}_N32'];W=np.array([ula(n,np.deg2rad(x)) for x in [-20.,20.,45.]])
        W_by_n[n]=W
        for repetition in range(config['timing_repetitions']):
            forward=timed('Native mean',n,lambda:complex_moments(A),repetition,
                'Whole native-complex mean/log-normalizer call, no tails')
            timed('Iid 8192 draws',n,lambda:sample_complex_bingham(A,8192,rng),repetition,
                'Whole exact iid sampler including eigenvectors and phases')
            report=timed('Full posterior report',n,lambda:posterior_report(A,W,truth=h,
                draws=config['tail_draws_per_batch'],rng=rng),repetition,
                'Mean + two iid batches + state regions + three scalar queries; nested costs not added')
            reports.append(dict(n=n,N=32,repetition=repetition,**report['numerics']))
            for kappa in [4.,20.]:
                c=transition_coefficients_complex(n,kappa);target=c.rho*forward.mean+(1-c.rho)*np.eye(n)/n
                fit=timed('Inverse mean fit',n,lambda:invert_complex_mean(target,initial=A),repetition,
                    'Whole inverse fit; includes its forward means; not added to prediction twice')
                reports.append(dict(n=n,N=32,repetition=repetition,kappa=kappa,inverse_residual=fit.residual,
                    inverse_iterations=fit.iterations,inverse_evaluations=fit.evaluations,
                    inverse_numerical_ok=fit.converged,inverse_moment_uncertainty=fit.moment_uncertainty))
        # An explicit out-of-range target is a limitation, not a usable mean.
        if n>2:
            try:
                result=complex_moments(np.diag(np.linspace(0,20000,n)),max_order=4096)
                means.append(dict(n=n,N=-1,span=20000.,numerical_ok=True,order=result.order,
                    failure='Unexpectedly resolved within fixed budget',negative_control=True))
            except ArithmeticError as error:
                means.append(dict(n=n,N=-1,span=20000.,numerical_ok=False,order=4096,
                    failure=str(error),negative_control=True))
        old=sample_complex_bingham(A,config['transition_draws'],rng).vectors
        M=complex_moments(A).mean
        for kappa in config['transition_kappas']:
            coefficient=transition_coefficients_complex(n,kappa)
            for case,draws,analytic in [
                ('conditional kernel',sample_complex_transition(h,kappa,rng,size=config['transition_draws']),
                 coefficient.a*np.outer(h,h.conj())+coefficient.b*(np.eye(n)-np.outer(h,h.conj()))),
                ('predictive mixture',sample_complex_transitions(old,kappa,rng),
                 coefficient.rho*M+(1-coefficient.rho)*np.eye(n)/n)]:
                records,error,se=matrix_rows(case,n,kappa,analytic,draws);entries.extend(records)
                transition.append(dict(n=n,kappa=kappa,case=case,draws=len(draws),rho=coefficient.rho,
                    frobenius_error=error,frobenius_mcse=se,standardized_error=error/se,
                    numerical_ok=error<=5*se,log_normalizer=transition_log_normalizer(n,kappa)))
        # The spatial-projection diagnostic is derived; v was not a known
        # training input and the sample's latent coefficients are never used.
        samples=sample_complex_bingham(A,config['transition_draws'],rng).vectors
        for j,v in enumerate(W):
            b=.3*proper_complex(rng,n);projected=samples*(samples.conj()@v)[:,None]
            loss=np.sum(abs(projected-b)**2,axis=1)
            expected=float(np.vdot(v,(M-M@M)@v).real+np.linalg.norm(M@v-b)**2)
            se=float(loss.std(ddof=1)/np.sqrt(len(loss)))
            risks.append(dict(n=n,query=j,analytic=expected,empirical=float(loss.mean()),mcse=se,
                numerical_ok=abs(loss.mean()-expected)<=5*se,
                scope='Derived P*v query, not a known-input training experiment'))
        print('Numerical dimension',n,'done',flush=True)
    nodes,w=leggauss(512);x=(nodes+1)/2;w=w/2
    for gap in [0.,.01,5.,50.,300.]:
        A=np.diag([0.,gap]);expected_mass=w*np.exp(gap*(x-1));Z=expected_mass.sum()
        exact_mean=float(expected_mass@x/Z);forward=complex_moments(A)
        references.append(dict(n=2,gap=gap,mean_error=abs(forward.mean[1,1].real-exact_mean),
            log_normalizer_error=abs(forward.log_normalizer-(gap+np.log(Z))),
            numerical_ok=abs(forward.mean[1,1].real-exact_mean)<1e-10,
            source='Independent Gauss-Legendre integration of native complex uniform-simplex coordinate'))
    frames=dict(mean_feasibility=pd.DataFrame(means),timings=pd.DataFrame(timings),
        transition_checks=pd.DataFrame(transition),matrix_comparison=pd.DataFrame(entries),
        reference_checks=pd.DataFrame(references),posterior_reports=pd.DataFrame(reports),risk_checks=pd.DataFrame(risks))
    np.savez_compressed(out/'parameters.npz',**parameters)
    tables={'parameters.npz':sha(out/'parameters.npz')}
    for name,frame in frames.items():
        path=out/f'{name}.csv';frame.to_csv(path,index=False);tables[path.name]=sha(path)
    checks=dict(actual_mean_failures=int((~frames['mean_feasibility'].query('not negative_control').numerical_ok).sum()),
        declared_out_of_range_targets=int(frames['mean_feasibility'].negative_control.sum()),
        transition_failures=int((~frames['transition_checks'].numerical_ok).sum()),
        reference_failures=int((~frames['reference_checks'].numerical_ok).sum()),
        risk_failures=int((~frames['risk_checks'].numerical_ok).sum()),
        tail_report_failures=sum(not row.get('numerical_ok',True) for row in reports),
        inverse_failures=sum(not row.get('inverse_numerical_ok',True) for row in reports),
        seconds=perf_counter()-start_all,
        tail_scope='Exact iid law, finite independent Monte Carlo quantiles; no deterministic tail-accuracy claim')
    if any(sha(ROOT/p)!=h for p,h in sources.items()):raise RuntimeError('Numerical sources changed during checks')
    write_json(out/'config.json',dict(config=config,sources=sources,source_hash=key,tables=tables,
        gate=gate,development_only=args.development,checks=checks,
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)))
    write_json(out/'checks.json',checks)
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
