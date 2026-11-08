"""Serial, source-verified dimension and concentration cost measurements."""
from pathlib import Path
import os
for variable in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[variable]="1"
os.environ.setdefault("MPLCONFIGDIR","/tmp/bayesian-st-matplotlib")
import argparse
from datetime import datetime,timezone
import hashlib,json,platform,sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import scipy
from lib.simulation_scaling import configuration,benchmark

SOURCES=["lib/simulation_scaling.py","scripts/run_simulation_scaling.py",
    "lib/geometry.py","lib/models.py","lib/spectral_series.py",
    "lib/beamforming_numerics.py","lib/contact_posterior.py","lib/reference.py",
    "lib/moments.py","benchmarks/case2/baselines.py"]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def portable(value):
    if isinstance(value,dict):return {str(k):portable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [portable(v) for v in value]
    if isinstance(value,np.generic):return portable(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value
def write(path,value):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(portable(value),indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def hardware():
    def read(path):
        try:return Path(path).read_text().strip()
        except OSError:return None
    return dict(platform=platform.platform(),python=platform.python_version(),
        numpy=np.__version__,scipy=scipy.__version__,logical_cpu_count=os.cpu_count(),
        cpu_affinity=len(os.sched_getaffinity(0)),cpu_quota=read("/sys/fs/cgroup/cpu.max"),
        memory_limit=read("/sys/fs/cgroup/memory.max"),blas_threads=1,
        timing="Serial process; CPU and wall clocks are separate. Other host activity can still affect wall clocks.")


def summary(frame):
    rows=[]
    group=["model","N","k","case","operation"]
    for keys,g in frame.groupby(group,sort=False):
        valid=g.numerical_ok.astype(bool)
        rows.append(dict(zip(group,keys),attempts=len(g),resolved=int(valid.sum()),
            wall_median=float(g.seconds.median()),wall_p10=float(g.seconds.quantile(.1)),
            wall_p25=float(g.seconds.quantile(.25)),wall_p75=float(g.seconds.quantile(.75)),
            wall_p90=float(g.seconds.quantile(.9)),cpu_median=float(g.cpu_seconds.median()),
            cpu_wall_ratio_median=float(g.cpu_wall_ratio.median()),
            parameter_bytes=int(g.parameter_bytes.iloc[0]),span=float(g.span.iloc[0]),
            scope="Measured operation or one endpoint stream; no failed attempts dropped"))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",choices=["quick","paper"],default="quick")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args();config=configuration(args.profile)
    out=args.output or ROOT/"results/simulation_redesign"/args.profile/"scaling"
    out.mkdir(parents=True,exist_ok=True)
    sources={name:sha(ROOT/name) for name in SOURCES}
    fingerprint=hashlib.sha256(json.dumps(dict(config=config,sources=sources),sort_keys=True).encode()).hexdigest()
    if (out/"manifest.json").exists():
        previous=json.loads((out/"manifest.json").read_text())
        if previous["fingerprint"]!=fingerprint:raise RuntimeError("Preserve the previous scaling study and use a new output")
        if not all(sha(out/name)==expected for name,expected in previous["tables"].items()):
            raise RuntimeError("Saved scaling artifacts changed")
        print("Verified completed scaling evidence");return
    if (out/"preregistered.json").exists():
        old=json.loads((out/"preregistered.json").read_text())
        if old["fingerprint"]!=fingerprint:raise RuntimeError("Frozen scaling configuration/source changed")
    declaration=dict(config=config,sources=sources,fingerprint=fingerprint,hardware=hardware(),
        created_utc=datetime.now(timezone.utc).isoformat())
    write(out/"preregistered.json",declaration)
    if args.prepare_only:
        print("Prepared",args.profile,"scaling protocol; no timing run");return
    gate=None
    if args.profile=="paper":
        from lib.simulation_validation import require_gate
        gate=require_gate(ROOT)
    write(out/"status.json",dict(status="running",fingerprint=fingerprint))
    start=perf_counter();result=benchmark(config,progress=lambda value:print(value,flush=True))
    frames={name:pd.DataFrame(result[name]) for name in ("timings","memory","capabilities")}
    frames["summary"]=summary(frames["timings"])
    for name,frame in frames.items():frame.to_csv(out/f"{name}.csv",index=False)
    np.savez_compressed(out/"parameters.npz",**result["parameters"])
    if any(sha(ROOT/name)!=expected for name,expected in sources.items()):
        raise RuntimeError("Scientific sources changed during scaling run")
    checks=dict(seconds=perf_counter()-start,cases=len(result["parameters"]),
        operations=len(frames["timings"]),unresolved_operations=int((~frames["timings"].numerical_ok.astype(bool)).sum()),
        measured_repetitions=config["repetitions"],
        means_do_not_certify_tails=True,all_targets_retained=True)
    tables={p.name:sha(p) for p in sorted(out.iterdir()) if p.suffix in (".csv",".npz")}
    write(out/"manifest.json",dict(**declaration,gate=gate,status="completed",tables=tables,checks=checks))
    write(out/"config.json",dict(**declaration,gate=gate,status="completed",tables=tables,checks=checks))
    write(out/"status.json",dict(status="completed",fingerprint=fingerprint,checks=checks))
    print(json.dumps(checks,indent=2),flush=True)


if __name__=="__main__":main()
