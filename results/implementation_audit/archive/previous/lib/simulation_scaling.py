"""Bounded costs for the draft's real conditional and complex marginal models.

Dimension and concentration are separate experimental factors.  All routines
retain unsuccessful numerical targets; a forward moment is not a tail check.
This module does not tune or compare statistical accuracy.  Stream timings use
the same 64 observations and one endpoint output for every eligible method.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter, process_time
import gc
import tracemalloc
import numpy as np

from .geometry import center, gaussian
from .models import increment
from .spectral_series import rank_one_bingham
from .beamforming_numerics import complex_moments, posterior_report
from .contact_posterior import contact_summary
from benchmarks.case2.baselines import GROUSE


def configuration(profile="paper"):
    if profile not in ("quick", "paper"):
        raise ValueError("Choose quick or paper")
    return dict(schema=1, profile=profile,
        seed=2026110815 if profile == "quick" else 2026110816,
        dimensions=[3, 8, 16, 32, 64, 128],
        models=["conditional_real", "marginal_complex"],
        spans=[80., 400.], observations=64, s=.2, tau2=1., beta=.98,
        repetitions=1 if profile == "quick" else 7,
        correction_batch=8 if profile == "quick" else 64,
        warmup_calls=1, moment_tolerance=1e-10, series_max_order=4096,
        complex_report_dimensions=[3, 8, 16, 32],
        real_report_dimensions=[3], report_draws=2048 if profile == "quick" else 16384,
        report_proposal_cap=200000 if profile == "quick" else 1000000,
        report_relative_mcse=.08 if profile == "quick" else .025,
        report_queries=3, grouse_step=.02, covariance_loading=.1,
        timing_order="One warmed repetition of every operation, then independently shuffled operations in each measured repetition; serial, BLAS one thread.",
        scope="Engineering costs, not accuracy rankings, dedicated worst-case latency or asymptotic complexity proofs.",
        rank_scope="Real conditional codimension one and native proper-complex marginal rank one. General-rank moments are not substituted.",
        fixed_span_scope="Span80 and400 isolate matrix dimension from concentration; T64 generated data are a separate factor.",
        report_scope="Real N3 deterministic sphere integration; complex N3/8/16/32 independent threshold/check batches with finite exact-rejection proposal caps. Other report dimensions are explicitly not evaluated.",
        report_repetition_scope="Repeated clocks reuse the same predeclared seeded report draws; clock repetitions are not independent statistical datasets.",
        stream_scope="All endpoint streams assimilate64 snapshots and output once. Known v is supplied to both real conditional methods. GROUSE and SCM are marginal-model references only.",
        memory_scope="Exact persistent array bytes and separate Python/NumPy-tracked temporary-allocation peaks. Native BLAS/LAPACK workspaces and total process RSS are not measured.")


@dataclass
class ScalingCase:
    model: str
    N: int
    k: int
    case: str
    A: np.ndarray
    U: np.ndarray
    V: np.ndarray | None
    G: np.ndarray
    direction: np.ndarray
    desired: np.ndarray
    queries: np.ndarray
    s: float
    tau2: float
    beta: float


def make_cases(config):
    """Create every predeclared target before any measured operation."""
    cases=[]
    for model_id,model in enumerate(config["models"]):
        complex_data=model == "marginal_complex"
        for N in config["dimensions"]:
            rng=np.random.default_rng(np.random.SeedSequence([config["seed"],model_id,N]))
            Q=np.linalg.qr(gaussian(rng,(N,N),complex_data))[0]
            G=np.linalg.qr(gaussian(rng,(N,N),complex_data))[0]
            h=Q[:,0]
            desired=gaussian(rng,N,complex_data);desired/=np.linalg.norm(desired)
            queries=gaussian(rng,(config["report_queries"],N),complex_data)
            queries/=np.linalg.norm(queries,axis=1)[:,None]
            count=config["observations"]+1
            if complex_data:
                V=None
                U=gaussian(rng,count,True)[:,None]*np.sqrt(config["tau2"])*h
                U+=np.sqrt(config["s"])*gaussian(rng,(count,N),True)
            else:
                V=gaussian(rng,(count,N))
                U=V-np.outer(V@h,h)+np.sqrt(config["s"])*gaussian(rng,(count,N))
            k=1 if complex_data else N-1
            arguments=dict(model=model,N=N,k=k,U=U,V=V,G=G,direction=h,
                desired=desired,queries=queries,s=config["s"],tau2=config["tau2"],beta=config["beta"])
            for span in config["spans"]:
                # A distinct, rotated spectrum prevents diagonal eigensolvers
                # and tied-mode shortcuts from understating the dense cost.
                spectrum=np.r_[0.,-np.linspace(.25,1.,N-1)*span]
                normal=(Q*spectrum)@Q.conj().T
                A=center(normal if complex_data else -normal)
                cases.append(ScalingCase(case=f"span_{span:g}",A=A,**arguments))
            A=np.zeros((N,N),dtype=complex if complex_data else float)
            for j,u in enumerate(U[:-1]):
                A=correction(A,u,config["s"],config["tau2"],
                    None if V is None else V[j],config["beta"])
            cases.append(ScalingCase(case="data_T64",A=A,**arguments))
    return cases


def correction(A,u,s,tau2,v=None,beta=1.):
    return center(beta*A+increment(u,s,tau2,v=v,complex_data=v is None))


def modal_projector(A,k):
    """Codimension-one extraction avoids constructing an N by (N-1) basis."""
    _,U=np.linalg.eigh(A)
    if k == 1:
        h=U[:,-1]
        return np.outer(h,h.conj())
    if k == len(A)-1:
        h=U[:,0]
        return np.eye(len(A))-np.outer(h,h.conj())
    raise NotImplementedError("This cost study benchmarks rank/codimension one")


def native_mean(A,k,tolerance=1e-10,max_order=4096):
    """Use the existing native rank-one series and exact complement duality."""
    if np.iscomplexobj(A) and k == 1:
        result=complex_moments(A,tol=tolerance,max_order=max_order)
        M=result.mean;bound=result.moment_error_bound;order=result.order
    elif not np.iscomplexobj(A) and k == len(A)-1:
        result=rank_one_bingham(-A,tol=tolerance,max_order=max_order)
        M=np.eye(len(A))-result.mean;bound=result.moment_error_bound;order=result.order
    else:
        raise NotImplementedError("No general-rank moment replacement is used")
    trace_error=float(abs(np.trace(M).real-k))
    if trace_error>1e-8 or not np.isfinite(M).all():
        raise ArithmeticError("Native mean failed finiteness/trace checks")
    return M,dict(mean_order=int(order),error_bound=float(bound),trace_error=trace_error)


def dense_mvdr(R,a):
    """Solve once when positive definiteness is known from the model.

    Covariance validation is outside the timer.  Repeating an eigensystem just
    to establish positivity would conflate the decision and extraction costs.
    """
    solution=np.linalg.solve(R,a)
    return solution/np.vdot(a,solution).real


def point_mvdr(h,a,s,tau2):
    """Exact rank-one Woodbury action for s I+tau2 h h^H."""
    solution=a-tau2/(s+tau2)*h*np.vdot(h,a)
    return solution/np.vdot(a,solution).real


def endpoint_stream(case,method,config):
    """One endpoint action from the same observed stream for each method."""
    N=case.N;T=config["observations"]
    if method == "GROUSE":
        estimator=GROUSE(N,1,step=config["grouse_step"],rng=config["seed"])
        estimator.basis=case.G[:,0,None].copy()
        for u in case.U[:T]:estimator.update(u)
        return point_mvdr(estimator.basis[:,0],case.desired,case.s,case.tau2)
    if method == "SCM":
        S=np.zeros((N,N),complex);mass=0.
        for u in case.U[:T]:
            S=case.beta*S+np.outer(u,u.conj());mass=case.beta*mass+1
        R=S/mass
        R+=config["covariance_loading"]*float(np.trace(R).real/N)*np.eye(N)
        return dense_mvdr(R,case.desired)
    A=np.zeros_like(case.A)
    for j,u in enumerate(case.U[:T]):
        A=correction(A,u,case.s,case.tau2,None if case.V is None else case.V[j],case.beta)
    if method == "mean":
        M,_=native_mean(A,case.k,config["moment_tolerance"],config["series_max_order"])
        if case.model == "conditional_real":return M@case.desired
        return dense_mvdr(case.s*np.eye(N)+case.tau2*M,case.desired)
    _,U=np.linalg.eigh(A)
    if case.model == "conditional_real":
        h=U[:,0]
        return case.desired-h*np.dot(h,case.desired)
    return point_mvdr(U[:,-1],case.desired,case.s,case.tau2)


def _operations(case,config):
    N=case.N
    final_u=case.U[-1];final_v=None if case.V is None else case.V[-1]
    operations={
        "correction":(lambda:correction(case.A,final_u,case.s,case.tau2,final_v,case.beta),config["correction_batch"]),
        "mode":(lambda:modal_projector(case.A,case.k),1),
        "mean":(lambda:native_mean(case.A,case.k,config["moment_tolerance"],config["series_max_order"]),1),
        "transport":(lambda:center(case.G@case.A@case.G.conj().T),1)}
    if case.model == "marginal_complex":
        R=case.s*np.eye(N)+case.tau2*np.outer(case.direction,case.direction.conj())
        operations["MVDR"]=(lambda:dense_mvdr(R,case.desired),1)
    if case.case == "data_T64":
        methods=["mode","mean"] if case.model == "conditional_real" else ["mode","mean","GROUSE","SCM"]
        for method in methods:
            operations["endpoint_"+method]=(lambda m=method:endpoint_stream(case,m,config),1)
        if N in (config["complex_report_dimensions"] if case.model == "marginal_complex" else config["real_report_dimensions"]):
            def report():
                if case.model == "conditional_real":
                    value=contact_summary(case.A,case.queries)
                    return value,value["numerics"]
                value=posterior_report(case.A,case.queries,draws=config["report_draws"],
                    seed=config["seed"]+100000+N,max_proposals=config["report_proposal_cap"],
                    relative_mcse_tolerance=config["report_relative_mcse"])
                return value,value["numerics"]
            operations["report"]=(report,1)
    return operations


def _result_diagnostics(operation,result):
    if operation == "mean":return dict(numerical_ok=True,**result[1])
    if operation == "report":
        info=result[1]
        return dict(numerical_ok=bool(info["numerical_ok"]),
            state_ok=info.get("state_ok",info.get("regions_numerical_ok")),
            tail_ok=info.get("tail_ok",info.get("response_quantiles_ok")),
            mean_order=info.get("mean_order"),error_bound=info.get("mean_error_bound"))
    value=np.asarray(result)
    return dict(numerical_ok=bool(np.isfinite(value).all()))


def benchmark(config,progress=None):
    """Run serial costs; the caller decides when the host is available."""
    cases=make_cases(config);timings=[];memory=[];capabilities=[];parameters={}
    rng=np.random.default_rng(config["seed"]+101)
    for case in cases:
        operations=_operations(case,config)
        prefix=dict(model=case.model,N=case.N,k=case.k,case=case.case,
            span=float(np.ptp(np.linalg.eigvalsh(case.A))),
            parameter_bytes=int(case.A.nbytes),scalar_bytes=int(case.A.dtype.itemsize))
        name=f"{case.model}_N{case.N}_{case.case}"
        parameters[name]=case.A
        for operation,(function,batch) in operations.items():
            # Warm-up affects caches, not the source/target or sample count.
            for _ in range(config["warmup_calls"]):
                try:function()
                except (ArithmeticError,np.linalg.LinAlgError):pass
            gc.collect();tracemalloc.start()
            failure=""
            try:
                result=function();diagnostics=_result_diagnostics(operation,result)
            except (ArithmeticError,np.linalg.LinAlgError) as error:
                failure=str(error);diagnostics=dict(numerical_ok=False)
            _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            memory.append(dict(prefix,operation=operation,python_numpy_peak_bytes=int(peak),
                persistent_grouse_basis_bytes=16*case.N if operation=="endpoint_GROUSE" else None,
                failure=failure,**diagnostics))
        for repetition in range(config["repetitions"]):
            for operation in rng.permutation(list(operations)):
                function,batch=operations[operation]
                failure="";diagnostics=dict(numerical_ok=False)
                start=perf_counter();cpu=process_time()
                completed=0
                try:
                    for _ in range(batch):
                        result=function();completed+=1
                    diagnostics=_result_diagnostics(operation,result)
                except (ArithmeticError,np.linalg.LinAlgError) as error:failure=str(error)
                elapsed=perf_counter()-start;cpu_elapsed=process_time()-cpu
                attempted=completed+bool(failure)
                timings.append(dict(prefix,operation=operation,repetition=repetition,batch=batch,
                    completed_calls=completed,attempted_calls=attempted,
                    seconds=elapsed/max(1,attempted),cpu_seconds=cpu_elapsed/max(1,attempted),
                    cpu_wall_ratio=cpu_elapsed/elapsed if elapsed else None,
                    output_cadence="one action at T64" if operation.startswith("endpoint_") else "one operation",
                    observations=config["observations"] if operation.startswith("endpoint_") else 1,
                    failure=failure,**diagnostics))
        evaluated=case.case=="data_T64" and "report" in operations
        capabilities.append(dict(prefix,mean_supported=True,report_evaluated=evaluated,
            report_reason="Predeclared continuous or iid report cohort" if evaluated else "No tail computation in this declared dimension/concentration cohort",
            general_rank_supported=False))
        if progress:progress(f"{case.model} N={case.N} {case.case} complete")
    return dict(timings=timings,memory=memory,capabilities=capabilities,parameters=parameters)
