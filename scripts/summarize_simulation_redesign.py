"""Summarize verified draft-5 simulation evidence without rerunning science.

The paper command requires all four components, the current general gate and
the final figure manifest.  Quick summaries are explicitly engineering output.
"""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import hashlib,json,sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from lib.simulation_validation import require_gate,verify_component,science_hash,gate_path

COMPONENTS=("geometry","beamforming","contact","scaling")
LABELS={"mean":r"$\overline P_t^+$","mode":r"$P_{\widehat{\mathcal X}_t}$",
    "mode_same_A":r"$P_{\widehat{\mathcal X}_t}$, same $A_t^+$",
    "iso":r"$\overline P(\kappa P_{\widehat{\mathcal X}_t})$",
    "isotropic":r"$\overline P(\kappa P_{\widehat{\mathcal X}_t})$",
    "scalar":"Regularized PCA","regularized":"Regularized projector least squares",
    "scm":"Loaded covariance","lw":"Ledoit--Wolf","grouse":"GROUSE",
    "oracle":"Known covariance","residual_pca":"PCA of residuals","ridge":"Ridge regression",
    "laplace":r"$\mathcal N(0,H_t^{-1})$","gaussian":"Extended Kalman filter",
    "mean_identity":r"$\overline P_t^+,\ G_t=I$",
    "mode_identity":r"$P_{\widehat{\mathcal X}_t},\ G_t=I$"}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def portable(x):
    if isinstance(x,dict):return {str(k):portable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [portable(v) for v in x]
    if isinstance(x,np.generic):return portable(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def write_json(path,value):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(portable(value),indent=2,allow_nan=False)+"\n")
    temporary.replace(path)
def fmt(x,digits=4):
    try:return f"{float(x):.{digits}g}" if np.isfinite(float(x)) else "unresolved"
    except (ValueError,TypeError):return str(x)
def table(headers,rows):
    def cell(v):return str(v).replace("|",r"\|").replace("\n"," ")
    return "\n".join(["| "+" | ".join(map(cell,headers))+" |",
        "| "+" | ".join("---" for _ in headers)+" |",
        *["| "+" | ".join(map(cell,row))+" |" for row in rows]])
def label(method):return LABELS.get(str(method),str(method))
def boolean(frame,column,default=True):
    if column not in frame:return pd.Series(default,index=frame.index,dtype=bool)
    return frame[column].fillna(False).astype(bool)


def verify_mapping(mapping,folder=ROOT):
    for name,expected in mapping.items():
        if isinstance(expected,dict):expected=expected["sha256"]
        path=folder/name
        if not path.is_file() or sha(path)!=expected:raise RuntimeError("Changed evidence: "+str(path))


def load(profile,allow_incomplete=False):
    if profile=="paper" and allow_incomplete:raise ValueError("Paper evidence cannot allow incomplete components")
    data={};pending={};inputs={};sources={}
    gate=require_gate(ROOT) if profile=="paper" else None
    base=ROOT/"results/simulation_redesign"/profile
    for component in COMPONENTS:
        try:
            meta=verify_component(ROOT,profile,component)
        except (FileNotFoundError,RuntimeError) as error:
            if not allow_incomplete:raise
            pending[component]=str(error);continue
        folder=base/component
        item={"meta":meta}
        for path in sorted(folder.glob("*.csv")):item[path.stem]=pd.read_csv(path)
        data[component]=item
        metadata=folder/("manifest.json" if (folder/"manifest.json").exists() else "config.json")
        inputs[str(metadata.relative_to(ROOT))]=sha(metadata)
        sources.update(meta["sources"])
    figures=base/"figures/manifest.json"
    if figures.exists():
        fig=json.loads(figures.read_text())
        if fig.get("profile")!=profile:raise RuntimeError("Figure profile does not match summary")
        if (fig.get("text_usetex") is not True or fig.get("figure_dpi")!=200
                or fig.get("savefig_dpi")!=200):raise RuntimeError("Figure protocol is not LaTeX/200 dpi")
        if not fig.get("sources") or not fig.get("input_manifests") or not fig.get("files"):
            raise RuntimeError("Figure manifest omits producers, inputs or artifacts")
        verify_mapping(fig["sources"]);verify_mapping(fig["input_manifests"])
        verify_mapping(fig["files"],figures.parent)
        inputs[str(figures.relative_to(ROOT))]=sha(figures)
        sources.update(fig["sources"])
        figures_ok=True
    else:
        figures_ok=False
        if profile=="paper":raise RuntimeError("Render the final verified paper figures before summarizing")
    return data,pending,inputs,sources,gate,figures_ok


def classify(row):
    if not bool(row.get("defined",True)):return "not_defined"
    if not bool(row.get("numerical_ok",True)):return "unresolved"
    if int(row.get("finite_pairs",0))<int(row.get("trials",0)):return "unresolved"
    if bool(row.get("analytic_tie",False)):return "exact_tie"
    lo,hi=row.get("lower",np.nan),row.get("upper",np.nan)
    if not np.isfinite(lo) or not np.isfinite(hi):return "unresolved"
    if lo>0:return "gain"
    if hi<0:return "loss"
    if lo==0 and hi==0:return "exact_tie"
    return "crosses_zero"


def grouped_counts(paired,groups):
    output=[]
    if paired.empty:return pd.DataFrame()
    for keys,g in paired.groupby(groups,dropna=False,sort=False):
        keys=keys if isinstance(keys,tuple) else (keys,)
        values=g.apply(classify,axis=1).value_counts()
        row=dict(zip(groups,keys),cells=len(g))
        row.update({name:int(values.get(name,0)) for name in
            ["gain","crosses_zero","loss","exact_tie","unresolved","not_defined"]})
        output.append(row)
    return pd.DataFrame(output)


def count_phrase(pairs,reference,endpoint="true_power"):
    selected=pairs[pairs.reference.eq(reference)&pairs.endpoint.eq(endpoint)]
    if selected.empty:return "not evaluated"
    counts=selected.apply(classify,axis=1).value_counts()
    unresolved=int(counts.get('unresolved',0))
    qualification=f"/{unresolved} unresolved" if unresolved else ""
    return f"{int(counts.get('gain',0))} gains/{int(counts.get('loss',0))} losses{qualification} in {len(selected)} cells"


def paired_from_metrics(frame,groups,endpoint):
    """Keep finite raw comparisons and explicit local/numerical restrictions."""
    output=[]
    if endpoint not in frame:return pd.DataFrame()
    for keys,g in frame.groupby(groups,dropna=False,sort=False):
        keys=keys if isinstance(keys,tuple) else (keys,)
        own=g[g.method.eq("mean")].set_index("trial")
        for method,h in g.groupby("method",sort=False):
            if method=="mean":continue
            other=h.set_index("trial");indices=own.index.intersection(other.index)
            defined=not (method=="ridge" and endpoint.startswith("cost_"))
            delta=other.loc[indices,endpoint]-own.loc[indices,endpoint]
            finite=np.isfinite(delta)
            status=(boolean(own.loc[indices],"numerical_ok")&boolean(other.loc[indices],"numerical_ok")
                &boolean(own.loc[indices],"local_ok")&boolean(other.loc[indices],"local_ok"))
            values=delta[finite].to_numpy();count=len(values)
            gain=float(values.mean()) if count else np.nan
            error=float(student_t.ppf(.975,count-1)*values.std(ddof=1)/np.sqrt(count)) if count>1 else np.nan
            output.append(dict(zip(groups,keys),reference=method,endpoint=endpoint,
                trials=len(indices),finite_pairs=count,resolved_pairs=int((status&finite).sum()),
                gain=gain,lower=gain-error,upper=gain+error,defined=defined,
                numerical_ok=bool(defined and count==len(indices) and status.all()),
                raw_finite_gain=bool(np.isfinite(error) and gain-error>0),
                raw_finite_loss=bool(np.isfinite(error) and gain+error<0)))
    return pd.DataFrame(output)


def geometry_text(item):
    geometry=item["geometry"];directions=item["directional"];regions=item["regions"]
    checks=item["meta"]["checks"]
    rows=[]
    for r in geometry.itertuples():
        rows.append([r.label,fmt(r.radius),fmt(r.rms_radius),fmt(r.mean_0),fmt(r.mean_1),fmt(r.mean_2),r.numerical_ok])
    text=["## Directional geometry and task identities","",
        "The real normal-line parameters $A^{(1)}=\\operatorname{diag}(0,-8,-80)$ and $A^{(2)}=\\operatorname{diag}(0,-80,-8)$ have the same mode and the same entire centered chordal-distance distribution. The complementary contact plane uses parameter $-A$ and mean $I-\\mathbb E[nn^T]$. The sphere displays the real projective density, antipodally symmetric; uniform density is one with respect to normalized invariant measure.","",
        table(["Belief","$\\rho_{0.05}$","$r_{\\mathrm{rms}}$","Mean eigenvalue 1","Mean eigenvalue 2","Mean eigenvalue 3","Numerically resolved"],rows),"",
        f"The swapped-pair distance-CDF error is {fmt(checks['primary_same_distance_cdf_error'])}, while the mean-response risks and expected soft commands change by direction. Native-series and independent continuous-integration mean discrepancy is at most {fmt(checks['maximum_native_mean_error'])}; maximum response-risk identity error is {fmt(checks['maximum_response_risk_identity_error'])}.","",
        "Equal posterior mass does not make an HPD set uniformly tighter than a ball. The table below bounds the normal/blocked response norm $\\|nn^Tv\\|$; it is not a bound on admissible-plane speed.","",
        table(["Belief","Query","Ball upper bound","HPD upper bound"],
            [[r.label,r.query,fmt(r.ball_bound),fmt(r.hpd_bound)] for r in regions[regions.case.isin([0,1])].itertuples()]),"",
        "The soft action is the draft's $v_t^B=[I+\\lambda(I-\\overline P_t^+)]^{-1}v_t^{\\mathrm{des}}$. Expected cost is checked under the same supplied belief, including independent quadrature. This is a conditional deterministic calculation; selected zero-response/two-snapshot constructions and quadrature nodes are not independent random tracking trials or demonstrations of a closed-loop controller.",""]
    return text


def beamforming_text(item):
    meta=item["meta"];cfg=meta["config"];paired=item["paired"];counts=grouped_counts(paired,["n","endpoint","reference"])
    rows=[[int(r.n),r.endpoint,label(r.reference),r.gain,r.crosses_zero,r.loss,r.exact_tie,r.unresolved,r.not_defined]
        for r in counts.itertuples()]
    cohorts=cfg["cohorts"]
    text=["## Beamforming: independent training episodes","",
        "This is the native proper-complex marginal extension: $u=hz+e$, $z\\sim\\mathcal{CN}(0,\\tau^2)$ and $e\\sim\\mathcal{CN}(0,sI)$. Its likelihood correction has no real-Gaussian factor one half. Estimators receive passive snapshots only; desired steering enters the distortionless decision. The uniform ambient-line prior is not a prior restricted to an arrival-angle curve.","",
        "The primary and higher-dimensional cohorts are "+"; ".join(f"$N={x['n']}$: {x['validation_trials']} validation/{x['test_trials']} test episodes per law/noise" for x in cohorts)+". All four horizons and both noise powers are retained for Haar signatures and ULA interferer separations $3^\\circ,10^\\circ,30^\\circ$ from desired $0^\\circ$.","",
        "Loaded covariance selects its loading continuously over the full convex-shrinkage interval, including the exact isotropic endpoint. Ledoit--Wolf adapts shrinkage to each observed covariance and has an explicit one-snapshot convention. GROUSE and the globally regularized modal covariance use independent validation. Their cost or information differences are not relabeled a Bingham-shape advantage.","",
        "Positive gain means comparator minus the posterior-mean loss; intervals are paired pointwise 95% Student intervals across independent episodes. The sign counts are exploratory, with no multiplicity correction. Crossing zero is inconclusive; only the structural one-snapshot control is an exact equality. Covariance-only comparators have no invented projector-response loss.","",
        table(["$N$","Endpoint","Comparator","Gain","Crosses zero","Loss","Exact tie","Unresolved","Undefined"],rows),""]
    laws=[]
    p=paired[paired.endpoint.eq("true_power")]
    for (N,law),g in p.groupby(["n","law"],sort=False):
        laws.append([int(N),law,*[count_phrase(g,r) for r in ["mode","iso","scalar","scm","lw","grouse"]]])
    text += ["The generation law matters; every predeclared separation remains visible:","",
        table(["$N$","Law",label("mode"),label("iso"),label("scalar"),label("scm"),label("lw"),"GROUSE"],laws),""]
    fixture=item["fixture"];checks=meta["checks"];tails=item["tails"]
    text += ["### Designed posterior and uncertainty","",
        "The selected-data complex fixture uses a unitary transformation fixing the mode and swapping complementary axes. The two beliefs therefore share their spectrum, mode and every centered chordal-distance quantile, while their task-optimal weights can differ. This establishes that radius and mode do not determine the decision; it is not a claim that typical random training yields the fixture's advantage.","",
        table(["Belief","Decision","Posterior expected power","Excess over its Bayes action"],
            [[r.distribution,label(r.method),fmt(r.expected_power),fmt(r.excess)] for r in fixture.itertuples()]),"",
        r"One-snapshot maximum $\|\overline P-\overline P_{\mathrm{iso}}\|_F$ is "+
        f"{fmt(checks['one_snapshot_maximum_anisotropy'])}; independent modal/PCA discrepancy is {fmt(checks['maximum_independent_pca_error'])}. The mode alone duplicates covariance PCA under the same prior, powers and weights.","",
        f"Native means: {checks['mean_failures']} failures. Evaluated posterior reports: {checks['tail_reports']}, with {checks['unresolved_tail_reports']} unresolved scalar reports and {checks['unresolved_state_reports']} unresolved state reports. These are bounded subsets with {cfg['posterior_draws']} draws in each independent training/checking batch. Dense scan angles are display coordinates, not independent repetitions; the declared quantile-precision checks concern the fixed query directions.","",
        "Central scalar intervals do not have uniform fixed-state frequentist coverage. At an exactly aligned query, the true covariance power is the upper support endpoint, excluded by every finite continuous equal-tail interval. State-region mass and repeated-data state inclusion answer different questions. The sphere used for complex $N>2$ is an explicitly labeled fixed-phase three-coordinate density section, not a marginal or full eight-/sixteen-dimensional posterior.",""]
    return text


def contact_tables(item):
    static=item["static_metrics"];dynamic=item["dynamic_metrics"]
    static_pairs=[];dynamic_pairs=[]
    endpoints=["response_mse","cost_1","cost_10"]
    for endpoint in endpoints:
        static_pairs.append(paired_from_metrics(static,["kind","t"],endpoint))
        dynamic_pairs.append(paired_from_metrics(dynamic,["condition","t"],endpoint))
    return pd.concat(static_pairs,ignore_index=True),pd.concat(dynamic_pairs,ignore_index=True)


def contact_text(item):
    cfg=item["meta"]["config"];checks=item["meta"]["checks"]
    static=item["static_metrics"];dynamic=item["dynamic_metrics"]
    sp,dp=contact_tables(item)
    text=["## Contact-normal tracking","",
        "The plant is an ideal maintained bilateral contact with known isotropic resistance. Applied free velocity $v_t$, noisy realized velocity $u_t$ and measured wrist orientation are available to corresponding methods. Surface position/normal and true normal rotations are scoring information. The real conditional likelihood is $u_t=P_{\\mathcal X_t}v_t+e_t$, with normal-line parameter $-A_t^+$ for the tangent-plane parameter $A_t^+$.","",
        f"The static study uses {cfg['static_validation_trials']} validation and {cfg['static_test_trials']} independent test datasets per equal-energy input program. Physical plane/sphere trajectories use {cfg['dynamic_validation_trials']} validation and {cfg['dynamic_test_trials']} test trajectories per condition. Gaussian errors, query coordinates and reporting times within a trajectory are not extra independent replicates.","",
        "Both the Bingham mode and independent conditional projector least squares use the observed requests; they agree under the same statistic. Residual PCA uses $v-u$, and ridge regression uses both signals and shrinks toward the same known-rank isotropic map. The local Gaussian is computed from the same Bingham curvature; the recursive extended Kalman filter additionally changes propagation and has its own validation-selected covariance. A finite local result can remain outside its valid chart.","",
        "The next tables count only complete numerical/local-validity comparisons as resolved. Raw finite endpoint differences are retained separately, so a local-approximation failure is not silently deleted or converted into a validated Bingham win. Ridge response maps are scored for prediction, and their unsupported soft decision cost is explicitly undefined.",""]
    for title,pairs in [("Static input programs",sp),("Physical tracking",dp)]:
        count=grouped_counts(pairs,["endpoint","reference"])
        rows=[[r.endpoint,label(r.reference),r.gain,r.crosses_zero,r.loss,r.exact_tie,r.unresolved,r.not_defined]
            for r in count.itertuples()]
        text += ["### "+title,"",table(["Endpoint","Comparator","Gain","Crosses zero","Loss","Exact tie","Unresolved","Undefined"],rows),""]
    singleton=sp[sp.t.eq(1)&sp.endpoint.eq("response_mse")&sp.reference.eq("isotropic")]
    text += ["A real conditional singleton adds a generally indefinite matrix of rank at most two. It can already yield directional uncertainty, unlike the exactly axial one-snapshot passive marginal posterior. The following singleton comparison was fixed by the horizon grid; both equal-energy input programs have the same first command and paired noise, so these two cells are not independent replications:","",
        table(["Input program","Mean-response gain over matched overlap $(\\mathrm{mm}/\\mathrm{s})^2$","Pointwise 95% interval","Resolved pairs"],
            [[r.kind,fmt(r.gain*1e6),f"[{fmt(r.lower*1e6)}, {fmt(r.upper*1e6)}]",f"{int(r.resolved_pairs)}/{int(r.trials)}"] for r in singleton.itertuples()]),""]
    # Display whole-cohort results, including local-invalid rows, with flags.
    rows=[];T=max(cfg["dynamic_times"])
    for (condition,method),g in dynamic[dynamic.t.eq(T)].groupby(["condition","method"],sort=False):
        rows.append([condition,label(method),len(g),int(np.isfinite(g.response_mse).sum()),fmt(g.response_mse.mean()*1e6),
            fmt(g.time_average_mse.mean()*1e6) if "time_average_mse" in g else "not evaluated",
            int((~boolean(g,"numerical_ok")).sum()),int((~boolean(g,"local_ok")).sum())])
    text += ["### Terminal and time-average response errors","",
        "All finite values remain in these descriptive means, including flagged local results. Response MSE units are $(\\mathrm{mm}/\\mathrm{s})^2$. Methods select their own settings on independent terminal response loss; time-average loss is a separate retained endpoint, not a universally optimized objective.","",
        table(["Condition","Prediction","Trajectories","Finite terminal values","Terminal MSE","Time-average MSE","Numerically unresolved","Local-chart unresolved"],rows),""]
    fisher=item["static_fisher"];frows=[]
    for (kind,t),g in fisher.groupby(["kind","t"],sort=False):
        frows.append([kind,int(t),fmt(g.input_energy.mean()),fmt(g.fisher_min.mean()),fmt(g.fisher_max.mean()),fmt(g.input_rank.mean())])
    text += ["### Excitation, not merely command magnitude","",
        "Every command has the same norm across balanced and directionally concentrated programs. The local information depends on the actual subspace; a small input-Gram eigenvalue alone does not establish weak normal information.","",
        table(["Input program","$t$","Total input energy","Mean $\\lambda_{\\min}(F_t)$","Mean $\\lambda_{\\max}(F_t)$","Input rank"],frows),""]
    text += ["### Fixed-time conditional modal certificate","",
        "The stationary certificate uses the residual dual $v-u=(nn^T)v-e$, which has the same sensor variance and the complementary modal normal. Known noise, deterministic inputs/transports and pointwise reporting times are required. Truth-dependent curvature/drift is an evaluation term. Invalid sufficient conditions remain invalid; they are not clipped to a radius or reported as successful confidence bounds.",""]
    for name in ["bounds","dynamic_bounds"]:
        b=item[name];groups=[x for x in ["condition","t"] if x in b]
        if not groups:groups=["t"]
        brows=[]
        for key,g in b.groupby(groups,dropna=False,sort=False):
            key=key if isinstance(key,tuple) else (key,)
            valid=boolean(g,"valid");bad=boolean(g,"violated",False)
            brows.append([*key,len(g),int(valid.sum()),int((~valid).sum()),int(bad.sum()),
                fmt(g.loc[valid,"bound"].mean()) if valid.any() else "invalid"])
        text += [name+":","",table([*groups,"Reports","Valid","Invalid","Violations","Mean valid bound"],brows),""]
    higher=item["higher_dimension"];rows=[]
    for (N,method),g in higher.groupby(["n","method"],sort=False):
        rows.append([int(N),int(g.t.iloc[0]),label(method),len(g),fmt(g.response_mse.mean()*1e6),
            fmt(g.normalized_response_mse.mean()),fmt(g.normal_chordal_error.mean()),
            int((~boolean(g,"numerical_ok")).sum())])
    text += ["### Higher-dimensional constraints","",
        "These are generalized single linear maintained constraints with isotropic resistance, not a claimed seven-joint robot with known Jacobians. Rank is $k=N-1$ and the native normal-line complement is used. Both dimension-normalized response-map error and normal chordal error are shown: a one-constraint task occupies a decreasing fraction of a larger space, so raw average response error alone would be misleading. The point-normal error for the mean and matched-overlap rows is that of their common modal plane, not a claim that a full-rank mean is itself a subspace estimate. More data are supplied as $t=4N$; this is not a fixed-sample dimension comparison.","",
        table(["$N$","$t$","Prediction","Datasets","MSE $(\\mathrm{mm}/\\mathrm{s})^2$","Normalized map error","Normal chordal error","Unresolved"],rows),"",
        "### Numerical and modeling scope","",
        "```json",json.dumps(checks,indent=2),"```","",
        "Posterior means minimize squared response loss under the maintained distribution; they need not dominate every fixed physical path. The draft's expected command is a soft quadratic decision, not a guaranteed admissible velocity or a closed-loop controller. Wrist transport $G_t=R_t^TR_{t-1}$ removes a measured coordinate change, not the surface's physical curvature. Forgetting under a changing normal does not inherit stationary credible-set calibration. Accurate integration does not establish model correctness or local-Gaussian validity.",""]
    return text,sp,dp


def scaling_text(item):
    cfg=item["meta"]["config"];checks=item["meta"]["checks"];frame=item["summary"]
    rows=[]
    for r in frame[(frame.case=="data_T64")&(frame.operation.str.startswith("endpoint_")|frame.operation.eq("report"))].itertuples():
        rows.append([r.model,int(r.N),r.operation,fmt(r.wall_median*1e3),fmt(r.cpu_median*1e3),
            f"{int(r.resolved)}/{int(r.attempts)}",fmt(r.cpu_wall_ratio_median)])
    text=["## Scaling, memory and output cadence","",
        f"All dimensions $N={','.join(map(str,cfg['dimensions']))}$ retain the declared fixed spans 80/400 and separate 64-observation data targets. The real model uses $k=N-1$ through exact normal-line duality; the proper-complex model uses $k=1$. General-rank moments are not silently substituted. The paper cost profile uses {cfg['repetitions']} warmed, randomly ordered serial repetitions with BLAS limited to one thread. Repeated timing draws are not statistical Monte Carlo datasets.","",
        "The dense natural-parameter array uses exactly $8N^2$ bytes for real float64 and $16N^2$ for complex128. One conditional correction has rank at most two and a marginal correction rank one, while the stored dense update requires $O(N^2)$ operations. Dense optional transport, eigenspace extraction, positive-series moments and reports are separate work. A whole native mean already includes its eigensystem; a report includes its own mean and integration/sampling. Their nested clocks must not be added twice.","",
        "Endpoint stream comparisons assimilate the same 64 observations and output once. Rank-one modal/GROUSE beamforming uses the exact Woodbury action; it is not charged an unnecessary dense covariance inverse. The mean uses a dense solve. Conditional point and mean actions both receive the known requests. Fixed engineering settings measure implementation costs, not statistical accuracy rankings.","",
        table(["Model","$N$","Operation","Median wall ms","Median CPU ms","Resolved/attempted","CPU/wall"],rows),"",
        "The generic real mean curve uses a common positive-series backend across dimensions; it is not the specialized three-dimensional contact controller's latency. The real $N=3$ report uses continuous sphere integration, and complex reports use two independent iid batches. Other report dimensions are explicitly not evaluated; a resolved mean is not a tail certificate. Temporary memory peaks are measured separately with Python/NumPy allocation tracking, excluding native LAPACK workspaces and total process RSS. Timing variability and host scheduling are not worst-case deadlines or proofs of empirical complexity exponents.","",
        f"Retained timing attempts: {checks['operations']}; unresolved: {checks['unresolved_operations']}. See the full timing, allocation, capability and configuration tables for all 36 targets and all repeats.",""]
    return text


def beam_summary(item,profile):
    p=item["paired"];primary=p[p.n.eq(8)]
    text=(f"The $N=8$ {'paper' if profile=='paper' else 'engineering quick'} study keeps Haar signatures and all three ULA separations, both noise powers and all four horizons. "
        f"For true-covariance output power, the posterior mean records: mode {count_phrase(primary,'mode')}; "
        f"matched modal-overlap control {count_phrase(primary,'iso')}; regularized PCA {count_phrase(primary,'scalar')}; loaded covariance {count_phrase(primary,'scm')}; "
        f"Ledoit--Wolf {count_phrase(primary,'lw')}; GROUSE {count_phrase(primary,'grouse')}. "
        "These are paired pointwise 95% intervals, without multiplicity correction. All ties and losses remain. One passive snapshot has exact mean/control equality, so its gain over the mode is regularization, not directional anisotropy.\n\n"
        "The designed two-belief example has the same mode and entire distance distribution but different expected-power actions. This proves that mode and radius can omit task-relevant structure; it does not establish a typical-training advantage. Native $N=16$ repeats and serial scaling through $N=128$ are separate, declared cohorts. Sphere images of the complex posterior are fixed-phase three-coordinate density sections, not full or marginal higher-dimensional densities. The practical value of the distribution must survive adaptive covariance baselines and justify its moment/report cost. Scalar credible bands, state-region mass and fixed-state frequentist coverage remain distinct.")
    return text


def contact_summary_text(item,profile):
    cfg=item["meta"]["config"];sp,dp=contact_tables(item)
    stat=sp[sp.endpoint.eq("response_mse")];dynamic=dp[dp.endpoint.eq("response_mse")]
    static_counts="; ".join(f"{name}: {count_phrase(stat,ref,'response_mse')}" for ref,name in
        [("mode","mode"),("isotropic","matched modal-overlap control"),("ridge","ridge regression")])
    dynamic_counts=count_phrase(dynamic,"gaussian","response_mse")
    text=(f"The contact {'paper' if profile=='paper' else 'engineering quick'} study observes applied requests and noisy realized velocities, using the real conditional likelihood. "
        "Normal densities are antipodally symmetric Bingham distributions on the sphere, with parameter $-A_t^+$ for a tangent-plane parameter $A_t^+$. Equal-energy input programs change the directional information without changing command magnitude. "
        "The conditional correction can retain anisotropy from one observation. "
        f"Static response comparisons give {static_counts}. Against the independently validated extended Kalman filter, physical tracking gives {dynamic_counts}. "
        "Only complete numerical/local-validity comparisons are resolved; raw finite differences and every invalid chart are retained. These cellwise intervals have no multiplicity guarantee.\n\n"
        "The swapped-axis beliefs have identical mode and every centered distance quantile, yet different response risks and soft expected commands. This isolates directional structure. A modal point estimator with the identical likelihood remains exactly the same modal estimate; gains from known inputs or frame correction are not full-distribution gains. The $N=8,16,32$ cases are generalized single constraints, not fabricated high-DOF robot contacts. Conditional bounds retain valid and invalid conditions at fixed times. The expected command is a soft penalty, not a guaranteed feasible contact controller. Global density integration, repeated-data calibration, local Gaussian validity and computation are reported separately.")
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",choices=["quick","paper"],default="paper")
    parser.add_argument("--allow-incomplete",action="store_true",help="Engineering quick preview only")
    args=parser.parse_args()
    data,pending,inputs,sources,gate,figures_ok=load(args.profile,args.allow_incomplete)
    base=ROOT/"results/simulation_redesign";base.mkdir(parents=True,exist_ok=True)
    title="Draft-5 simulation redesign" if args.profile=="paper" else "Engineering quick simulation redesign — not paper evidence"
    text=["# "+title,"",
        "This report is generated from verified saved scientific artifacts; it performs no simulation, posterior sampling, fitting or tuning. The supplied draft's simulations section did not define the experimental protocol.","",
        "## Execution and evidence status","",
        table(["Item","Status","Scope"],
            [[name,"Completed, source/artifact verified" if name in data else "Pending",pending.get(name,"Numerical and modeling qualifications remain separate")] for name in COMPONENTS]+[
                ["General gate","Passed" if gate else "Quick engineering only","Core/source fingerprints are checked before formal paper execution"],
                ["Figures","Verified" if figures_ok else "Pending","LaTeX, 200 dpi, producers, inputs and exports"],
                ["Notebooks","Separate execution handoff","Execution must be verified after this summary; no notebook is inspected to derive simulation claims"]]),""]
    if "geometry" in data:text+=geometry_text(data["geometry"])
    if "beamforming" in data:text+=beamforming_text(data["beamforming"])
    contact_pairs={}
    if "contact" in data:
        sections,sp,dp=contact_text(data["contact"]);text+=sections
        contact_pairs=dict(static=sp,dynamic=dp)
    if "scaling" in data:text+=scaling_text(data["scaling"])
    text += ["## Interpretation","",
        "The draft's exact corrections and task identities are conditional mathematical statements. Random-training gains, geometry-specific losses, information imbalance, numerical limitations and measured compute cost are separate empirical findings. Broad or anisotropic distributions can matter to a task even when their modes agree, but a designed example does not predetermine a benchmark win. Strong point, shrinkage and local controls remain necessary. Historical beamforming/contact studies and their source certificates remain preserved under their original protocols.",""]
    paths={"findings.md":"\n".join(text)}
    paths["beamforming_summary.md"]=beam_summary(data["beamforming"],args.profile) if "beamforming" in data else "Beamforming evidence is pending; this is not a completed paper summary.\n"
    paths["contact_summary.md"]=contact_summary_text(data["contact"],args.profile) if "contact" in data else "Contact evidence is pending; this is not a completed paper summary.\n"
    for name,value in paths.items():(base/name).write_text(value.rstrip()+"\n")
    outputs={str((base/name).relative_to(ROOT)):sha(base/name) for name in paths}
    generated={}
    for name,frame in contact_pairs.items():
        path=base/("contact_paired_"+name+".csv");frame.to_csv(path,index=False)
        outputs[str(path.relative_to(ROOT))]=sha(path);generated[name]=len(frame)
    validation_path=gate_path(ROOT) if gate else None
    metadata=dict(schema=1,profile=args.profile,status="completed" if not pending else "engineering incomplete",
        paper_evidence=bool(args.profile=="paper" and not pending and figures_ok),
        created_utc=datetime.now(timezone.utc).isoformat(),
        generator=dict(path="scripts/summarize_simulation_redesign.py",sha256=sha(__file__)),
        source_hashes=sources,verified_input_hashes=inputs,verified_components=list(data),pending=pending,
        figures_verified=figures_ok,gate_sha256=sha(validation_path) if gate else None,
        gate_path=str(validation_path.relative_to(ROOT)) if gate else None,
        science_hash=gate["science_hash"] if gate else science_hash(ROOT),outputs=outputs,
        application_summaries={name:dict(words=len(paths[name].split()),paragraphs=len(paths[name].strip().split("\n\n")))
            for name in ["beamforming_summary.md","contact_summary.md"]},
        generated_contact_pair_rows=generated,
        statistics="Paired independent-trajectory pointwise Student95% intervals; no multiplicity correction. Finite numerical/local-invalid comparisons are retained but not labeled resolved gains; undefined functionals are separate.")
    write_json(base/"summary_manifest.json",metadata)
    print(json.dumps(dict(profile=args.profile,paper_evidence=metadata["paper_evidence"],
        components=list(data),pending=list(pending),application_summaries=metadata["application_summaries"]),indent=2))


if __name__=="__main__":main()
