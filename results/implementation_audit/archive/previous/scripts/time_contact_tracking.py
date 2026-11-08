"""Serial timing of the frozen contact comparison after other runs finish.

This reruns one preassigned physical trajectory without retuning.  Timing
includes the point tracker that supplies a causal nominal rotation. Full
posterior reports retain the same nine queries, three scalar components,
state regions and numerical checks as the formal study. Average costs are
not per-observation deadline or real-time guarantees.
"""
from __future__ import annotations

import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy
from lib.contact_mechanics import generate_contact
from lib.contact_tracking_uncertainty import (METHOD_LABELS, estimate_path,
    report, scope, selected_for_condition, shared_inputs)
from lib.contact_posterior import _angular_rule, _radial_rule
from lib.contact_process_prediction import kernel_parameters, kappa_from_chordal_rms
from lib.contact_gaussian import _angles


METHODS = ["raw", "temperature", "process_identity", "process_motion",
           "gaussian_identity", "gaussian_motion"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def portable(value):
    if isinstance(value, dict):
        return {str(key): portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(item) for item in value]
    if isinstance(value, np.generic):
        return portable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    path.write_text(json.dumps(portable(value), indent=2, allow_nan=False)+"\n")


def read_optional(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def hardware():
    model = None
    for line in (read_optional("/proc/cpuinfo") or "").splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    return dict(platform=platform.platform(), machine=platform.machine(),
        cpu_model=model, visible_cpu_count=os.cpu_count(),
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        cgroup_cpu_max=read_optional("/sys/fs/cgroup/cpu.max"),
        cgroup_cpu_stat=read_optional("/sys/fs/cgroup/cpu.stat"),
        load_average=os.getloadavg() if hasattr(os, "getloadavg") else None,
        numerical_threads={name:os.environ.get(name) for name in
            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")})


def clear_numerical_caches():
    # Equal startup policy for every timed invocation. Constructors and
    # quadrature-rule/kernel preparation then occur inside timed code paths.
    for function in (_angular_rule, _radial_rule, kernel_parameters,
                     kappa_from_chordal_rms, _angles):
        function.cache_clear()


def verify_evidence(folder):
    config_path, selection_path = folder/"config.json", folder/"selected.json"
    status_path = folder/"status.json"
    document = json.loads(config_path.read_text())
    status = json.loads(status_path.read_text())
    selection = json.loads(selection_path.read_text())
    if status.get("status") != "completed" or document.get("development_only"):
        raise RuntimeError("Complete the frozen formal comparison before isolated timing")
    if selection.get("source_hash") != document.get("source_hash"):
        raise RuntimeError("Selected policies differ from completed evidence")
    if not selection.get("selection_completed_before_test"):
        raise RuntimeError("Policy selection was not frozen before test")
    for name, expected in document["sources"].items():
        if sha(ROOT/name) != expected:
            raise RuntimeError(f"Numerical source differs from formal evidence: {name}")
    for name, expected in document["tables"].items():
        if sha(folder/name) != expected:
            raise RuntimeError(f"Formal evidence changed: {name}")
    if status.get("source_hash") != document.get("source_hash"):
        raise RuntimeError("Completion status and configuration differ")
    return document, selection


def timed_method(config, data, spec, policies, repetition, order):
    clear_numerical_caches()
    method = spec["method"]
    # No practical method gets a precomputed G for free. Identity process
    # methods need no extra point tracker; raw/temperature need their raw
    # sufficient-statistic recursion; motion methods need the common causal
    # point tracker selected in Stage 1.
    required = ({"raw": policies["raw"]} if spec["kind"] == "raw"
                else {"motion": policies["motion"]} if spec.get("nominal") == "motion"
                else {})
    wall_start, cpu_start = perf_counter(), process_time()
    tracker_wall_start, tracker_cpu_start = perf_counter(), process_time()
    shared = shared_inputs(config, data["U"], data["V"], required) if required else {}
    tracker_wall = perf_counter()-tracker_wall_start
    tracker_cpu = process_time()-tracker_cpu_start
    filter_wall_start, filter_cpu_start = perf_counter(), process_time()
    path = estimate_path(config, data["U"], data["V"], spec, shared)
    filter_wall = perf_counter()-filter_wall_start
    filter_cpu = process_time()-filter_cpu_start
    estimation_wall = perf_counter()-wall_start
    estimation_cpu = process_time()-cpu_start
    completed = path["diagnostics"]["completed_steps"] == config["T"]
    report_rows = []
    for t in config["report_times"]:
        begin_wall, begin_cpu = perf_counter(), process_time()
        try:
            _, metrics, _, _ = report(config, data, path, t)
            row = dict(repetition=repetition, order=order, method=method,
                t=t, numerical_ok=bool(metrics["numerical_ok"]),
                integration_numerical_ok=bool(metrics["integration_numerical_ok"]),
                local_approximation_ok=metrics["local_approximation_ok"],
                posterior_summary_wall_seconds=metrics["report_seconds"],
                mean_wall_seconds=metrics["report_mean_seconds"],
                interval_wall_seconds=metrics["report_interval_seconds"],
                region_wall_seconds=metrics["report_region_seconds"],
                failure="")
        except (ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
            row = dict(repetition=repetition, order=order, method=method,
                t=t, numerical_ok=False, integration_numerical_ok=False,
                local_approximation_ok=None,
                posterior_summary_wall_seconds=np.nan,
                mean_wall_seconds=np.nan, interval_wall_seconds=np.nan,
                region_wall_seconds=np.nan, failure=str(error))
        row["report_call_wall_seconds"] = perf_counter()-begin_wall
        row["report_call_cpu_seconds"] = process_time()-begin_cpu
        row["benchmark_scoring_overhead_wall_seconds"] = (
            row["report_call_wall_seconds"]-row["posterior_summary_wall_seconds"])
        report_rows.append(row)
    reports = pd.DataFrame(report_rows)
    numerical_report_wall = (float(reports.posterior_summary_wall_seconds.sum())
                            if reports.posterior_summary_wall_seconds.notna().all() else np.nan)
    report_call_wall = float(reports.report_call_wall_seconds.sum())
    row = dict(repetition=repetition, order=order, method=method,
        label=METHOD_LABELS[method], condition="sphere_small", trial=0,
        observations=config["T"], completed_steps=path["diagnostics"]["completed_steps"],
        completed=completed, reports_numerical_ok=bool(reports.numerical_ok.all()),
        estimation_wall_seconds=estimation_wall, estimation_cpu_seconds=estimation_cpu,
        shared_tracker_wall_seconds=tracker_wall if required else 0.,
        shared_tracker_cpu_seconds=tracker_cpu if required else 0.,
        filter_wall_seconds=filter_wall, filter_cpu_seconds=filter_cpu,
        three_report_posterior_seconds=numerical_report_wall,
        three_report_call_wall_seconds=report_call_wall,
        three_report_call_cpu_seconds=float(reports.report_call_cpu_seconds.sum()),
        max_single_report_posterior_seconds=float(reports.posterior_summary_wall_seconds.max()),
        estimation_ms_per_observation=1000*estimation_wall/config["T"],
        amortized_ms_with_three_reports=1000*(estimation_wall+numerical_report_wall)/config["T"],
        cpu_fraction_of_estimation_wall=estimation_cpu/estimation_wall,
        total_measured_wall_seconds=estimation_wall+report_call_wall,
        total_measured_cpu_seconds=estimation_cpu+float(reports.report_call_cpu_seconds.sum()),
        failure=path["diagnostics"]["failure_reason"])
    # Nested clocks explain the implementation; only the outer measured wall
    # and CPU values above define total cost. No nested timer is added twice.
    row.update({"nested_"+name: float(value) for name, value in path["clocks"].items()})
    return row, report_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path,
        default=ROOT/"results/contact_tracking/uncertainty")
    parser.add_argument("--output", type=Path,
        default=ROOT/"results/contact_tracking/timing")
    args = parser.parse_args()
    document, selection = verify_evidence(args.evidence)
    config = document["config"]
    if config["T"] != 600 or config["report_times"] != [60, 180, 600]:
        raise RuntimeError("This timing protocol is fixed to the completed 600-observation study")
    selected = selected_for_condition(selection["selected"], "sphere_small")
    temperature = next(p for p in selected if p["method"] == "temperature")
    specs = {p["method"]:p for p in selected}
    specs["raw"] = dict(temperature, method="raw", temperature=1.)
    if set(specs) != set(METHODS):
        raise RuntimeError("Frozen study does not contain the six retained timing methods")
    policies = document["policies"][scope("sphere_small")]
    # Generate once, outside every estimator/report timer. Mechanical
    # simulation and independent truth scoring are not deployed inference.
    data = generate_contact(config["test_seed"], 0, "sphere_small",
        **{name:config[name] for name in ["T", "dt", "d", "v0", "sigma"]})
    data_hash = hashlib.sha256()
    for name in ("U", "V"):
        data_hash.update(name.encode())
        data_hash.update(np.ascontiguousarray(data[name]).tobytes())
    sources = dict(document["sources"])
    sources["scripts/time_contact_tracking.py"] = sha(Path(__file__))
    protocol = dict(repetitions=3, condition="sphere_small", trial=0,
        test_seed=config["test_seed"], observations=600, methods=METHODS,
        report_times=config["report_times"], clean_scalar_queries_per_report=27,
        method_order="Fixed cyclic shifts by two positions across three repetitions",
        numerical_caches="Cleared before each timed method; rule/kernel preparation and six-observation Gaussian warmup counted",
        shared_motion="Recomputed causal point tracker inside each motion method's timed boundary",
        intervals="Full same-code clean quantiles and state regions at60/180/600; not sampled or reduced queries",
        timing_boundary="Estimator wall/CPU include required shared tracker, allocations, predictive scores, mean calculations and full path storage. Report calls include benchmark truth scoring; numerical summary wall is retained separately.",
        nested_clocks="Inverse fit is within prediction; Gaussian warmup within prediction/update; nested clocks are never added to outer totals",
        excluded="Imports, fixed physical data generation, metadata/hash verification and CSV writing",
        realtime_scope="Averages over600 observations and three fixed reports; not worst-case deadlines or a20Hz control guarantee",
        contention_scope="Run serially only after other formal compute jobs finish; CPU/wall ratios and cgroup status retained",
        source_hash=document["source_hash"], input_arrays_sha256=data_hash.hexdigest(),
        evidence_config_sha256=sha(args.evidence/"config.json"),
        selected_sha256=sha(args.evidence/"selected.json"), sources=sources)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"status.json").exists():
        raise RuntimeError("Preserve completed timing evidence; choose a fresh --output directory")
    save_json(args.output/"protocol.json", protocol)
    start_hardware = hardware()
    start = datetime.now(timezone.utc).isoformat()
    rows, reports = [], []
    total_start = perf_counter()
    for repetition in range(3):
        order = METHODS[2*repetition:]+METHODS[:2*repetition]
        for position, method in enumerate(order):
            row, report_rows = timed_method(config, data, specs[method], policies,
                repetition, position)
            rows.append(row)
            reports.extend(report_rows)
            pd.DataFrame(rows).to_csv(args.output/"runs.csv", index=False)
            pd.DataFrame(reports).to_csv(args.output/"reports.csv", index=False)
            print(repetition, method, f"{row['estimation_ms_per_observation']:.3f} ms/observation, "
                f"{row['three_report_posterior_seconds']:.3f} s for3reports", flush=True)
    frame = pd.DataFrame(rows)
    summary_rows = []
    for method in METHODS:
        group = frame[frame.method == method]
        row = dict(method=method, label=METHOD_LABELS[method], repetitions=len(group),
            all_completed=bool(group.completed.all()),
            all_reports_numerical_ok=bool(group.reports_numerical_ok.all()))
        for name in ["estimation_wall_seconds", "estimation_cpu_seconds", "shared_tracker_wall_seconds",
            "filter_wall_seconds", "three_report_posterior_seconds", "max_single_report_posterior_seconds",
            "estimation_ms_per_observation", "amortized_ms_with_three_reports",
            "cpu_fraction_of_estimation_wall", "total_measured_wall_seconds"]:
            row[name+"_median"] = float(group[name].median())
            row[name+"_min"] = float(group[name].min())
            row[name+"_max"] = float(group[name].max())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output/"summary.csv", index=False)
    if any(sha(ROOT/name) != expected for name, expected in sources.items()):
        raise RuntimeError("Sources changed during timing; retain raw data but do not treat it as frozen")
    findings = ["# Serial contact timing", "",
        "Three fresh filter runs per method use the preassigned small-sphere held-out trajectory0. "
        "Each starts from the same uniform prior and uses the frozen policies. Numerical caches "
        "are reset before each run. The causal point tracker is recomputed and charged to every "
        "method that uses its rotations. Imports and mechanical data generation are excluded.", "",
        "| Method | Estimation (ms/observation) | Three full reports (s) | Largest report (ms) | CPU / wall |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples():
        findings.append(f"| {row.label} | {row.estimation_ms_per_observation_median:.3f} | "
            f"{row.three_report_posterior_seconds_median:.3f} | "
            f"{1000*row.max_single_report_posterior_seconds_max:.1f} | "
            f"{row.cpu_fraction_of_estimation_wall_median:.3f} |")
    findings += ["", "Each full report computes nine vector-query means,27 clean scalar intervals, "
        "state regions and numerical checks. Numerical report time excludes independent truth scoring; "
        "the complete call time and that scoring overhead remain in reports.csv. Costs of prediction, "
        "inverse fitting, startup and moments are nested explanations, not extra terms to add to wall time.", "",
        "These are measured average costs for one fixed trajectory, not a worst-case latency guarantee. "
        "In particular, a report taking more than50ms cannot be justified as synchronous20Hz reporting "
        "by averaging it over600 observations. Contended in-study timing tables remain unchanged."]
    (args.output/"findings.md").write_text("\n".join(findings)+"\n")
    status = dict(status="completed", started_utc=start,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        total_wall_seconds=perf_counter()-total_start,
        starting_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        protocol_sha256=sha(args.output/"protocol.json"), hardware_start=start_hardware,
        hardware_end=hardware(),
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        all_completed=bool(frame.completed.all()),
        all_reports_numerical_ok=bool(frame.reports_numerical_ok.all()),
        minimum_estimation_cpu_wall_ratio=float(frame.cpu_fraction_of_estimation_wall.min()),
        possible_estimation_contention=bool((frame.cpu_fraction_of_estimation_wall < .9).any()),
        tables={name:sha(args.output/name) for name in
            ["runs.csv", "reports.csv", "summary.csv", "findings.md"]})
    save_json(args.output/"status.json", status)
    print(json.dumps(portable(status), indent=2))


if __name__ == "__main__":
    main()
