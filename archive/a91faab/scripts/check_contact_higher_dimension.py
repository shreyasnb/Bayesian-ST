"""Bounded numerical feasibility check, not a seven-joint robot experiment.

Run only after the small-contact comparison has been interpreted.  Gr(6,7)
uses the genuine rank-one complement for its mean and Monte Carlo tails.
Gr(3,7) uses the existing general-rank sampler; its process projection is
explicitly unsupported.  Fixed budgets leave failed diagnostics unresolved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy

from lib.geometry import center, orth
from lib.moments import (chain_diagnostics, effective_sample_size, mean_bingham,
                         sample_bingham)
from lib.process_prediction import stochastic_predict
from lib.spectral_series import rank_one_bingham


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def batch_quantile_se(values, probability, batches=16):
    """Contiguous-batch diagnostic; not an exact tail-error certificate."""
    pieces = np.array_split(np.asarray(values), batches)
    estimates = [np.quantile(piece, probability) for piece in pieces]
    return float(np.std(estimates, ddof=1) / np.sqrt(batches))


def projected_values(result, A, k, query, complement):
    X = result.bases
    P = np.einsum("bik,bjk->bij", X, X)
    if complement:
        P = np.eye(len(A))[None, :, :] - P
    mode = np.linalg.eigh(A)[1][:, -k:]
    Pmode = mode @ mode.T
    response = np.einsum("bij,j->bi", P, query)
    squared_distance = np.maximum(0., k - np.einsum("ij,bji->b", Pmode, P))
    hpd_loss = np.maximum(0., np.trace(A @ Pmode) - np.einsum("ij,bji->b", A, P))
    return P, response, squared_distance, hpd_loss


def summarize_tails(case, chain_id, values, scalar_rows):
    P, response, distance, hpd = values
    quantities = [(f"response_{j}", response[:, j], .025, .975)
                  for j in range(response.shape[1])]
    quantities += [("squared_chordal_distance", distance, .05, .95),
                   ("HPD_loss", hpd, .05, .95)]
    for name, samples, low, high in quantities:
        scalar_rows.append(dict(case=case, chain=chain_id, quantity=name,
            mean=float(np.mean(samples)), standard_deviation=float(np.std(samples, ddof=1)),
            ess=effective_sample_size(samples), probability_low=low,
            probability_high=high, qlow=float(np.quantile(samples, low)),
            qhigh=float(np.quantile(samples, high)),
            qlow_batch_mcse=batch_quantile_se(samples, low),
            qhigh_batch_mcse=batch_quantile_se(samples, high)))


def benchmark(config, out):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config["seed"])
    # One fixed rotation prevents coordinate-diagonal tails from accidentally
    # suppressing orientation uncertainty. These are synthetic posterior
    # parameters, not observations of a physical seven-joint arm.
    rotation = orth(rng.standard_normal((7, 7)))
    query = rng.standard_normal(7)
    query *= .02 / np.linalg.norm(query)
    chain_rows, scalar_rows, mean_rows, case_rows = [], [], [], []
    for k in (6, 3):
        for span in config["spans"]:
            spectrum = (np.array([0., .2, .3, .5, .7, .9, 1.]) if k == 6
                        else np.array([-1., -.7, -.4, .2, .4, .7, 1.]) / 2)
            A = center(rotation @ np.diag(span * spectrum) @ rotation.T)
            case = f"n7_k{k}_span{span:g}"
            complement = k == 6
            target_A, sampler_k = (-A, 1) if complement else (A, k)
            series_mean = None
            if complement:
                begin = perf_counter()
                try:
                    value = rank_one_bingham(-A, tol=1e-10,
                        max_order=config["series_max_order"])
                    series_mean = np.eye(7) - value.mean
                    mean_rows.append(dict(case=case, method="Codimension-one series",
                        seconds=perf_counter()-begin, resolved=bool(value.accuracy_met),
                        order=value.order, truncation_bound=value.moment_error_bound,
                        maximum_mean_entry=np.max(abs(series_mean)), failure=""))
                except ArithmeticError as error:
                    mean_rows.append(dict(case=case, method="Codimension-one series",
                        seconds=perf_counter()-begin, resolved=False,
                        order=config["series_max_order"], truncation_bound=np.nan,
                        maximum_mean_entry=np.nan, failure=str(error)))
            else:
                # Validate the actual dispatcher: never substitute the
                # codimension-one calculation for this rank-three parameter.
                try:
                    mean_bingham(A, k, backend="series")
                except ValueError:
                    pass
                else:
                    raise AssertionError("General-rank series was silently accepted")
            try:
                stochastic_predict(np.zeros((7, 7)), k, 100.)
                process_supported = True
                process_reason = "Supported rank/codimension-one inverse path; runtime not benchmarked here"
            except (NotImplementedError, ArithmeticError) as error:
                process_supported = not isinstance(error, NotImplementedError)
                process_reason = str(error)
            # This uniform-parameter support check does not benchmark an
            # inverse fit at the case's concentrated parameter.
            results, projected = [], []
            for chain_id, initialization in enumerate(("mode", "random", "opposite")):
                chain_rng = np.random.default_rng(np.random.SeedSequence(
                    [config["seed"], k, int(span), chain_id]))
                if initialization == "mode":
                    initial = None
                elif initialization == "random":
                    initial = orth(chain_rng.standard_normal((7, sampler_k)))
                else:
                    initial = np.linalg.eigh(target_A)[1][:, :sampler_k]
                begin = perf_counter()
                result = sample_bingham(target_A, sampler_k, chain_rng,
                    draws=config["draws"], burnin=config["burnin"], initial=initial)
                seconds = perf_counter()-begin
                results.append(result)
                values = projected_values(result, A, k, query, complement)
                projected.append(values)
                summarize_tails(case, chain_id, values, scalar_rows)
                plane_mean = np.eye(7)-result.mean if complement else result.mean
                chain_rows.append(dict(case=case, k=k, span=span, chain=chain_id,
                    initialization=initialization, sampler_rank=sampler_k,
                    draws=result.draws, burnin=result.burnin, seconds=seconds,
                    acceptance=result.acceptance, step_size=result.step_size,
                    spectral_ess=result.ess, maximum_spectral_mcse=result.mcse,
                    reference_mean_error=(np.linalg.norm(plane_mean-series_mean, 2)
                                          if series_mean is not None else np.nan)))
            diagnostics = chain_diagnostics(results)
            case_scalars = pd.DataFrame([row for row in scalar_rows if row["case"] == case])
            min_scalar_ess = float(case_scalars.ess.min())
            width = case_scalars.qhigh-case_scalars.qlow
            scaled_tail_se = (case_scalars[["qlow_batch_mcse", "qhigh_batch_mcse"]].max(axis=1)
                              / np.maximum(width, 1e-14))
            max_relative_tail_se = float(scaled_tail_se.max())
            max_tail_disagreement = 0.
            for _, group in case_scalars.groupby("quantity"):
                scale = max(float((group.qhigh-group.qlow).mean()), 1e-14)
                disagreement = max(group.qlow.max()-group.qlow.min(),
                                   group.qhigh.max()-group.qhigh.min()) / scale
                max_tail_disagreement = max(max_tail_disagreement, disagreement)
            diagnostic_ok = bool(diagnostics["min_ess"] >= 50 and
                diagnostics["max_rhat"] < 1.1 and
                diagnostics["max_standardized_difference"] < 5 and
                min_scalar_ess >= 50 and max_relative_tail_se < .05 and
                max_tail_disagreement < .15)
            case_rows.append(dict(case=case, n=7, k=k, local_dimension=k*(7-k),
                span=span, series_mean_resolved=series_mean is not None,
                **diagnostics, scalar_min_ess=min_scalar_ess,
                maximum_tail_mcse_fraction_of_width=max_relative_tail_se,
                maximum_between_chain_tail_difference_fraction_of_width=max_tail_disagreement,
                diagnostics_passed=diagnostic_ok, process_rank_supported=process_supported,
                process_check_parameter="Uniform only; concentrated inverse not benchmarked",
                process_reason=process_reason,
                sampler_seconds=sum(row["seconds"] for row in chain_rows if row["case"] == case)))
            for name, rows in (("chains.csv", chain_rows), ("scalar_tails.csv", scalar_rows),
                               ("means.csv", mean_rows), ("summary.csv", case_rows)):
                pd.DataFrame(rows).to_csv(out/name, index=False)
            print(case, "diagnostics", diagnostic_ok, "seconds", case_rows[-1]["sampler_seconds"], flush=True)
    return pd.DataFrame(case_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-study-decision", required=True, type=Path,
        help="Path to the completed small-study findings; required before this optional extension")
    parser.add_argument("--draws", type=int, default=2048)
    parser.add_argument("--burnin", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=ROOT/"results/contact_tracking/feasibility")
    args = parser.parse_args()
    if not args.small_study_decision.is_file() or args.small_study_decision.stat().st_size < 40:
        parser.error("Finish and interpret the small-contact study before this benchmark")
    if args.draws < 256 or args.burnin < 256 or args.draws > 8192 or args.burnin > 4096:
        parser.error("This feasibility check uses 256..8192 draws and 256..4096 burn-in sweeps")
    config = dict(seed=6245299, draws=args.draws, burnin=args.burnin,
        spans=[8., 200., 2000., 80000.], series_max_order=4096,
        initializations=["mode", "random", "opposite"],
        purpose="Numerical feasibility only; no seven-joint robot or physical scaling experiment",
        query_norm=.02,
        input_information="Not applicable: synthetic posterior parameter stress; no generated inputs or Fisher comparison",
        budget="Fixed once; unresolved independent-chain or tail diagnostics are retained without retries",
        diagnostics="ESS>=50, split Rhat<1.1, standardized mean discrepancy<5, tail batch MCSE<5% of width, between-chain tail discrepancy<15% of width; diagnostics are not accuracy guarantees",
        tail_scope="One preassigned seven-dimensional query response, modal chordal distance, and HPD energy loss",
        process_scope="Gr(6,7) has the existing generic rank-one complement; new 3x3 contact specialization does not apply. Gr(3,7) inverse-moment fitting is unsupported.")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"config.json").write_text(json.dumps(config, indent=2))
    start = perf_counter()
    summary = benchmark(config, args.output)
    sources = ["scripts/check_contact_higher_dimension.py", "lib/moments.py",
               "lib/spectral_series.py", "lib/process_prediction.py", "lib/geometry.py"]
    meta = dict(status="completed", seconds=perf_counter()-start,
        starting_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        small_study_decision=str(args.small_study_decision),
        small_study_decision_sha256=digest(args.small_study_decision),
        sources={name:digest(ROOT/name) for name in sources},
        tables={path.name:digest(path) for path in args.output.glob("*.csv")},
        unresolved_cases=summary.loc[~summary.diagnostics_passed, "case"].tolist(),
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    (args.output/"status.json").write_text(json.dumps(meta, indent=2))
    lines = ["# Higher-dimensional numerical feasibility", "",
        "This fixed-budget check does not implement or validate a seven-joint robot. "
        "The synthetic parameter spectra are concentration stress tests, not matched input-information experiments.", "",
        "Gr(6,7) uses an exact mathematical complement to a rank-one normal law. "
        "Its deterministic series has a finite truncation budget; tail calculations still use independent Markov chains. "
        "The new three-dimensional radial/azimuthal integration cannot be reused unchanged in seven dimensions.", "",
        "Gr(3,7) has twelve local coordinates. Its mean and tails use the actual general-rank sampler. "
        "General-rank process moment inversion remains unsupported. No rank-one shortcut or process scalability claim follows.", "",
        "| Rank | Span | Sampling time (s) | Minimum scalar ESS | Tail MCSE / width | Diagnostics |",
        "| --- | ---: | ---: | ---: | ---: | --- |"]
    for row in summary.itertuples():
        lines.append(f"| {row.k} | {row.span:g} | {row.sampler_seconds:.2f} | {row.scalar_min_ess:.1f} | {row.maximum_tail_mcse_fraction_of_width:.3f} | {'Pass' if row.diagnostics_passed else 'Unresolved'} |")
    lines += ["", "These diagnostics bound neither mixing bias nor all possible tail errors. "
        "A favorable result here would still require mechanically generated motion, known-kinematics baselines, "
        "isotropic-noise assumptions, shared noisy encoders, rank/constraint/joint-limit checks, "
        "and normalized input/Fisher comparisons before a physical arm study. "
        "The current study stops at numerical feasibility and makes no higher-dimensional calibration claim.", "",
        "Known orientation constraints also reduce the unknown family: if the columns of B span ker(J_o), "
        "the joint-velocity space is B ker(n_x^T J_pos B). At a fixed configuration, the unknown surface "
        "normal n_x still has only two orientation coordinates. An unstructured Gr(3,7) benchmark "
        "therefore cannot treat all twelve Grassmann coordinates as independent unknown physical geometry "
        "when every method has access to the same known kinematics."]
    (args.output/"findings.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
