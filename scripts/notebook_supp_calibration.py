"""Visible supplementary calibration and predictive-score notebook cell sources.

These are source templates, not runtime renderers. Each generated notebook cell
contains the actual Matplotlib artists and reads the frozen numerical records.
"""
from __future__ import annotations

from textwrap import dedent

from scripts.notebook_supplementary_common import plot_cell


_COVERAGE = r'''
frame = _normalize(evidence.frame(
    "prediction", "prediction_calibration_summary", "calibration_summary",
    "prediction_calibration", "calibration"))
if "split" in frame and frame.split.eq("test").any():
    frame = frame[frame.split.eq("test")]
if "stage" in frame and frame.stage.eq("test").any():
    frame = frame[frame.stage.eq("test")]
target_column = _column(frame, "target", "kind")
regime_column = _column(frame, "regime", "condition")
group = frame[frame[regime_column].eq(REGIME) & frame.t.eq(REPORT_TIME)]
assert not group.empty
fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=DPI, layout="constrained")
for ax, kind, tag in zip(
        axes, ["state_hpd", "clean_response", "noisy_observation"],
        ["(a)", "(b)", "(c)"]):
    subset = group[group[target_column].eq(kind)]
    if subset.empty and kind == "state_hpd":
        subset = group[group[target_column].eq("state_ball")]
    for method in ["exact_known", "known", "known_forgetting", "misspecified", "learned"]:
        selected = subset[subset.method.eq(method)]
        if selected.empty:
            continue
        xs, ys, lows, highs = [], [], [], []
        for probability, rows in selected.groupby("nominal", sort=True):
            if "coverage" in rows:
                if len(rows) != 1:
                    raise ValueError("Calibration summary has duplicated probability cells")
                row = rows.iloc[0]
                inclusion, lower, upper = row.coverage, row["lower"], row["upper"]
            else:
                from scipy.stats import beta as beta_distribution
                binary = rows.groupby("episode").covered.mean()
                if not np.isin(binary, [0., 1.]).all():
                    raise ValueError("Each trial must provide one binary coverage outcome")
                count, successes = len(binary), int(binary.sum())
                inclusion = successes / count
                lower = (0. if successes == 0 else
                         beta_distribution.ppf(.025, successes, count-successes+1))
                upper = (1. if successes == count else
                         beta_distribution.ppf(.975, successes+1, count-successes))
            xs.append(probability)
            ys.append(inclusion)
            lows.append(lower)
            highs.append(upper)
        ax.plot(xs, ys, marker=marker(method), color=color(method),
                linestyle="--" if method == "misspecified" else "-",
                label=method_label(method))
        ax.errorbar(xs, ys,
            yerr=[np.asarray(ys)-np.asarray(lows), np.asarray(highs)-np.asarray(ys)],
            fmt="none", ecolor=color(method), alpha=.65, elinewidth=.7, capsize=2.)
    ax.plot([0., 1.], [0., 1.], "--", color="#70777D", linewidth=1.)
    ax.set(xlim=(.43, 1.02), ylim=(-.01, 1.03), xticks=[.5, .8, .95],
           xlabel="Credible level", ylabel="Empirical coverage")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#BFC4C8", alpha=.30, linewidth=.45)
    ax.text(.015, .985, tag, transform=ax.transAxes, ha="left", va="top")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center",
           ncol=2, frameon=False, fontsize=10.)
'''

_WIDTHS = r'''
frame = _normalize(evidence.frame("prediction", "calibration"))
frame = frame[frame.split.eq("test") & frame.cohort.eq("calibration")
    & frame.stress.eq("matched") & frame.t.eq(64) & np.isclose(frame.nominal, .95)]
group = frame[frame.condition.eq(REGIME)]
methods = ["exact_known", "known_forgetting", "misspecified", "learned"]
count = int(evidence.config("prediction")["calibration_test_episodes"])
quantities = [
    ("state_ball", "Chordal radius"),
    ("clean_response", r"Interval width of $q^H Pq$" if example == "beamforming"
        else r"Interval width of $e_1^T Pq$"),
    ("noisy_observation", r"Interval width of $\operatorname{Re}(u_{t,1})$"),
]
fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=DPI, layout="constrained")
for ax, (kind, xlabel), tag in zip(axes, quantities, ["(a)", "(b)", "(c)"]):
    arrays = []
    for method in methods:
        rows = group[group.kind.eq(kind) & group.method.eq(method)].sort_values("episode")
        if (len(rows) != count or rows.episode.nunique() != count
                or not rows.numerical_ok.eq(True).all()):
            raise ValueError("Report-size cohort is missing, duplicated or unresolved")
        lower, upper = rows.lower.to_numpy(), rows.upper.to_numpy()
        width = upper - lower
        if (not np.isfinite(width).all() or np.any(width < 0)
                or not np.allclose(width, rows.width, rtol=1e-12, atol=1e-12)):
            raise ValueError("Saved interval endpoints and width disagree")
        if kind == "state_ball":
            if not np.allclose(lower, 0., atol=1e-14) or np.any(upper > 1.+1e-10):
                raise ValueError("The state ball must use squared chordal distance")
            width = np.sqrt(upper)
        arrays.append(width)
    boxes = ax.boxplot(arrays, vert=False, patch_artist=True,
        medianprops={"color": "white", "linewidth": 1.2},
        flierprops={"markersize": 1.8, "alpha": .35})
    ax.set_yticks(np.arange(1, len(methods)+1),
                  labels=[method_label(method) for method in methods])
    for box, method in zip(boxes["boxes"], methods):
        box.set_facecolor(color(method))
    ax.set_xlabel(xlabel)
    ax.set_xlim(left=0.)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#BFC4C8", alpha=.30, linewidth=.45)
    ax.text(.015, .985, tag, transform=ax.transAxes, ha="left", va="top")
'''

_SCORES = r'''
frame = _normalize(evidence.frame(
    "prediction", "prediction_scores", "scores", "prediction_adaptation", "adaptation"))
if "split" in frame and frame.split.eq("test").any():
    frame = frame[frame.split.eq("test")]
if "stage" in frame and frame.stage.eq("test").any():
    frame = frame[frame.stage.eq("test")]
if "control" in frame:
    frame = frame[frame.control.eq("mean")]
if "cohort" in frame:
    frame = frame[frame.cohort.eq("calibration")]
if "stress" in frame:
    frame = frame[frame.stress.eq("matched")]
if frame.empty:
    raise ValueError("Predictive scores require the matched calibration observations")
if "method" in frame and frame.method.eq("learned").any():
    frame = frame[frame.method.eq("learned")]
gaussian = _column(frame, "gaussian_score", "gaussian_log_score")
mixture = _column(frame, "prequential_score", "mixture_log_score", "mixture_score")
predicted = next((name for name in ("risk_predicted", "predicted_risk", "posterior_risk")
                  if name in frame), None)
realized = next((name for name in ("risk_realized", "realized_risk", "response_loss", "future_task_loss")
                 if name in frame), None)
if predicted is None or realized is None:
    raise KeyError("Predictive scores need saved predicted and realized response risks")
keys = [name for name in ("condition", "regime", "episode") if name in frame]
columns = list(dict.fromkeys([gaussian, mixture, predicted, realized]))
episode = frame.groupby(keys, sort=False)[columns].mean().reset_index()
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI, layout="constrained")
axes[0].scatter(episode[gaussian], episode[mixture], s=10, alpha=.45, color=color("mean"))
axes[1].scatter(episode[predicted], episode[realized], s=10, alpha=.45, color=color("mean"))
for ax, x, y in [(axes[0], episode[gaussian], episode[mixture]),
                  (axes[1], episode[predicted], episode[realized])]:
    values = np.r_[np.asarray(x, float), np.asarray(y, float)]
    values = values[np.isfinite(values)]
    if len(values):
        low, high = float(values.min()), float(values.max())
        padding = max(.06*(high-low), .01*max(abs(low), abs(high)), 1e-10)
        lower = max(low-padding, .8*low) if low > 0 else low-padding
        ax.plot([lower, high+padding], [lower, high+padding],
                "--", color="#70777D", linewidth=1.)
        ax.set(xlim=(lower, high+padding), ylim=(lower, high+padding))
axes[0].set(xlabel="Mean Gaussian log predictive density",
            ylabel="Mean Bingham-mixture log predictive density")
axes[1].set(xlabel="Predicted squared response error",
            ylabel="Observed squared response error")
for ax, tag in zip(axes, ["(a)", "(b)"]):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#BFC4C8", alpha=.30, linewidth=.45)
    ax.text(.015, .985, tag, transform=ax.transAxes, ha="left", va="top")
'''


def build_cells(example):
    """Return the seven missing calibration pages and one score-comparison page."""
    if example not in ("beamforming", "contact"):
        raise ValueError(example)
    cells = []
    coverage_pages = {
        1: ("prior_predictive", 32),
        3: ("fixed_state", 32),
        4: ("fixed_state", 64),
        5: ("drifting", 32),
    }
    names = {"prior_predictive": "Prior-predictive sampling",
             "fixed_state": "Fixed-state repetitions", "drifting": "Subspace drift"}
    for page, (regime, report_time) in coverage_pages.items():
        body = f'REGIME = {regime!r}\nREPORT_TIME = {report_time}\n' + dedent(_COVERAGE)
        note = (f"{names[regime]} at $t={report_time}$. Panels show the latent subspace, "
                "the clean scalar response and the next noisy scalar observation, respectively. "
                "Error bars are pointwise exact binomial 95% confidence intervals across independent trials; "
                "the diagonal indicates agreement between credible level and empirical coverage.")
        cells.append(plot_cell(example, "predictive_calibration", page, body,
            title=f"Prediction coverage at $t={report_time}$: {names[regime].lower()}", note=note))
    for page, regime in [(7, "prior_predictive"), (8, "fixed_state"), (9, "drifting")]:
        body = f'REGIME = {regime!r}\n' + dedent(_WIDTHS)
        note = (f"{names[regime]} at $t=64$. Each value is one independent trial's 95% report size. "
                "Panel (a) uses the chordal radius, the square root of the saved squared-distance threshold. "
                "Panels (b,c) use upper minus lower endpoints in the three-dimensional calibration fixture units. "
                "Boxes show medians and interquartile ranges, with 1.5-IQR whiskers and every outlier; "
                "these are distributions of report sizes, not confidence intervals for their mean.")
        cells.append(plot_cell(example, "predictive_calibration", page, body,
            title=f"Sizes of 95% reports: {names[regime].lower()}", note=note))
    cells.append(plot_cell(example, "predictive_scores", 1, dedent(_SCORES),
        title="Predictive density and response error",
        note="Each point averages reporting times within one independent trial and probability regime. "
             "The left panel compares the normalized Bingham-mixture density with its Gaussian moment approximation; "
             "higher mean log predictive density is better. The right panel compares predicted posterior squared "
             "response error with observed squared error on the same targets. The dashed lines indicate equality; "
             "they are reference lines, not fitted regressions."))
    return cells
