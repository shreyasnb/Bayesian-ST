"""Visible supplementary comparison cells from the frozen paper cohorts.

Only source strings are assembled here. All Matplotlib artists are emitted
directly into notebook cells, and no statistical study is run by this module.
"""
from scripts.notebook_supplementary_common import plot_cell


_GRID = r'''
from matplotlib.patches import Rectangle

field = "stationary" if example == "beamforming" else "metrics"
frame = _normalize(evidence.frame("performance", field))
if "split" in frame:
    frame = frame[frame.split.eq("test")]
if example == "contact":
    frame = frame[frame.endpoint.eq("checkpoint")]
endpoint = (_finite_column(frame, "heldout_ratio_db") if example == "beamforming"
            else _finite_column(frame, "response_mse"))
scale = 1. if example == "beamforming" else 1e6
groups = (list(frame.groupby("n", sort=True)) if example == "beamforming"
          else list(frame.groupby("condition", sort=False)))

# Use the common color limits of the full declared comparison grid.
prepared = []
maximum = 0.
for identifier, group in groups:
    paired = group.pivot_table(index=["episode", "t"], columns="method",
                              values=endpoint, aggfunc="mean")
    methods = [method for method in [
        "mode", "iso", "mode_loaded", "scm_loaded", "great_window", "great_ew",
        "lw", "regularized", "mean_identity", "mode_identity", "oracle"
    ] if method in paired]
    methods.extend(method for method in paired.columns
                   if method not in methods and method != "mean")
    times = sorted(group.t.unique())
    difference = np.full((len(methods), len(times)), np.nan)
    lower, upper = difference.copy(), difference.copy()
    for i, method in enumerate(methods):
        for j, time in enumerate(times):
            pairs = paired.xs(time, level="t")[[method, "mean"]].dropna()
            if len(pairs):
                difference[i, j], lower[i, j], upper[i, j] = _mean_interval(
                    scale * (pairs[method] - pairs["mean"]))
    finite = np.abs(difference[np.isfinite(difference)])
    maximum = max(maximum, float(finite.max()) if finite.size else 0.)
    prepared.append((identifier, methods, times, difference, lower, upper))

identifier, methods, times, difference, lower, upper = prepared[GROUP_INDEX]
norm = colors.TwoSlopeNorm(vmin=-max(maximum, 1e-8), vcenter=0.,
                          vmax=max(maximum, 1e-8))
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout="constrained")
image = ax.imshow(difference, aspect="auto", cmap="RdBu", norm=norm,
                  interpolation="nearest", rasterized=True)
for i in range(len(methods)):
    for j in range(len(times)):
        if np.isfinite(lower[i, j]) and lower[i, j] <= 0 <= upper[i, j]:
            ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False,
                                  hatch="///", edgecolor="#70777D", linewidth=0.))
labels = []
for method in methods:
    labels.append(r"$\overline P_t^+,\ G_t=I$" if method == "mean_identity" else
                  r"$P_{\widehat{\mathcal X}_t},\ G_t=I$" if method == "mode_identity"
                  else method_label(method, example))
xlabels = ([f"{time/identifier:g}" for time in times] if example == "beamforming"
           else [str(time) for time in times])
ax.set(xticks=np.arange(len(times)), xticklabels=xlabels,
       yticks=np.arange(len(methods)), yticklabels=labels,
       xlabel=r"Observations per sensor $L/n$" if example == "beamforming"
       else "Number of observations")
fig.colorbar(image, ax=ax,
             label="Output power difference (dB)" if example == "beamforming"
             else r"Response MSE difference $((\mathrm{mm}/\mathrm{s})^2)$",
             shrink=.85)
ax.legend(handles=[Patch(facecolor="none", hatch="///", edgecolor="#70777D",
                         label="Interval contains zero")], frameon=False,
          loc="lower center", bbox_to_anchor=(.5, 1.02), fontsize=10.)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(False)
'''


_MOVING = r'''
frame = evidence.frame("performance", "moving_episodes", "episode_metrics", "moving", "stationary")
endpoint = _column(frame, "heldout_ratio_db", "output_oracle_db", "heldout_power")
frame = _episode_losses(frame, endpoint)
condition = list(frame.condition.drop_duplicates())[CONDITION_INDEX]
frame = frame[frame.condition.eq(condition)]
if "n" in frame:
    n = 32 if 32 in set(frame.n) else int(frame.n.iloc[0])
    frame = frame[frame.n.eq(n)]
baseline = frame[frame.method.eq("mean")].set_index("episode")[endpoint]
references = ["mode", "iso", "isotropic", "great_window", "great", "great_ew",
              "mode_loaded", "regularized", "scm_loaded"]
samples, labels, palette = [], [], []
for method in references:
    other = frame[frame.method.eq(method)].set_index("episode")[endpoint]
    common = baseline.index.intersection(other.index)
    if len(common):
        samples.append((other.loc[common] - baseline.loc[common]).to_numpy())
        labels.append(method_label(method, example))
        palette.append(color(method))
if not samples:
    raise ValueError("The saved comparison has no paired trials")

fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout="constrained")
boxes = ax.boxplot(samples, vert=False, patch_artist=True,
                   medianprops={"color": "white", "linewidth": 1.2},
                   flierprops={"markersize": 1.8, "alpha": .35})
ax.set_yticks(np.arange(1, len(labels)+1), labels=labels)
for patch, shade in zip(boxes["boxes"], palette):
    patch.set_facecolor(shade)
    patch.set_alpha(.85)
ax.axvline(0., linestyle="--", color="#222222", linewidth=1.)
ax.set_xlabel("Output power difference: method $-$ posterior mean (dB)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#BFC4C8", alpha=.30, linewidth=.45)
'''


_MISMATCH = r'''
frame = _normalize(evidence.frame("prediction", "prediction_mismatch", "mismatch"))
if "method" in frame and frame.method.eq("learned").any():
    frame = frame[frame.method.eq("learned")]
if "control" in frame:
    frame = frame[frame.control.eq("mean")]
stress = _column(frame, "stress", "mismatch", "condition")
if example == "beamforming":
    numerator = _finite_column(frame, "task_loss_mean")
    denominator = _finite_column(frame, "oracle_power")
    valid = (np.isfinite(frame[numerator]) & np.isfinite(frame[denominator])
             & frame[denominator].gt(0) & frame[numerator].gt(0))
    if not valid.any():
        raise ValueError("The saved mismatch comparison has no positive power pairs")
    endpoint = "observed_oracle_ratio_db"
    frame[endpoint] = np.nan
    frame.loc[valid, endpoint] = 10*np.log10(
        frame.loc[valid, numerator] / frame.loc[valid, denominator])
else:
    endpoint = _finite_column(frame, "response_loss_mean", "response_loss")
score = _column(frame, "prequential_score", "mixture_log_score", "logscore")
conditions = list(frame[stress].drop_duplicates())
condition_labels = {
    "matched": "Matched", "desired_contamination": "Desired signal contamination",
    "unequal_rank_two_powers": "Unequal interference powers",
    "correlated_noise": "Correlated noise", "anisotropic_noise": "Anisotropic noise",
    "fast_drift": "Faster motion", "additional_interference": "Additional interference",
    "known_resistance_correct_noise": "Known resistance",
    "known_resistance_raw_isotropic": "Incorrect noise covariance",
    "anisotropic_damping": "Incorrect resistance", "wrong_resistance": "Incorrect resistance",
    "known_anisotropic_resistance": "Known resistance",
    "friction": "Friction and sticking", "contact_loss": "Contact loss",
    "position_differences": "Position differences",
}
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI, layout="constrained")
for ax, quantity, scale in [(axes[0], endpoint, 1e6 if example == "contact" else 1.),
                             (axes[1], score, 1.)]:
    samples = [frame[frame[stress].eq(condition)].groupby("episode")[quantity]
               .mean().to_numpy()*scale for condition in conditions]
    boxes = ax.boxplot(samples, vert=False, patch_artist=True,
                       medianprops={"color": "white"},
                       flierprops={"markersize": 1.5, "alpha": .35})
    ax.set_yticks(np.arange(1, len(conditions)+1), labels=[
        condition_labels.get(condition, str(condition).replace("_", " "))
        for condition in conditions])
    for index, patch in enumerate(boxes["boxes"]):
        patch.set_facecolor([color("mean"), color("mode"), color("iso"), "#70777D"][index % 4])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#BFC4C8", alpha=.30, linewidth=.45)
axes[0].set_xlabel("Output power relative to ideal MVDR (dB)" if example == "beamforming"
                   else r"Response MSE $((\mathrm{mm}/\mathrm{s})^2)$")
axes[1].set_xlabel("Mean log predictive density")
for ax, label in zip(axes, ["(a)", "(b)"]):
    ax.text(.015, .985, label, transform=ax.transAxes, ha="left", va="top")
'''


def build_cells(example):
    """Return every comparison and mismatch page for one application."""
    if example not in ("beamforming", "contact"):
        raise ValueError("Unknown application: " + str(example))
    cells = []
    identifiers = (32, 64, 128) if example == "beamforming" else (
        "static_balanced", "static_concentrated", "plane", "sphere")
    for index, identifier in enumerate(identifiers):
        label = (f"All comparisons at {identifier} sensors" if example == "beamforming"
                 else "All comparisons: " + identifier.replace("_", " "))
        cells.append(plot_cell(example, "additional_comparisons", index+1,
            _GRID.replace("GROUP_INDEX", str(index)), title=label,
            note="Each color is the mean same-trial difference, method minus posterior mean. "
                 "Positive values favor the posterior mean. Hatching marks a pointwise 95% "
                 "Student interval containing zero; intervals are not simultaneous. "
                 "All predeclared methods and observation counts are retained. "
                 + ("The ideal MVDR reference uses the true interference-plus-noise "
                    "covariance for evaluation." if example == "beamforming" else
                    "Identity-transport controls use $G_t=I$." if identifier in ("plane", "sphere") else "")))
    if example == "beamforming":
        for index, condition in enumerate(("static", "smooth", "abrupt", "noise_change", "power_change")):
            cells.append(plot_cell(example, "additional_comparisons", index+4,
                _MOVING.replace("CONDITION_INDEX", str(index)),
                title="Paired comparisons: " + condition.replace("_", " "),
                note="These boxes retain all same-trial comparator-minus-posterior-mean "
                     "differences for the stated condition at 32 sensors. Each independent "
                     "trial first averages normalized output power over its evaluation "
                     "times and converts that result to dB. Positive differences favor "
                     "the posterior mean. Boxes show trial distributions, not confidence intervals."))
    cells.append(plot_cell(example, "model_mismatch", 1, _MISMATCH,
        title="Model mismatch",
        note="All declared mismatch conditions are retained for the learned-variance "
             "posterior-mean method. Boxes describe independent-trial distributions. "
             "Measurements are scored before updating the model; higher log predictive "
             "density is better. "
             + ("The ideal MVDR reference uses the true covariance. " if example == "beamforming" else "")
             + "Changing a physical observation model is distinct from uncertainty in its parameters."))
    return cells
