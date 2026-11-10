"""Inline application tracking-error plots from the frozen checkpoint records."""
from scripts.notebook_supplementary_common import plot_cell


_BODY = r'''
from scipy.stats import t as student_t

if example == "beamforming":
    tracking_rows = _normalize(evidence.frame("performance", "moving_checkpoints"))
    tracking_rows = tracking_rows[tracking_rows.condition.eq("smooth") & tracking_rows.n.eq(32)].copy()
    square_column = "response_map_error"
else:
    tracking_rows = _normalize(evidence.frame("performance", "metrics"))
    tracking_rows = tracking_rows[tracking_rows.condition.eq("sphere") & tracking_rows.endpoint.eq("checkpoint")].copy()
    square_column = "response_map_frobenius"

assert not tracking_rows.empty
assert not tracking_rows.duplicated(["method", "episode", "t"]).any()
assert np.all(tracking_rows[square_column].dropna().to_numpy() >= 0.)
tracking_rows["matrix_error"] = np.sqrt(tracking_rows[square_column] / 2.)
if example == "contact":
    # The selected point methods report plane projectors. Their chordal
    # distance is the projector Frobenius distance divided by sqrt(2).
    tracking_rows["point_error"] = tracking_rows["matrix_error"]
else:
    tracking_rows["point_error"] = tracking_rows["chordal_error"]

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI, layout="constrained")
specifications = [
    (axes[0], "point_error", ("mode", "great_window", "great_ew", "grouse")),
    (axes[1], "matrix_error", ("mean", "mode", "iso")),
]
for ax, metric, methods in specifications:
    for method in methods:
        selected = tracking_rows[tracking_rows.method.eq(method)]
        if selected.empty:
            continue
        points = []
        for time, group in selected.groupby("t", sort=True):
            values = group.set_index("episode")[metric].to_numpy(float)
            # Every curve uses the same independent test-trial repetition unit.
            assert len(values) == 256 and np.isfinite(values).all()
            mean, lower, upper = _mean_interval(values)
            half = student_t.ppf(.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values))
            np.testing.assert_allclose([mean, lower, upper],
                [values.mean(), values.mean()-half, values.mean()+half], rtol=1e-12, atol=1e-14)
            points.append((time, mean, lower, upper))
        points = np.asarray(points)
        ax.plot(points[:,0], points[:,1], marker=marker(method), color=color(method), label=method_label(method))
        ax.fill_between(points[:,0], points[:,2], points[:,3], color=color(method), alpha=.10)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of observations")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#BFC4C8", alpha=.30, linewidth=.45)
    ax.legend(frameon=False, fontsize=10., loc="best")
axes[0].set_ylabel(r"$d_{\mathrm{ch}}(\widehat{\mathcal X}_t,\mathcal X_t)$")
axes[1].set_ylabel(r"$\|D_t-P_{\mathcal X_t}\|_{\mathrm F}/\sqrt{2}$")
for ax, tag in zip(axes, ["(a)", "(b)"]):
    ax.text(.015, .985, tag, transform=ax.transAxes, ha="left", va="top")
'''


def build_cells(example):
    setting = ('The saved smooth-motion array experiment uses 32 sensors and a rank-one interference subspace.'
        if example == 'beamforming' else
        'The saved sphere-contact experiment tracks a rank-two plane in three dimensions using measured frame changes.')
    note = setting + r''' Panel (a) reports chordal distance between the point-estimated and generating subspaces,
$d_{\mathrm{ch}}(\widehat{\mathcal X},\mathcal X)=\|P_{\widehat{\mathcal X}}-P_{\mathcal X}\|_{\mathrm F}/\sqrt{2}$.
Panel (b) reports the same scaled Frobenius error for the method's reported matrix $D_t$;
for the posterior mean and isotropic control, $D_t$ need not be a projector, so this is a matrix error rather than a subspace distance.
Curves are means across the same 256 independent test trials, with pointwise 95% Student confidence intervals.
The two panels use their declared moving application cohorts, rather than the separate theorem fixtures.
The generating projector is used only for evaluation and is not supplied to the estimators.'''
    return [plot_cell(example, 'tracking_error', 1, _BODY,
        title='Tracking error in the application', note=note)]
