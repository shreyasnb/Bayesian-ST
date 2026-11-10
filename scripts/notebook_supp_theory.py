"""Visible Matplotlib source for the saved tracking and theory figures.

The returned source is copied into notebook cells. This module does not render
figures or alter the frozen numerical implementation.
"""
from scripts.notebook_supplementary_common import plot_cell


_TRACKING_TABLES = r'''
all_rows = _normalize(evidence.frame("theory", "tracking"))
phase = _normalize(evidence.frame("theory", "tracking_phase"))
rows = all_rows.copy()
for flag in ("nonuniform_prior", "zero_signal"):
    if flag in rows:
        rows = rows[~rows[flag].fillna(False).astype(bool)]
if "design" in rows:
    rows = rows[rows.design.eq("balanced")]
base = rows[rows.condition.eq("static") & np.isclose(rows.beta, 1.)]
if base.empty:
    raise ValueError("Missing static tracking cohort")
'''

_TRACKING_DRAW = r'''
times = sorted(tracking_rows.t.unique())
summary = []
for t in times:
    group = tracking_rows[tracking_rows.t.eq(t)]
    for key in ("gamma", "epsilon", "bound", "ceiling", "valid"):
        if group[key].dropna().nunique() > 1:
            raise ValueError("Mixed deterministic tracking cohorts: " + key)
    summary.append((t, group.modal_error.median(), group.modal_error.quantile(.25),
                    group.modal_error.quantile(.75), group.bound.iloc[0],
                    group.ceiling.iloc[0], bool(group.valid.iloc[0])))
points = np.asarray(summary)
tracking_ax.plot(points[:,0], points[:,1], "o-", color=color("mean"), label="Tracking error")
tracking_ax.fill_between(points[:,0], points[:,2], points[:,3], color=color("mean"), alpha=.12)
tracking_ax.plot(points[:,0], points[:,4], "s-", color=color("iso"), label="Theorem bound")
tracking_ax.plot(points[:,0], points[:,5], "--", color="#222222", label="Distance ceiling")
ineligible = points[:,6] == 0
if ineligible.any():
    tracking_ax.scatter(points[ineligible,0], points[ineligible,5], marker="x",
                        color="#70777D", label=r"$\epsilon_t\geq\gamma_t/2$")
tracking_ax.set(xscale="log" if len(times)>1 else "linear",
    yscale="log" if np.all(points[:,1:4]>0) else "linear",
    xlabel=r"Measurement index $t$", ylabel=r"$d_c(\widehat{\mathcal X}_t,\mathcal X_t)$")
if len(times)==1:
    tracking_ax.set_xlim(times[0]*.9, times[0]*1.1)
    tracking_ax.set_xticks(times)
if not np.all(points[:,1:4]>0):
    tracking_ax.set_ylim(-.05, 1.08*points[:,5].max())
tracking_ax.legend(frameon=False, ncol=2, fontsize=LEGEND_FONT_SIZE, loc="lower center", bbox_to_anchor=(.5,1.02))
'''

_TRACKING_PAGE1 = _TRACKING_TABLES + r'''
fig = plt.figure(figsize=FIGSIZE, dpi=DPI, layout="constrained")
grid = fig.add_gridspec(2,2,height_ratios=(1.3,1.))
axes = [fig.add_subplot(grid[0,:]), fig.add_subplot(grid[1,0]), fig.add_subplot(grid[1,1])]
tracking_rows, tracking_ax = base, axes[0]
''' + _TRACKING_DRAW + r'''
last = base[base.t.eq(base.t.max())]
axes[1].scatter(last.epsilon, last.actual_perturbation, s=12, color=color("mean"), alpha=.6)
x, y = np.asarray(last.epsilon), np.asarray(last.actual_perturbation)
finite = np.isfinite(x) & np.isfinite(y)
low, high = min(x[finite].min(),y[finite].min()), max(x[finite].max(),y[finite].max())
padding = max(.06*(high-low), .01*max(abs(low),abs(high)), 1e-10)
lower = max(low-padding,.8*low) if low>0 else low-padding
axes[1].plot([lower,high+padding], [lower,high+padding], "--", color="#70777D", linewidth=1.)
axes[1].set(xlim=(lower,high+padding),ylim=(lower,high+padding))
axes[1].set(xlabel=r"Perturbation allowance $\epsilon_t$", ylabel="Operator\nperturbation")
phase = phase.sort_values(["beta","drift_rate"]).copy()
phase["state"] = phase.valid.astype(bool).astype(int)
table = phase.pivot(index="beta", columns="drift_rate", values="state")
image = axes[2].imshow(table.to_numpy(), origin="lower", aspect="auto",
    cmap=colors.ListedColormap(["#D7DCE1","#0072B2"]),
    norm=colors.BoundaryNorm([-.5,.5,1.5],2), interpolation="nearest")
beta_labels = []
for value in table.index:
    gap = 1-float(value)
    if 0 < gap < .001 and np.isclose(gap,10.**int(np.round(np.log10(gap))),rtol=1e-6,atol=0):
        beta_labels.append(r"$1-10^{" + str(int(np.round(np.log10(gap)))) + r"}$")
    else:
        beta_labels.append(f"{value:.9g}")
axes[2].set(xticks=np.arange(len(table.columns)), xticklabels=[f"{v:g}" for v in table.columns],
    yticks=np.arange(len(table.index)), yticklabels=beta_labels,
    xlabel="Subspace angular speed", ylabel=r"$\beta$")
bar = fig.colorbar(image, ax=axes[2], ticks=[0,1], shrink=.85)
bar.ax.set_yticklabels([r"$\epsilon\geq\gamma/2$", r"$\epsilon<\gamma/2$"])
for ax, tag in zip(axes, ["(a)","(b)","(c)"]):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(color="#BFC4C8", alpha=.30, linewidth=.45)
    ax.text(.015,.985,tag,transform=ax.transAxes,ha="left",va="top")
'''

_TRACKING_PAGE2 = _TRACKING_TABLES + r'''
keys = [key for key in ("condition","beta","nonuniform_prior","zero_signal","design") if key in all_rows]
terminal = all_rows[all_rows.t.eq(all_rows.groupby(keys).t.transform("max"))]
contributions = terminal.groupby(keys,dropna=False)[["initialization","drift","noise","epsilon","gamma"]].first().reset_index()
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout="constrained")
x = np.arange(len(contributions))
bottom = np.zeros(len(x))
for key, label, c in [("initialization","Initialization",color("mode")),
                       ("drift","Unmodelled motion",color("iso")),
                       ("noise","Measurement noise",color("mean"))]:
    ax.bar(x, contributions[key], bottom=bottom, label=label, color=c, width=.7)
    bottom += contributions[key]
ax.scatter(x,.5*contributions.gamma,marker="_",s=75,color="#222222",label=r"$\gamma_T/2$",zorder=4)
labels = []
for row in contributions.to_dict("records"):
    text = {"static":"Static","smooth":"Smooth","abrupt":"Abrupt"}.get(row["condition"],row["condition"])
    if row.get("nonuniform_prior"): text = "Nonuniform prior"
    if row.get("zero_signal"): text = "Zero signal"
    if row.get("design")=="concentrated": text = "Concentrated inputs"
    beta = float(row["beta"])
    gap = 1-beta
    beta_text = (r"1-10^{"+str(int(np.round(np.log10(gap))))+r"}"
        if 0<gap<.001 and np.isclose(gap,10.**int(np.round(np.log10(gap))),rtol=1e-6,atol=0)
        else f"{beta:.9g}")
    labels.append(text+"\n"+r"$\beta="+beta_text+r"$")
ax.set(xticks=x,xticklabels=labels,ylabel=r"$\epsilon_T$")
ax.tick_params(axis="x",rotation=35)
ax.set_yscale("symlog",linthresh=.01)
ax.legend(frameon=False,ncol=2,fontsize=LEGEND_FONT_SIZE,loc="lower center",bbox_to_anchor=(.5,1.02))
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y",color="#BFC4C8",alpha=.30,linewidth=.45)
'''


def _special_tracking(index):
    return _TRACKING_TABLES + r'''
keys = [key for key in ("condition","beta","nonuniform_prior","zero_signal","design") if key in all_rows]
mask = np.zeros(len(all_rows),bool)
for flag in ("nonuniform_prior","zero_signal"):
    if flag in all_rows:
        mask |= all_rows[flag].fillna(False).astype(bool).to_numpy()
if "design" in all_rows:
    mask |= all_rows.design.ne("balanced").to_numpy()
special_groups = list(all_rows[mask].groupby(keys,dropna=False,sort=False))
selection, tracking_rows = special_groups[''' + str(index) + r''']
fig, tracking_ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout="constrained")
''' + _TRACKING_DRAW + r'''
tracking_ax.spines[["top","right"]].set_visible(False)
tracking_ax.grid(color="#BFC4C8",alpha=.30,linewidth=.45)
'''

_CURVATURE = r'''
curve = evidence.frame("theory","curvature")
accuracy = evidence.frame("theory","inverse_fisher_summary","inverse_fisher")
fig = plt.figure(figsize=FIGSIZE, dpi=DPI, layout="constrained")
grid = fig.add_gridspec(2,2)
axes = [fig.add_subplot(grid[0,:]),fig.add_subplot(grid[1,0]),fig.add_subplot(grid[1,1])]
x = _column(curve,"inverseHvariance","inverse_h_variance","inverse_hessian_variance","inverse_curvature","inverse_curvature_variance","laplace_variance")
y = _column(curve,"empirical_localvariance","empirical_local_variance","empirical_variance","empirical_log_coordinate_variance","local_variance")
valid_name = next((name for name in ("local_approximation_eligible","local_valid","local_ok","resolved") if name in curve),None)
valid = np.ones(len(curve),bool) if valid_name is None else curve[valid_name].fillna(False).astype(bool).to_numpy()
axes[0].scatter(curve.loc[valid,x],curve.loc[valid,y],s=14,color=color("mean"),label="Local regime")
if (~valid).any():
    axes[0].scatter(curve.loc[~valid,x],curve.loc[~valid,y],s=18,facecolors="none",edgecolors=color("mode"),marker="^",label="Outside local regime")
x_values, y_values = np.asarray(curve[x]), np.asarray(curve[y])
finite = np.isfinite(x_values)&np.isfinite(y_values)
low,high = min(x_values[finite].min(),y_values[finite].min()),max(x_values[finite].max(),y_values[finite].max())
padding = max(.06*(high-low),.01*max(abs(low),abs(high)),1e-10)
lower = max(low-padding,.8*low) if low>0 else low-padding
axes[0].plot([lower,high+padding],[lower,high+padding],"--",color="#70777D",linewidth=1.)
axes[0].set(xlim=(lower,high+padding),ylim=(lower,high+padding))
if np.min(curve[[x,y]].to_numpy())>0:
    axes[0].set_xscale("log");axes[0].set_yscale("log")
axes[0].set_xlabel(r"$H_A^{-1}$ tangent variance")
axes[0].set_ylabel("Coordinate\nvariance")
axes[0].legend(frameon=False,fontsize=LEGEND_FONT_SIZE)
L = _column(accuracy,"L","observations","t")
mse = _column(accuracy,"mse","tangent_mse","empirical_mse")
ref = _column(accuracy,"reftrace","inverse_fisher_trace","reference_trace")
stationary = accuracy[np.isclose(accuracy.beta,1.)] if "beta" in accuracy else accuracy
if "sigma2" in stationary:
    stationary = stationary[np.isclose(stationary.sigma2,stationary.sigma2.min())]
for label,values,c,style in [("Empirical MSE",stationary[mse],color("mean"),"-"),
                            (r"$\operatorname{tr}(\mathcal F^{-1})$",stationary[ref],"#70777D","--")]:
    axes[1].plot(stationary[L],values,marker="o",linestyle=style,color=c,label=label)
if {"tangent_mse_lower","tangent_mse_upper"}.issubset(stationary):
    axes[1].fill_between(stationary[L],stationary.tangent_mse_lower,stationary.tangent_mse_upper,color=color("mean"),alpha=.12)
axes[1].set_xscale("log");axes[1].set_yscale("log")
axes[1].set_xlabel(r"Stationary block length $L$");axes[1].set_ylabel("Tangent MSE")
axes[1].legend(frameon=False,fontsize=LEGEND_FONT_SIZE)
fraction = _column(accuracy,"local_fraction","fraction_local")
bias = _column(accuracy,"biasnorm","bias_norm","tangent_bias_norm")
axes[2].plot(stationary[L],stationary[fraction],"o-",color=color("mean"),label="Local fraction")
normalized_bias = stationary[bias]/np.sqrt(stationary[ref])
axes[2].plot(stationary[L],normalized_bias,"^--",color=color("mode"),label=r"Bias$/\sqrt{\operatorname{tr}(\mathcal F^{-1})}$")
axes[2].set_xscale("log");axes[2].set_xlabel(r"Stationary block length $L$")
axes[2].set_ylabel("Locality /\nbias");axes[2].set_ylim(bottom=0)
axes[2].legend(frameon=False,fontsize=LEGEND_FONT_SIZE)
for ax,tag in zip(axes,["(a)","(b)","(c)"]):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(color="#BFC4C8",alpha=.30,linewidth=.45)
    ax.text(.015,.985,tag,transform=ax.transAxes,ha="left",va="top")
'''

_BLOCK = r'''
checks = evidence.frame("theory","block_checks")
trade = _normalize(evidence.frame("performance","blocks","block_metrics","block_updates"))
fig,axes = plt.subplots(1,3,figsize=FIGSIZE,dpi=DPI,layout="constrained")
L = _column(checks,"L","block_length")
if "metric" in checks and "error" in checks:
    for index,(metric,group) in enumerate(checks.groupby("metric",sort=False)):
        if index>3:break
        points = group.groupby(L).error.max()
        axes[0].plot(points.index,np.maximum(points,1e-16),"o-",label=str(metric).replace("_"," "))
else:
    for name in ("parameter_error","mean_error","density_error","mode_error"):
        if name in checks:
            points = checks.groupby(L)[name].max()
            axes[0].plot(points.index,np.maximum(points,1e-16),"o-",label=name.replace("_error",""))
axes[0].axhline(1e-10,color="#70777D",linestyle="--",linewidth=.8)
axes[0].set_yscale("log");axes[0].set_xlabel(r"Block length $L$")
axes[0].set_ylabel("Aggregation\ndiscrepancy");axes[0].legend(frameon=False,fontsize=LEGEND_FONT_SIZE)
L = _column(trade,"L","block_length")
if "condition" in trade:
    condition = "sphere" if "sphere" in set(trade.condition) else trade.condition.iloc[0]
    trade = trade[trade.condition.eq(condition)]
endpoint = ("true_ratio_db" if example=="beamforming" else
    next((name for name in ("response_mse","response_loss","loss") if name in trade),None))
if example=="beamforming" and endpoint not in trade:
    raise KeyError("The block comparison requires saved exact covariance power")
if endpoint is None and "delayed_response_mse" in trade:
    for column,label,c,m in [("delayed_response_mse","Delayed action",color("mean"),"o"),
                            ("common_endpoint_response_mse","Common endpoint",color("mode"),"s"),
                            ("constant_block_response_mse","Block likelihood",color("iso"),"^")]:
        points = trade.groupby(L)[column].mean()*1e6
        axes[1].plot(points.index,points,marker=m,color=c,label=label)
    groups = []
elif endpoint is None:
    endpoint = _column(trade,"delayed_ratio_db","delayed_loss","mean_delayed_power_db")
    groups = [("Delayed output",trade)]
elif "kind" in trade:
    groups = trade.groupby("kind",sort=False)
elif "comparison" in trade:
    groups = trade.groupby("comparison",sort=False)
elif "schedule" in trade:
    groups = trade.groupby("schedule",sort=False)
else:
    groups = [("Delayed output",trade)]
for index,(name,group) in enumerate(groups):
    points = group.groupby(L)[endpoint].mean()
    axes[1].plot(points.index,points*(1e6 if example=="contact" and "mse" in endpoint else 1.),
        marker=("o","s","^")[index%3],label={"delayed_exact":"Delayed exact aggregation","constant_state_window":"Block likelihood"}.get(str(name),str(name).replace("_"," ")))
axes[1].set_xlabel(r"Block length $L$")
axes[1].set_ylabel(r"$\ell_t$ (dB)" if example=="beamforming" else r"$\mathcal E_t(D_t)$ $((\mathrm{mm}/\mathrm{s})^2)$")
axes[1].legend(frameon=False,fontsize=LEGEND_FONT_SIZE)
delay = next((name for name in ("delay_s","mean_delay_s","action_delay_s","maximum_output_age_s","maximum_delay_s") if name in trade),None)
cost = next((name for name in ("processing_ms","total_ms","computation_ms","elapsed_ms") if name in trade),None)
if cost is not None:
    points = trade.groupby(L)[cost].mean()
    axes[2].plot(points.index,points,"o-",color=color("mean"))
    axes[2].set_xlabel(r"Block length $L$");axes[2].set_ylabel("Computation time (ms)")
elif delay is not None:
    points = trade.groupby(L)[delay].mean()
    axes[2].plot(points.index,points,"o-",color=color("mode"))
    axes[2].set_xlabel(r"Block length $L$");axes[2].set_ylabel("Action delay (s)")
else:
    raise KeyError("Block tradeoff needs a saved cost or delay")
for ax,tag in zip(axes,["(a)","(b)","(c)"]):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(color="#BFC4C8",alpha=.30,linewidth=.45)
    ax.text(.015,.985,tag,transform=ax.transAxes,ha="left",va="top")
'''


def build_cells(example):
    """Return the six missing theory pages for one application."""
    tracking_note = (r"Tracking error is the chordal distance "
        r"$d_c(\widehat{\mathcal X}_t,\mathcal X_t)="
        r"\|P_{\widehat{\mathcal X}_t}-P_{\mathcal X_t}\|_F/\sqrt{2}$. "
        r"The curve and shading show the median and interquartile range across Monte Carlo simulations. "
        r"The fixed-time bound is evaluated only when $\epsilon_t<\gamma_t/2$; "
        r"crosses retain ineligible conditions. The true path, gap and perturbation are analysis quantities, not online credible radii.")
    return [
        plot_cell(example,"tracking_bound",1,_TRACKING_PAGE1,title="Tracking error and the fixed-time bound",note="This is the saved static, uniform-prior theory fixture with $\\beta=1$. "+tracking_note),
        plot_cell(example,"tracking_bound",2,_TRACKING_PAGE2,title="Contributions to the tracking bound",note=r"The terminal allowance is $\epsilon_T=$ initialization + motion + noise. The separate marks show $\gamma_T/2$; contributions remain defined when the gap is zero."),
        plot_cell(example,"tracking_bound",3,_special_tracking(0),title=("Tracking with zero signal" if example=="beamforming" else "Tracking with concentrated inputs"),note=tracking_note+" The first saved special fixture is kept separate from the balanced-input, nonzero-signal reference."),
        plot_cell(example,"tracking_bound",4,_special_tracking(1),title="Tracking with a nonuniform prior",note=tracking_note+" This saved special fixture has a nonuniform initial prior. Ineligible bounds are not tested for violations."),
        plot_cell(example,"posterior_curvature",1,_CURVATURE,title="Posterior curvature and local estimation error",note=r"$H_A$ is the Hessian of the negative invariant log density at its mode, in tangent coordinates; $\mathcal F$ is the expected Fisher matrix. Their inverses are local references. The coordinate-variance ordinate is the empirical variance in the saved log coordinates. The panels retain cases outside the recorded local regime, estimation bias and the fraction of Monte Carlo simulations satisfying the locality criterion."),
        plot_cell(example,"block_updates",1,_BLOCK,title="Block aggregation and delayed decisions",note="Exact aggregation, delayed actions and a constant-state block likelihood are separate comparisons. Aggregation errors below 1e-16 use that display floor; saved tables retain their full values. Prediction comparisons use the same block endpoints and the recorded condition. The block-likelihood comparison assumes one fixed subspace over its observation window; delayed aggregation retains the streamed estimate and reports after a block arrives."),
    ]
