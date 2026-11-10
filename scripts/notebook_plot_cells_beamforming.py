"""Complete editable Matplotlib cells for the beamforming walkthrough.

The strings are inserted as notebook source, not executed as hidden renderers.
All selection and numerical summaries are provided by the data-only reader.
"""
from __future__ import annotations

from textwrap import dedent

DATA_IMPORT = 'from lib.notebook_beamforming_data import load_beamforming_plot_data'

STYLE_CODE = r'''
STYLE = {
    "mean": (r"$\overline P_t^+$", "#0072B2", "o"),
    "mode": (r"$P_{\widehat{\mathcal X}_t}$", "#D55E00", "s"),
    "iso": (r"$\overline P_t^{\mathrm{iso}}$", "#CC79A7", "^"),
    "isotropic": (r"$\overline P_t^{\mathrm{iso}}$", "#CC79A7", "^"),
    "scm_loaded": ("MVDR with covariance shrinkage", "#484848", "X"),
    "great_window": ("GREAT (sliding window)", "#009E73", "D"),
    "great": ("GREAT (sliding window)", "#009E73", "D"),
    "great_ew": ("GREAT (exponential weighting)", "#56B4E9", "v"),
    "known": ("True variances", "#111111", "s"),
    "exact_known": (r"True variances, $\beta=1$", "#111111", "s"),
    "known_forgetting": ("True variances, forgetting", "#8172B2", "D"),
    "misspecified": ("Fixed variances", "#999999", "^"),
    "learned": ("Learned variances", "#0072B2", "o"),
}
'''


def _cell(name, page, size, body, *, style=True):
    source = f'''# Edit these controls independently for this figure.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

data = load_beamforming_plot_data(ROOT, {name!r}, page={page})
DPI = 300
FIGSIZE = {size!r}
SAVE_PDF = True
PDF_PATH = EDITABLE_OUTPUT / {f'{name}_page{page}.pdf'!r}
'''
    if style:
        source += dedent(STYLE_CODE)
    source += '\n'+dedent(body).strip()+'\n'
    source += '''
# Save only this editable figure; the published result catalogue is untouched.
if SAVE_PDF:
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_PATH, dpi=DPI, bbox_inches=None)
plt.show()
'''
    return source


PLOT_CELLS = {}
PLOT_CELLS['setup', 1] = _cell('setup', 1, (15,5), r'''
fig = plt.figure(figsize=FIGSIZE, dpi=DPI, layout="constrained")
grid = fig.add_gridspec(1, 3, width_ratios=(1., 1.3, 1.))
axes = [fig.add_subplot(grid[0, j]) for j in range(3)]
n = data["n"]
axes[0].plot(np.arange(n), np.zeros(n), color="#70777D", linewidth=1.)
axes[0].scatter(np.arange(n), np.zeros(n), s=10, facecolors="white", edgecolors="#222222", zorder=3)
centre = (n-1)/2
for start, color_, label in [((centre,1.),"#009E73",r"$a$"),
                             ((centre-.35*n,1.),"#D55E00",r"$h_t$")]:
    axes[0].annotate("", xy=(centre,.06), xytext=start,
                     arrowprops={"arrowstyle":"->", "color":color_, "linewidth":1.4})
    axes[0].text(start[0],start[1]+.04,label,color=color_,ha="center")
axes[0].set(xlim=(-1.,n),ylim=(-.18,1.2),yticks=[],xticks=[0,n-1],xlabel="Sensor index")
axes[0].spines[["top","right","left"]].set_visible(False)
image = axes[1].imshow(abs(data["U"][:min(32,len(data["U"]))]).T,
    origin="lower",aspect="auto",cmap="Greys",interpolation="nearest",rasterized=True)
axes[1].set(xlabel="Observation index",ylabel="Sensor index")
fig.colorbar(image,ax=axes[1],label=r"$|u_{t,j}|$",shrink=.85)
for method, pattern in zip(data["methods"],data["patterns"]):
    if method not in ("mean","mode","iso","isotropic"):
        continue
    label,color_,symbol = STYLE[str(method)]
    axes[2].plot(data["scan"],10*np.log10(np.maximum(pattern,1e-12)),color=color_,label=label)
axes[2].axvline(data["desired_deg"],color="#009E73",linestyle=":",linewidth=1.)
axes[2].axvline(data["interfering_deg"],color="#D55E00",linestyle=":",linewidth=1.)
axes[2].set(xlabel="Arrival angle (deg)",ylabel=r"$|w^H a(\theta)|^2$ (dB)")
axes[2].legend(frameon=False,fontsize=10.)
axes[2].spines[["top","right"]].set_visible(False)
axes[2].grid(color="#BFC4C8",alpha=.30,linewidth=.45)
for ax,tag in zip(axes,["(a)","(b)","(c)"]):
    ax.text(.015,.985,tag,transform=ax.transAxes,ha="left",va="top")
''')

PLOT_CELLS['information', 1] = _cell('information', 1, (15,5), r'''
fig, axes = plt.subplots(1,2,figsize=FIGSIZE,dpi=DPI,layout="constrained")
for expected, estimated in data["eigenvalues"]:
    axes[0].scatter(expected,estimated,color="#0072B2",s=12,alpha=.6)
low,high = data["identity_limits"]
axes[0].plot([low,high],[low,high],"--",color="#70777D",linewidth=1.)
axes[0].set(xscale="log",yscale="log",xlim=(low,high),ylim=(low,high),
    xlabel="Analytical Fisher eigenvalue",ylabel="Estimated Fisher eigenvalue")
for curve in data["design"]:
    color_ = "#D55E00" if curve["method"]=="conditional" else "#0072B2"
    label = "Recorded inputs" if curve["method"]=="conditional" else "Inputs marginalized"
    axes[1].plot(curve["x"],curve["y"],"o-",color=color_,label=label)
axes[1].set(xlabel=data["design_xlabel"],ylabel="Fisher coefficient")
axes[1].legend(frameon=False,fontsize=10.)
for ax,tag in zip(axes,["(a)","(b)"]):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(color="#BFC4C8",alpha=.30,linewidth=.45)
    ax.text(.015,.985,tag,transform=ax.transAxes,ha="left",va="top")
''',style=False)

PLOT_CELLS['task_performance', 1] = (
    _cell('task_performance', 1, (7.5, 5), r'''
fig, ax = plt.subplots(figsize=FIGSIZE,dpi=DPI,layout="constrained")
for curve in data["curves"]:
    label,color_,symbol = STYLE[curve["method"]]
    points = curve["points"]
    ax.plot(points[:,0],points[:,1],marker=symbol,color=color_,label=label)
    if curve["method"]=="mean":
        ax.fill_between(points[:,0],points[:,2],points[:,3],color=color_,alpha=.10)
ax.set(xlabel="Number of sensors",ylabel="Output power relative to ideal MVDR (dB)")
fig.legend(*ax.get_legend_handles_labels(),loc="outside lower center",ncol=2,frameon=False,fontsize=10.)
ax.spines[["top","right"]].set_visible(False)
ax.grid(color="#BFC4C8",alpha=.30,linewidth=.45)
ax.text(.015,.985,"(a)",transform=ax.transAxes,ha="left",va="top")
''').replace("'task_performance_page1.pdf'", "'task_performance_output_power.pdf'")
    + '\n\n' + _cell('task_performance', 1, (7.5, 5), r'''
fig, ax = plt.subplots(figsize=FIGSIZE,dpi=DPI,layout="constrained")
labels = [STYLE[item["method"]][0] for item in data["paired"]]
samples = [item["values"] for item in data["paired"]]
boxes = ax.boxplot(samples,vert=False,patch_artist=True,
    medianprops={"color":"white","linewidth":1.2},
    flierprops={"markersize":1.8,"alpha":.35})
# Set tick labels explicitly: boxplot's labels keyword was removed in 3.11.
ax.set_yticks(np.arange(1,len(labels)+1),labels=labels)
for patch,item in zip(boxes["boxes"],data["paired"]):
    patch.set(facecolor=STYLE[item["method"]][1],alpha=.85)
ax.axvline(0.,linestyle="--",color="#222222",linewidth=1.)
ax.set_xlabel("Output power difference: method $-$ posterior mean (dB)")
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="x",color="#BFC4C8",alpha=.30,linewidth=.45)
ax.text(.015,.985,"(b)",transform=ax.transAxes,ha="left",va="top")
''').replace("'task_performance_page1.pdf'", "'task_performance_paired_difference.pdf'")
)

PLOT_CELLS['posterior_anisotropy', 1] = (
    _cell('posterior_anisotropy', 1, (7.5, 5), r'''
fig = plt.figure(figsize=FIGSIZE,dpi=DPI,layout="constrained")
sphere = fig.add_subplot(111,projection="3d")
nodes,log_density = data["nodes"],data["log_density"]
# Color settings remain here so the palette and density range are editable.
cmap = colors.LinearSegmentedColormap.from_list("blue_pink_yellow",[
    (0.,"#263a8b"),(.28,"#55209a"),(.53,"#b33695"),
    (.72,"#ed78a4"),(.88,"#ffc85a"),(1.,"#f4f52b")],N=256)
norm = colors.LogNorm(vmin=data["density_limits"][0],vmax=data["density_limits"][1],clip=True)
sphere.computed_zorder = False
surface = sphere.plot_surface(nodes[...,0],nodes[...,1],nodes[...,2],
    facecolors=cmap(norm(np.exp(log_density))),shade=False,linewidth=0,
    antialiased=False,rstride=1,cstride=1,zorder=1)
surface.set_rasterized(True)
ELEVATION, AZIMUTH = 25., -50.
el,az = np.deg2rad([ELEVATION,AZIMUTH])
facing = np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
phi = np.linspace(0.,2*np.pi,241)
for plane in ((0,1),(0,2),(1,2)):
    ring = np.zeros((len(phi),3))
    ring[:,plane[0]],ring[:,plane[1]] = 1.006*np.cos(phi),1.006*np.sin(phi)
    visible = np.where((ring@facing>=0.)[:,None],ring,np.nan)
    sphere.plot(*visible.T,color="#B7C0CB",linewidth=.5,alpha=.55,zorder=3)
mode = np.array([1.,0.,0.])
visible = mode if mode@facing>=0 else -mode
sphere.scatter(*visible,marker="o",s=12,color="#D55E00",depthshade=False,zorder=4)
sphere.set_proj_type("ortho")
sphere.view_init(elev=ELEVATION,azim=AZIMUTH)
sphere.set_box_aspect((1.,1.,1.))
sphere.set(xlim=(-1.12,1.12),ylim=(-1.12,1.12),zlim=(-1.12,1.12),
    xticks=(-1.,0.,1.),yticks=(-1.,0.,1.),zticks=(-1.,0.,1.))
sphere.set_xlabel(r"$x_1$",labelpad=-2)
sphere.set_ylabel(r"$x_2$",labelpad=-2)
sphere.set_zlabel("")
sphere.text2D(.95,.67,r"$x_3$",transform=sphere.transAxes,rotation=90,ha="center",va="center",fontsize=11.)
sphere.tick_params(labelsize=10.,pad=-2,length=1)
sphere.grid(False)
for axis in (sphere.xaxis,sphere.yaxis,sphere.zaxis):
    axis.pane.set_facecolor((1.,1.,1.,0.))
    axis.pane.set_edgecolor((.65,.68,.72,.35))
    axis.line.set_color("#70777D")
    axis.line.set_linewidth(.65)
fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=sphere,
    label=r"$q(U_3x)/q(\widehat{\mathcal X})$",shrink=.64,pad=.015)
sphere.text2D(.02,.97,"(a)",transform=sphere.transAxes,ha="left",va="top")
''').replace("'posterior_anisotropy_page1.pdf'", "'posterior_anisotropy_sphere.pdf'")
    + '\n\n' + _cell('posterior_anisotropy', 1, (7.5, 5), r'''
fig, ax_spectrum = plt.subplots(figsize=FIGSIZE,dpi=DPI,layout="constrained")
p,pi = data["mean_eigenvalues"],data["iso_eigenvalues"]
ax_spectrum.plot(np.arange(2,len(p)+1),p[1:],"o-",color=STYLE["mean"][1],label=STYLE["mean"][0])
ax_spectrum.plot(np.arange(2,len(pi)+1),pi[1:],"^--",color=STYLE["iso"][1],label=STYLE["iso"][0])
ax_spectrum.set(yscale="log",xlabel=r"Eigenvalue index $j$",ylabel="Eigenvalue")
ax_spectrum.legend(frameon=False,fontsize=10.)
ax_spectrum.spines[["top","right"]].set_visible(False)
ax_spectrum.grid(color="#BFC4C8",alpha=.30,linewidth=.45)
ax_spectrum.text(.015,.985,"(b)",transform=ax_spectrum.transAxes,ha="left",va="top")
''').replace("'posterior_anisotropy_page1.pdf'", "'posterior_anisotropy_eigenvalues.pdf'")
    + '\n\n' + _cell('posterior_anisotropy', 1, (15, 5), r'''
fig, ax_difference = plt.subplots(figsize=FIGSIZE,dpi=DPI,layout="constrained")
ax_difference.step(data["differences"],data["cdf"],where="post",color=STYLE["mean"][1])
ax_difference.axvline(0.,color="#70777D",linestyle="--",linewidth=1.)
ax_difference.set(xlabel="Output power reduction (dB)",ylabel="Empirical CDF")
ax_difference.spines[["top","right"]].set_visible(False)
ax_difference.grid(color="#BFC4C8",alpha=.30,linewidth=.45)
ax_difference.text(.015,.985,"(c)",transform=ax_difference.transAxes,ha="left",va="top")
''').replace("'posterior_anisotropy_page1.pdf'", "'posterior_anisotropy_empirical_cdf.pdf'")
)

PLOT_CELLS['noise_adaptation', 1] = _cell('noise_adaptation', 1, (15,5), r'''
fig,axes = plt.subplots(2,2,figsize=FIGSIZE,dpi=DPI,layout="constrained")
for ax,key in [(axes[0,0],"sigma2"),(axes[0,1],"tau2")]:
    panel = data["variance"][key]
    for trace in panel["traces"]:
        label,color_,symbol = STYLE[trace["method"]]
        ax.step(trace["x"],trace["y"],where="post",color=color_,marker=symbol,
                linestyle="--" if trace["method"]=="misspecified" else "-",label=label)
        if trace["method"]=="learned":
            ax.fill_between(trace["x"],trace["low"],trace["high"],color=color_,alpha=.13,step="post")
    ax.step(panel["x"],panel["truth"],where="post",color="#222222",linestyle=":",
            linewidth=1.35,label=r"$\sigma_t^2$" if key=="sigma2" else r"$\tau_t^2$")
    for event in panel["events"]:
        ax.axvline(event,color="#70777D",linestyle=":",linewidth=.7)
    ax.set(xlabel=data["time_label"],ylabel=r"Noise variance $\sigma^2$" if key=="sigma2" else r"Signal variance $\tau^2$")
for ax,key in [(axes[1,0],"task"),(axes[1,1],"score")]:
    for trace in data[key]:
        label,color_,symbol = STYLE[trace["method"]]
        p = trace["points"]
        ax.plot(p[:,0],p[:,1],marker=symbol,color=color_,
                linestyle="--" if trace["method"]=="misspecified" else "-",label=label)
        ax.fill_between(p[:,0],p[:,2],p[:,3],color=color_,alpha=.09)
    ax.set(xlabel=data["time_label"],ylabel="Output power relative to ideal MVDR (dB)" if key=="task" else "Mean log predictive density")
handles = {}
for ax,tag in zip(axes.flat,["(a)","(b)","(c)","(d)"]):
    for handle,label in zip(*ax.get_legend_handles_labels()):handles.setdefault(label,handle)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(color="#BFC4C8",alpha=.30,linewidth=.45)
    ax.text(.015,.985,tag,transform=ax.transAxes,ha="left",va="top")
fig.legend(list(handles.values()),list(handles),loc="outside lower center",ncol=3,frameon=False,fontsize=10.)
''')

_CALIBRATION_BODY = r'''
fig,axes = plt.subplots(1,3,figsize=FIGSIZE,dpi=DPI,layout="constrained")
for ax,panel,tag in zip(axes,data["panels"],["(a)","(b)","(c)"]):
    for trace in panel["traces"]:
        label,color_,symbol = STYLE[trace["method"]]
        ax.plot(trace["x"],trace["y"],marker=symbol,color=color_,
                linestyle="--" if trace["method"]=="misspecified" else "-",label=label)
        ax.errorbar(trace["x"],trace["y"],
            yerr=[trace["y"]-trace["low"],trace["high"]-trace["y"]],fmt="none",
            ecolor=color_,alpha=.65,elinewidth=.7,capsize=2.)
    ax.plot([0.,1.],[0.,1.],"--",color="#70777D",linewidth=1.)
    ax.set(xlim=(.43,1.02),ylim=(-.01,1.03),xticks=[.5,.8,.95],
        xlabel="Credible level",ylabel="Empirical coverage")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(color="#BFC4C8",alpha=.30,linewidth=.45)
    ax.text(.015,.985,tag,transform=ax.transAxes,ha="left",va="top")
fig.legend(*axes[0].get_legend_handles_labels(),loc="outside lower center",ncol=2,frameon=False,fontsize=10.)
'''
for _page in (2,6):
    PLOT_CELLS['predictive_calibration',_page] = _cell(
        'predictive_calibration',_page,(15,5),_CALIBRATION_BODY)

PLOT_CELLS['computation_time',1] = _cell('computation_time',1,(15,10),r'''
fig = plt.figure(figsize=FIGSIZE,dpi=DPI,layout="constrained")
grid = fig.add_gridspec(2,1)
axes = [fig.add_subplot(grid[0]),fig.add_subplot(grid[1])]
for curve in data["curves"]:
    label,color_,symbol = STYLE[curve["method"]]
    label += ", every observation" if curve["interval"]==1 else ", every "+str(curve["interval"])+" observations"
    axes[0].plot(curve["n"],curve["elapsed"],marker=symbol,linestyle=curve["line"],color=color_,label=label)
axes[0].set_xscale("log",base=2)
axes[0].set_yscale("log")
axes[0].set(xlabel="Number of sensors",ylabel="Time for 64 observations (ms)")
axes[0].legend(frameon=False,fontsize=10.,ncol=2,loc="lower center",bbox_to_anchor=(.5,1.02))
phase_styles = {
    "correction":("Parameter update","#0072B2"),
    "action":("Task decision","#D55E00"),
    "report":("Uncertainty report","#CC79A7"),
    "fitting":("EM","#009E73"),
    "overhead":("Other","#B8BEC4"),
}
positions = np.arange(len(data["phase_rows"]))
bottom = np.zeros(len(positions))
for phase in data["phases"]:
    label,color_ = phase_styles[phase["key"]]
    axes[1].barh(positions,phase["values"],left=bottom,color=color_,height=.7,label=label)
    bottom += phase["values"]
row_labels = []
for row in data["phase_rows"]:
    label = STYLE["mean"][0]
    if row["method"]=="mean_report":label += " + uncertainty"
    if row["method"]=="adaptive_mean":label += " + EM ("+str(int(row["em_iterations"]))+" iterations)"
    row_labels.append(label+"\ndecisions every "+str(int(row["action_cadence"]))+" obs.")
axes[1].set(yticks=positions,yticklabels=row_labels,xlabel="Time for 64 observations (ms)")
fig.legend(*axes[1].get_legend_handles_labels(),loc="outside lower center",ncol=3,frameon=False,fontsize=10.)
for ax,tag in zip(axes,["(a)","(b)"]):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="x" if ax is axes[1] else "both",color="#BFC4C8",alpha=.30,linewidth=.45)
    ax.text(.015,.985,tag,transform=ax.transAxes,ha="left",va="top")
''')


def get_plot_cell(name, page=1):
    """Return ordinary source to place directly into one notebook code cell."""
    return PLOT_CELLS[name,page]
