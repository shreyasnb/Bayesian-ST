"""Visible Matplotlib cells for the contact notebook.

Templates generate ordinary notebook source, never a runtime plotting facade.
Each generated cell owns its complete artists, labels, figure controls and save.
"""
from textwrap import dedent

DATA_IMPORT = ('from lib.notebook_contact_data import '
               'load_contact_plot_data, contact_plot_input_hashes')

_STYLE = r'''
LABELS = {
    "mean": r"Posterior mean $\overline P_t^+$", "mode": r"Mode $P_{\widehat{\mathcal X}_t}$",
    "iso": r"Isotropic mean $\overline P_t^{\mathrm{iso}}$", "isotropic": r"Isotropic mean $\overline P_t^{\mathrm{iso}}$",
    "regularized": "Regularized projector", "great_window": "GREAT (sliding window)",
    "known": "True variances", "exact_known": r"True variances, $\beta=1$",
    "known_forgetting": r"True variances, $\beta=0.90$",
    "misspecified": "Fixed variance estimates", "learned": "EM variance estimates",
}
COLORS = {"mean":"#0072B2", "mode":"#D55E00", "iso":"#CC79A7", "isotropic":"#CC79A7",
          "regularized":"#8C564B", "great_window":"#009E73", "known":"#111111",
          "exact_known":"#111111", "known_forgetting":"#8172B2",
          "misspecified":"#999999", "learned":"#0072B2"}
MARKERS = {"mean":"o", "mode":"s", "iso":"^", "isotropic":"^", "regularized":"P",
           "great_window":"D", "known":"s", "exact_known":"s", "known_forgetting":"D",
           "misspecified":"^", "learned":"o"}
'''


def _cell(name, page, figsize, code, style=True):
    section = 'main' if name in {'information', 'task_performance', 'posterior_anisotropy'} else 'supplementary'
    source = f'''# Edit these controls, the labels below, or any Matplotlib call in this cell.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D

data = load_contact_plot_data(ROOT, {name!r}, page={page})
DPI = 300
FIGSIZE = {figsize!r}
SAVE_PDF = True
PDF_PATH = EDITABLE_OUTPUT / {f'{name}_page{page}.pdf'!r}
DRAFT_SECTION = {section!r}
DRAFT_PATH = ROOT / 'results' / DRAFT_SECTION / 'contact' / PDF_PATH.name
'''
    if style:source += _STYLE
    source += '\n'+dedent(code).strip()+'\n'
    source += '''\n# Custom notebook exports never overwrite the frozen paper catalogue.
if SAVE_PDF:
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_PATH, dpi=DPI, bbox_inches=None)
    DRAFT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DRAFT_PATH.write_bytes(PDF_PATH.read_bytes())
plt.show()
'''
    return source


def _split_panel(name, figsize, filename, code, style=True):
    """Generate visible source for one independently saved notebook figure."""
    source = _cell(name, 1, figsize, code, style=style)
    return source.replace(f"EDITABLE_OUTPUT / '{name}_page1.pdf'",
                          f"EDITABLE_OUTPUT / '{filename}'", 1)


PLOT_CELLS = {}
PLOT_CELLS[('setup', 1)] = _cell('setup', 1, (15,5), r'''
fig = plt.figure(figsize=FIGSIZE, dpi=DPI, layout="constrained")
grid = fig.add_gridspec(1, 2, width_ratios=(1.15, 1.))
ax = fig.add_subplot(grid[0, 0], projection="3d")

# The display mesh is geometric only; it is not an integration rule.
theta, phi = np.meshgrid(np.linspace(0., np.pi, 41), np.linspace(0., 2*np.pi, 81), indexing="ij")
mesh = np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], axis=-1)
surface = ax.plot_surface(*(data["radius"]*1e3*mesh).transpose(2, 0, 1),
                          color="#CAD6E4", alpha=.22, linewidth=0., shade=False)
surface.set_rasterized(True)
positions, normals = data["positions"], data["normals"]
ax.plot(*(positions*1e3).T, color="#222222", linewidth=1.2)
for index in np.linspace(0, len(positions)-1, 4, dtype=int):
    ax.quiver(*(positions[index]*1e3), *normals[index], length=8., normalize=True,
              color="#D55E00", linewidth=.8)
ax.set_proj_type("ortho")
ax.view_init(elev=25., azim=-50.)
ax.set_box_aspect((1., 1., 1.))
ax.set_xlabel(r"$q_1$ (mm)", labelpad=-1)
ax.set_ylabel(r"$q_2$ (mm)", labelpad=-1)
ax.set_zlabel(r"$q_3$ (mm)", labelpad=-1)
ax.tick_params(labelsize=10., pad=-2)
ax.text2D(.02, .97, "(a)", transform=ax.transAxes, ha="left", va="top")

ax = fig.add_subplot(grid[0, 1])
V, U, clean = data["V"], data["U"], data["clean"]
normal = normals[0]/np.linalg.norm(normals[0])
tangent = V[0]-(V[0]@normal)*normal
tangent = tangent/np.linalg.norm(tangent)
for vector, color, label in [(V[0], "#0072B2", r"$v_t$"),
                              (clean[0], "#D55E00", r"$P_{\mathcal X_t}v_t$")]:
    x, y = 1e3*(vector@tangent), 1e3*(vector@normal)
    ax.annotate("", xy=(x, y), xytext=(0., 0.),
                arrowprops={"arrowstyle":"->", "color":color, "linewidth":1.5})
    ax.plot([], [], color=color, label=label)
ax.scatter(1e3*(U[0]@tangent), 1e3*(U[0]@normal), marker="o", s=18,
           facecolors="white", edgecolors="#CC79A7", label=r"$u_t$")
vectors = np.asarray([np.zeros(3), V[0], clean[0], U[0]])
ax.update_datalim(1e3*np.column_stack([vectors@tangent, vectors@normal]))
ax.autoscale_view()
ax.axhline(0., color="#70777D", linewidth=.8)
ax.set_xlabel("Tangential velocity (mm/s)")
ax.set_ylabel("Normal velocity (mm/s)")
ax.legend(frameon=False, fontsize=10.)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#BFC4C8", alpha=.30, linewidth=.45)
ax.text(.015, .985, "(b)", transform=ax.transAxes, ha="left", va="top")
''', style=False)

PLOT_CELLS[('information', 1)] = (
_split_panel('information', (15, 5), 'information_contours.pdf', r'''
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout="constrained")
levels = [.01, .05, .15, .3]
contour_styles = [":", "--", "-.", "-"]
ax.contourf(data["z1"], data["z2"], data["kl"], levels=[0., levels[-1]], colors=["#F4EDF7"])
ax.contour(data["z1"], data["z2"], data["kl"], levels=levels,
                colors=["#7B3294"], linewidths=1.2, linestyles=contour_styles)
handles = [Line2D([], [], color="#7B3294", linestyle=style,
                  label=r"$D_{\mathrm{KL}}="+f"{level:g}"+r"$")
           for level, style in zip(levels, contour_styles)]
ax.legend(handles=handles, frameon=False, fontsize=10., loc="lower left")
ax.plot(0., 0., "+", color="#303030", markersize=6.)
ax.set(xlabel=r"$z_1$", ylabel=r"$z_2$", xlim=(-.78, .78), ylim=(-.78, .78),
            xticks=[-.6, 0., .6], yticks=[-.6, 0., .6])
ax.set_aspect("equal", adjustable="box")

for point, color in zip(data["marked_points"], ["#D55E00", "#0072B2"]):
    ax.plot(*point, "o", color=color, markersize=4.)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#BFC4C8", alpha=.30, linewidth=.45)
ax.text(.015, .985, "(a)", transform=ax.transAxes, ha="left", va="top")
''', style=False).replace("DRAFT_SECTION = 'main'", "DRAFT_SECTION = 'supplementary'") +
_split_panel('information', (15, 5), 'information_rotation.pdf', r'''
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout="constrained")
for values, label, color, style, point, angle, marked_kl in zip(
        [data["kl_first"], data["kl_second"]],
        [r"$z=(\tan\delta,0)^\top$", r"$z=(0,\tan\delta)^\top$"],
        ["#D55E00", "#0072B2"], ["-", "-"], data["marked_points"],
        data["marked_angles"], data["marked_kl"]):
    ax.plot(data["angle_deg"], values, style, color=color, label=label)
    ax.plot(angle, marked_kl, "o", color=color, markersize=4.)
# Evaluate the two endpoints with the same exact KL function and saved-input Fisher matrix.
from lib.final_information_visuals import exact_contact_kl
connector_angles = np.asarray(data["marked_angles"], dtype=float)
connector_kl = exact_contact_kl(np.asarray(data["marked_points"]), {"F": data["fisher_matrix"]})
np.testing.assert_allclose(connector_kl, data["marked_kl"], rtol=0., atol=1e-14)
ax.plot(connector_angles, connector_kl, "-", color="#8A8A8A", linewidth=1.,
        zorder=8, label="_contact_kl_connector")
for angle, divergence, label in zip(connector_angles, connector_kl, [r"$5^\circ$", r"$30^\circ$"]):
    ax.annotate(label, xy=(angle, divergence), xytext=(0, 9), textcoords="offset points",
                ha="center", va="bottom", fontsize=TICK_FONT_SIZE, zorder=9)
ax.set(xlabel=r"$\delta\ ({}^\circ)$", ylabel=r"$D_{\mathrm{KL}}(p_0\,\Vert\,p_z)$", xlim=(0.,35.))
ax.set_yscale("log")
ax.set_yticks([1e-5, 1e-2, 1e1])
ax.minorticks_off()
ax.legend(frameon=False, fontsize=10., loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#BFC4C8", alpha=.30, linewidth=.45)

''', style=False)
)

PLOT_CELLS[('task_performance', 1)] = _cell('task_performance', 1, (15,5), r'''
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI, layout="constrained")
for curve in data["curves"]:
    method, points = curve["method"], curve["points"]
    axes[0].plot(points[:,0], points[:,1], marker=MARKERS[method], color=COLORS[method], label=LABELS[method])
    if method == "mean":
        axes[0].fill_between(points[:,0], points[:,2], points[:,3], color=COLORS[method], alpha=.10)
axes[0].set_xlabel(r"Measurement index $t$")
axes[0].set_ylabel(r"$\mathcal E_t(D_t)$ $((\mathrm{mm}/\mathrm{s})^2)$")
axes[0].margins(y=.12)
for row, point in enumerate(data["absolute"], 1):
    method = point["method"]
    axes[1].errorbar(point["mean"], row,
        xerr=[[point["mean"]-point["lower"]], [point["upper"]-point["mean"]]],
        fmt=MARKERS[method], color=COLORS[method], capsize=3., markersize=5.)
axes[1].set(yticks=np.arange(1,len(data["absolute"])+1),
    yticklabels=[LABELS[point["method"]] for point in data["absolute"]],
    ylim=(.5,len(data["absolute"])+.5),
    xlabel=r"Time-averaged $\mathcal E_t(D_t)$ $((\mathrm{mm}/\mathrm{s})^2)$")
for ax, label in zip(axes, ["(a)", "(b)"]):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#BFC4C8", alpha=.30, linewidth=.45)
    ax.text(.015,.985,label,transform=ax.transAxes,ha="left",va="top")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False, fontsize=10.)
''')

_DIRECTIONAL_RISK_DATA = r'''
# Use the saved unit-query risks for v(theta)/v0 = cos(theta)e1 + sin(theta)e2.
# These are already r_t(v(theta))/v0**2; do not multiply or divide by v0**2.
ANISOTROPY_PATH = ROOT / 'results/contact/data/paper/performance/anisotropy.npz'
with np.load(ANISOTROPY_PATH, allow_pickle=False) as saved:
    query_family = np.asarray(saved['query_family']) == 0
    theta = np.asarray(saved['query_angles'])[query_family].copy()
    unit_queries = np.asarray(saved['unit_queries'])[query_family].copy()
    risk_full = np.asarray(saved['risks_full'])[query_family].copy()
    risk_iso = np.asarray(saved['risks_isotropic'])[query_family].copy()
np.testing.assert_allclose(unit_queries,
    np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)]),
    rtol=0., atol=1e-14)
data = dict(data, theta=theta, risk_full=risk_full, risk_iso=risk_iso)
'''

PLOT_CELLS[('posterior_anisotropy', 1)] = (
_split_panel('posterior_anisotropy', (15, 5), 'posterior_anisotropy_spheres.pdf', r'''
# All display choices are local to this cell. The saved density values stay fixed.
ELEVATION, AZIMUTH = 25., -50.
COLOR_DECADES = 4.
DENSITY_COLORS = [(0.,"#263a8b"),(.28,"#55209a"),(.53,"#b33695"),
                  (.72,"#ed78a4"),(.88,"#ffc85a"),(1.,"#f4f52b")]
cmap = colors.LinearSegmentedColormap.from_list("density_blue_pink_yellow", DENSITY_COLORS, N=256)
cmap.set_under(cmap(0.)); cmap.set_over(cmap(1.))
peak = max(float(np.max(values)) for values in data["log_density"])
norm = colors.LogNorm(vmin=np.exp(peak-COLOR_DECADES*np.log(10.)), vmax=np.exp(peak), clip=True)
fig = plt.figure(figsize=FIGSIZE, dpi=DPI, layout="constrained")
grid = fig.add_gridspec(1,2)
spheres = [fig.add_subplot(grid[0,index],projection="3d") for index in range(2)]
el, az = np.deg2rad([ELEVATION, AZIMUTH])
facing = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
for ax, nodes, log_density, label in zip(spheres,data["nodes"],data["log_density"],["(a)","(b)"]):
    ax.computed_zorder = False
    rgba = cmap(norm(np.exp(log_density)))
    surface = ax.plot_surface(nodes[...,0],nodes[...,1],nodes[...,2],facecolors=rgba,
        shade=False,linewidth=0,antialiased=False,rstride=1,cstride=1,zorder=1)
    surface.set_rasterized(True)
    phi = np.linspace(0.,2*np.pi,241)
    for first, second in [(0,1),(0,2),(1,2)]:
        ring = np.zeros((len(phi),3))
        ring[:,first],ring[:,second] = 1.006*np.cos(phi),1.006*np.sin(phi)
        visible = np.where((ring@facing>=0.)[:,None],ring,np.nan)
        ax.plot(*visible.T,color="#B7C0CB",linewidth=.5,alpha=.55,zorder=3)
    mode = data["mode"]/np.linalg.norm(data["mode"])
    visible_mode = mode if mode@facing>=0. else -mode
    ax.scatter(*visible_mode,marker="o",s=12,color="#D55E00",depthshade=False,zorder=4)
    ax.set_proj_type("ortho"); ax.view_init(elev=ELEVATION,azim=AZIMUTH)
    ax.set_box_aspect((1.,1.,1.))
    ax.set(xlim=(-1.12,1.12),ylim=(-1.12,1.12),zlim=(-1.12,1.12),
        xticks=(-1.,0.,1.),yticks=(-1.,0.,1.),zticks=(-1.,0.,1.))
    ax.set_xlabel(r"$n_1$",labelpad=-2);ax.set_ylabel(r"$n_2$",labelpad=-2)
    ax.set_zlabel("")
    ax.text2D(.95,.67,r"$n_3$",transform=ax.transAxes,rotation=90,ha="center",va="center",fontsize=11.)
    ax.tick_params(labelsize=10.,pad=-2,length=1);ax.grid(False)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.pane.set_facecolor((1.,1.,1.,0.));axis.pane.set_edgecolor((.65,.68,.72,.35))
        axis.line.set_color("#70777D");axis.line.set_linewidth(.65)
    ax.text2D(.02,.97,label,transform=ax.transAxes,ha="left",va="top")
fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=spheres,label=r"$q([n])$",shrink=.64,pad=.015)
''', style=True) +
_split_panel('posterior_anisotropy', (15, 5), 'posterior_anisotropy_risk.pdf', _DIRECTIONAL_RISK_DATA + r'''
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout="constrained")
ax.plot(data["theta"],data["risk_full"],color=COLORS["mean"],label=r"$\overline P_t^+$")
ax.plot(data["theta"],data["risk_iso"],"--",color=COLORS["iso"],label=r"$\overline P_t^{\mathrm{iso}}$")
ax.set_xlabel(r"$\theta$ (rad)")
ax.set_ylabel(r"$r_t(v(\theta))/v_0^2$")
ax.set(xticks=np.linspace(0.,2*np.pi,5),xticklabels=[r"$0$",r"$\pi/2$",r"$\pi$",r"$3\pi/2$",r"$2\pi$"])
ax.legend(frameon=False,fontsize=10.)
ax.spines[["top","right"]].set_visible(False)
ax.grid(color="#BFC4C8",alpha=.30,linewidth=.45)
ax.text(.015,.985,"(c)",transform=ax.transAxes,ha="left",va="top")
''', style=True).replace(
    '    DRAFT_PATH.write_bytes(PDF_PATH.read_bytes())\nplt.show()',
    '''    DRAFT_PATH.write_bytes(PDF_PATH.read_bytes())
    PNG_PATH = PDF_PATH.with_suffix('.png')
    fig.savefig(PNG_PATH, dpi=DPI, bbox_inches=None)
    DRAFT_PNG_PATH = DRAFT_PATH.with_suffix('.png')
    DRAFT_PNG_PATH.write_bytes(PNG_PATH.read_bytes())
plt.show()''')
)

PLOT_CELLS[('noise_adaptation', 1)] = _cell('noise_adaptation', 1, (15,5), r'''
fig = plt.figure(figsize=FIGSIZE,dpi=DPI,layout="constrained")
grid = fig.add_gridspec(1,3)
axes = [fig.add_subplot(grid[0,index]) for index in range(3)]
for curve in data["variance"]:
    method = curve["method"]
    axes[0].step(curve["t"],curve["median"],where="post",color=COLORS[method],
        linestyle="--" if method=="misspecified" else "-",marker=MARKERS[method],label=LABELS[method])
    if method=="learned":
        axes[0].fill_between(curve["t"],curve["lower"],curve["upper"],color=COLORS[method],alpha=.13,step="post")
axes[0].step(data["truth_t"],data["truth"],where="post",color="#222222",linestyle=":",linewidth=1.35,label=r"$\sigma_t^2$")
for event in data["events"]:axes[0].axvline(event,color="#70777D",linestyle=":",linewidth=.7)
axes[0].set_xlabel(data["time_label"])
axes[0].set_ylabel(r"$\widehat\sigma_t^2$ $((\mathrm{mm}/\mathrm{s})^2)$")
for ax, name in zip(axes[1:], ["future","scores"]):
    for curve in data[name]:
        method, points = curve["method"], curve["points"]
        ax.plot(points[:,0],points[:,1],marker=MARKERS[method],color=COLORS[method],
            linestyle="--" if method=="misspecified" else "-",label=LABELS[method])
        ax.fill_between(points[:,0],points[:,2],points[:,3],color=COLORS[method],alpha=.09)
    ax.set_xlabel(data["time_label"])
axes[1].set_ylabel("Squared velocity\nerror "+r"$((\mathrm{mm}/\mathrm{s})^2)$")
axes[2].set_ylabel("Log predictive\ndensity")
for ax, label in zip(axes,["(a)","(b)","(c)"]):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(color="#BFC4C8",alpha=.30,linewidth=.45)
    ax.text(.015,.985,label,transform=ax.transAxes,ha="left",va="top")
handles = {}
for ax in axes:
    for handle,label in zip(*ax.get_legend_handles_labels()):handles.setdefault(label,handle)
fig.legend(list(handles.values()),list(handles),loc="outside lower center",ncol=3,frameon=False,fontsize=10.)
''')

_CALIBRATION_CODE = r'''
fig, axes = plt.subplots(1,3,figsize=FIGSIZE,dpi=DPI,layout="constrained")
for ax,panel,label in zip(axes,data["panels"],["(a)","(b)","(c)"]):
    for curve in panel["curves"]:
        method,points = curve["method"],curve["points"]
        ax.plot(points[:,0],points[:,1],marker=MARKERS[method],color=COLORS[method],
            linestyle="--" if method=="misspecified" else "-",label=LABELS[method])
        ax.errorbar(points[:,0],points[:,1],yerr=[points[:,1]-points[:,2],points[:,3]-points[:,1]],
            fmt="none",ecolor=COLORS[method],alpha=.65,elinewidth=.7,capsize=2.)
    ax.plot([0.,1.],[0.,1.],"--",color="#70777D",linewidth=1.)
    ax.set(xlim=(.43,1.02),ylim=(-.01,1.03),xticks=[.5,.8,.95])
    ax.set_xlabel("Credible level");ax.set_ylabel("Empirical coverage")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(color="#BFC4C8",alpha=.30,linewidth=.45)
    ax.text(.015,.985,label,transform=ax.transAxes,ha="left",va="top")
fig.legend(*axes[0].get_legend_handles_labels(),loc="outside lower center",ncol=2,frameon=False,fontsize=10.)
'''
for _page in (2,6):
    PLOT_CELLS[('predictive_calibration',_page)] = _cell('predictive_calibration',_page,(15,5),_CALIBRATION_CODE)

PLOT_CELLS[('computation_time', 1)] = (
_split_panel('computation_time', (15, 5), 'computation_time_scaling.pdf', r'''
fig, ax = plt.subplots(figsize=FIGSIZE,dpi=DPI,layout="constrained")
for curve in data["scaling"]:
    method, interval = curve["method"],curve["interval"]
    label=LABELS[method]+("\nevery observation" if interval==1 else "\nevery "+str(interval)+" observations")
    ax.plot(curve["n"],curve["milliseconds"],marker=MARKERS[method],linestyle=curve["style"],
        color=COLORS[method],label=label)
ax.set_xscale("log",base=2);ax.set_yscale("log")
ax.set_xticks([3,16,128],labels=["3","16","128"])
ax.set_yticks([10.,100.,1000.],labels=[r"$10^1$",r"$10^2$",r"$10^3$"])
ax.minorticks_off()
ax.set_xlabel("Ambient dimension")
ax.set_ylabel("Time for 64\nobservations (ms)")
ax.legend(frameon=False,fontsize=10.,ncol=2,loc="lower center",bbox_to_anchor=(.5,1.02))
ax.spines[["top","right"]].set_visible(False)
ax.grid(color="#BFC4C8",alpha=.30,linewidth=.45)
# ax.text(.015,.985,"(a)",transform=ax.transAxes,ha="left",va="top")
''', style=True) +
_split_panel('computation_time', (15, 7.5), 'computation_time_phases.pdf', r'''
fig, ax = plt.subplots(figsize=FIGSIZE,dpi=DPI,layout="constrained")
phase_labels = {"correction":"Parameter update","action":"Task decision","report":"Uncertainty report",
                "fitting":"EM","overhead":"Other"}
phase_colors = {"correction":"#0072B2","action":"#D55E00","report":"#CC79A7",
                "fitting":"#009E73","overhead":"#B8BEC4"}
positions=np.arange(len(data["rows"]))
bottom=np.zeros(len(positions))
for phase in data["phases"]:
    key,values=phase["key"],phase["values"]
    ax.barh(positions,values,left=bottom,color=phase_colors[key],height=.7,label=phase_labels[key])
    bottom += values
row_labels=[]
for row in data["rows"]:
    label=r"$\overline P_t^+$"
    if row["method"]=="mean_report":label+=" + uncertainty"
    if row["method"]=="adaptive_mean":label+=" + EM ("+str(int(row["em_iterations"]))+" iterations)"
    row_labels.append(label+"\ndecisions every "+str(int(row["action_cadence"]))+" obs.")
ax.set(yticks=positions,yticklabels=row_labels)
ax.set_xlabel("Time for 64 observations (ms)")
ax.set_ylabel("")
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="x",color="#BFC4C8",alpha=.30,linewidth=.45)
# ax.text(.015,.985,"(b)",transform=ax.transAxes,ha="left",va="top")
fig.legend(*ax.get_legend_handles_labels(),loc="outside lower center",ncol=3,frameon=False,fontsize=10.)
''', style=True)
)


from scripts.notebook_publication_style import apply_publication_style
from scripts.notebook_paper_sizing import contact_paper_layout
for _key, _source in list(PLOT_CELLS.items()):
    PLOT_CELLS[_key] = contact_paper_layout(apply_publication_style(_source))


def get_plot_cell(name, page=1):
    return PLOT_CELLS[(name,page)]
