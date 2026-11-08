"""Density geometry on spheres, with the state-space interpretation explicit.

A real line in R^3 is represented by two antipodal unit vectors.  Its Bingham
law is evaluated relative to normalized invariant measure, so a uniform law has
q_A=1.  A complex line in C^2 instead maps bijectively to the Bloch sphere;
its Bingham density becomes a spherical von Mises--Fisher density, not a real
antipodally symmetric Bingham law.

For ambient dimension above three, ``restriction_log_density`` evaluates the
original density only on an explicitly supplied real-coefficient great sphere.
It is neither a marginal density nor a new normalized probability on that
sphere.  Plotting never fits a distribution, computes a normalizer, or samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import numpy as np
from .plotting import plt, save
from matplotlib import colors
from matplotlib.lines import Line2D

BLUE = '#0072B2'
ORANGE = '#D55E00'
HPD_COLOR = '#252525'
GREY = '#68737A'
DENSITY_CMAP = colors.LinearSegmentedColormap.from_list(
    'bingham_density_blue_pink',
    ['#245FA8', '#4A65B2', '#8260B1', '#B75DA9', '#E66DAB', '#FF94C2'], N=256)
# A continuous blue-to-pink scale keeps low-density parts blue; values below
# the displayed logarithmic floor keep that same blue instead of becoming white.
DENSITY_CMAP.set_under('#245FA8')
DENSITY_CMAP.set_over('#FF94C2')


@dataclass(frozen=True)
class SphereMesh:
    """Display mesh only; equal angular pixels are not equal-area quadrature."""
    nodes: np.ndarray
    polar: np.ndarray
    azimuth: np.ndarray

    @property
    def shape(self):
        return self.nodes.shape[:-1]


def sphere_mesh(polar=145, azimuth=289):
    """Unit sphere including both poles and a closed azimuthal seam."""
    if polar < 17 or azimuth < 33:
        raise ValueError('A sphere needs at least 17 polar and 33 azimuthal nodes')
    theta = np.linspace(0., np.pi, int(polar))
    phi = np.linspace(0., 2*np.pi, int(azimuth))
    T, F = np.meshgrid(theta, phi, indexing='ij')
    nodes = np.stack((np.sin(T)*np.cos(F), np.sin(T)*np.sin(F), np.cos(T)), axis=-1)
    return SphereMesh(nodes, theta, phi)


def bingham_sphere_mesh(A, polar=145, azimuth=289):
    """Mode-aligned display mesh with more vertices near concentrated caps.

    The nodes still cover the exact unit sphere.  Their angular spacing is
    nonuniform and must never be used as area weights.  This avoids missing a
    narrow posterior peak between display vertices; it does not enlarge that
    peak on the sphere or alter the probability density.
    """
    A = _hermitian(A)
    if A.shape != (3, 3) or np.max(abs(A.imag)) > 1e-12:
        raise ValueError('Adaptive real-sphere display needs a real 3 by 3 parameter')
    values, U = np.linalg.eigh(A.real)
    U = U[:, ::-1]
    gaps = values[-1]-values[-2::-1]
    if gaps[-1] < 1e-10:
        return sphere_mesh(polar, azimuth)
    if polar < 17 or azimuth < 33:
        raise ValueError('A sphere needs at least 17 polar and 33 azimuthal nodes')
    concentration = max(1., .5*np.log1p(gaps[-1]))
    q = np.tanh(concentration*np.linspace(-1., 1., int(polar)))/np.tanh(concentration)
    theta = np.arccos(np.clip(q[::-1], -1., 1.))
    base = np.linspace(0., 2*np.pi, int(azimuth))
    ratio = np.sqrt((1+gaps[0])/(1+gaps[1]))
    phi = np.unwrap(np.arctan2(ratio*np.sin(base), np.cos(base)))
    T, F = np.meshgrid(theta, phi, indexing='ij')
    local = np.stack((np.cos(T), np.sin(T)*np.cos(F), np.sin(T)*np.sin(F)), axis=-1)
    return SphereMesh(local @ U.T, theta, phi)


def _hermitian(A):
    A = np.asarray(A)
    if (A.ndim != 2 or A.shape[0] != A.shape[1] or not np.isfinite(A).all()
            or not np.allclose(A, A.conj().T, atol=1e-10, rtol=1e-12)):
        raise ValueError('A must be a finite symmetric or Hermitian matrix')
    return (A + A.conj().T)/2


def _unit_nodes(nodes):
    nodes = np.asarray(nodes, float)
    if nodes.shape[-1] != 3 or not np.isfinite(nodes).all():
        raise ValueError('Sphere coordinates must be finite three-vectors')
    if not np.allclose(np.sum(nodes*nodes, axis=-1), 1., atol=2e-10, rtol=0):
        raise ValueError('Display coordinates must lie on the unit sphere')
    return nodes


def real_bingham_log_density(A, log_normalizer, nodes):
    """Log q_A on real RP^2, under the normalized invariant measure.

    ``A`` is the parameter of the NORMAL line.  For a contact tangent-plane
    parameter A_t^+, callers must pass -A_t^+ and its rank-one normalizer.
    The measure lifts to dOmega/(4*pi) on the sphere, not dOmega.
    """
    A = _hermitian(A)
    if A.shape != (3, 3) or np.iscomplexobj(A) and np.max(abs(A.imag)) > 1e-12:
        raise ValueError('The real normal sphere needs a real 3 by 3 parameter')
    if not np.isfinite(log_normalizer):
        raise ValueError('Supply the verified normalized-measure log normalizer')
    nodes = _unit_nodes(nodes)
    return np.einsum('...i,ij,...j->...', nodes, A.real, nodes) - float(log_normalizer)


def real_bingham_density(A, log_normalizer, nodes):
    """Invariant density, dimensionless and equal to one for a uniform law."""
    return np.exp(real_bingham_log_density(A, log_normalizer, nodes))


def contact_normal_log_density(A_plane, plane_log_normalizer, nodes):
    """Normal-line density induced by a real Gr(2,3) tangent-plane belief.

    Complement duality gives psi_2(A)=tr(A)+psi_1(-A).  Thus the supplied
    normalizer can be that of the tangent-plane posterior; trace centering is
    not assumed by this conversion.
    """
    A_plane = _hermitian(A_plane)
    return real_bingham_log_density(-A_plane,
        float(plane_log_normalizer)-float(np.trace(A_plane).real), nodes)


def bloch_projectors(nodes):
    """P(r)=(I+r.sigma)/2 for r on S^2; r and -r are distinct complex lines."""
    nodes = _unit_nodes(nodes)
    P = np.empty(nodes.shape[:-1] + (2, 2), complex)
    P[..., 0, 0] = (1 + nodes[..., 2])/2
    P[..., 1, 1] = (1 - nodes[..., 2])/2
    P[..., 0, 1] = (nodes[..., 0] - 1j*nodes[..., 1])/2
    P[..., 1, 0] = P[..., 0, 1].conj()
    return P


def bloch_log_density(A, log_normalizer, nodes):
    """Complex CP^1 density on its Bloch sphere, with measure dOmega/(4*pi)."""
    A = _hermitian(A)
    if A.shape != (2, 2) or not np.isfinite(log_normalizer):
        raise ValueError('The Bloch display needs a 2 by 2 complex-line parameter')
    return np.einsum('ij,...ji->...', A, bloch_projectors(nodes)).real - float(log_normalizer)


def restriction_log_density(A, frame, log_normalizer, nodes):
    """Actual q_A values on [frame @ x], x real and unit in R^3.

    Columns of ``frame`` are three real/complex orthonormal directions.  The
    supplied normalizer belongs to the original ambient rank-one distribution.
    Integrating these values with sphere area does not generally give one.
    The caller must label the display as a restriction, not a marginal.
    """
    A = _hermitian(A)
    E = np.asarray(frame)
    if (E.shape != (len(A), 3) or not np.isfinite(E).all()
            or not np.allclose(E.conj().T @ E, np.eye(3), atol=1e-10, rtol=0)):
        raise ValueError('The restriction frame must have three orthonormal columns')
    if not np.isfinite(log_normalizer):
        raise ValueError('Supply the original ambient log normalizer')
    nodes = _unit_nodes(nodes)
    C = E.conj().T @ A @ E
    return np.einsum('...i,ij,...j->...', nodes, C.real, nodes) - float(log_normalizer)


def complex_simplex_log_density(eigenvalues, log_normalizer, squared_coordinates):
    """Exact phase-integrated amplitude density for a complex line in C^3.

    In an eigenbasis, r_i=|h_i|^2 obey a tilted Dirichlet(1,1,1) law.  Its
    density is 2*exp(sum_i lambda_i*r_i-psi_1(A)) with respect to dr_1 dr_2
    on r_i>=0,sum r_i=1.  This is an actual marginal of squared amplitudes,
    unlike a great-sphere density restriction.  An equilateral display adds
    a constant area Jacobian if one instead labels Euclidean triangle density.
    """
    lam = np.asarray(eigenvalues, float)
    r = np.asarray(squared_coordinates, float)
    if (lam.shape != (3,) or r.shape[-1] != 3 or not np.isfinite(lam).all()
            or not np.isfinite(r).all() or np.min(r) < -1e-12
            or not np.allclose(r.sum(axis=-1), 1., atol=1e-10, rtol=0)
            or not np.isfinite(log_normalizer)):
        raise ValueError('Need three finite eigenvalues and simplex squared coordinates')
    return np.log(2.) + np.einsum('...i,i->...', r, lam) - float(log_normalizer)


def shared_density_norm(log_densities: Sequence[np.ndarray], decades=4.):
    """One log color normalization for all supplied actual density arrays.

    The lowest displayed color clips values below max(q_A)*10**(-decades).
    This changes color resolution only; it does not truncate probability mass.
    """
    if decades <= 0 or not log_densities:
        raise ValueError('Need densities and a positive color range')
    arrays = [np.asarray(a, float) for a in log_densities]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('Density values must be finite')
    maximum = max(float(a.max()) for a in arrays)
    if maximum > np.log(np.finfo(float).max) - 1:
        raise ArithmeticError('Density colors overflow; check the normalizer')
    return colors.LogNorm(vmin=np.exp(maximum-decades*np.log(10)),
                          vmax=np.exp(maximum), clip=True)


def _rotation(rotation):
    if rotation is None:
        angle = np.deg2rad(43.)
        return np.array([[np.cos(angle), 0., -np.sin(angle)],
                         [0., 1., 0.],
                         [np.sin(angle), 0., np.cos(angle)]])
    R = np.asarray(rotation, float)
    if R.shape != (3, 3) or not np.allclose(R.T @ R, np.eye(3), atol=1e-10):
        raise ValueError('Display rotation must be orthogonal')
    return R


def ball_curves(mode, radius, points=721):
    """Two cap boundaries for a centered real-line chordal ball."""
    h = np.asarray(mode, float)
    if h.shape != (3,) or not np.isclose(h @ h, 1., atol=1e-10) or not 0 < radius < 1:
        raise ValueError('Need a unit mode and a radius strictly between zero and one')
    pivot = np.eye(3)[np.argmin(abs(h))]
    e2 = pivot - h*(h @ pivot)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(h, e2)
    phi = np.linspace(0., 2*np.pi, int(points))
    curve = np.sqrt(1-radius*radius)*h[:, None] + radius*(
        e2[:, None]*np.cos(phi) + e3[:, None]*np.sin(phi))
    return [curve.T, -curve.T]


def hpd_curves(A, loss, points=1441):
    """Real rank-one Bingham density-loss boundary, with gaps handled exactly.

    Curve vertices satisfy lambda_max(A)-x^T A x=loss.  Critical azimuths are
    included explicitly, closing both cap boundaries and the broad-belief
    girdle boundaries.  This visualizes a supplied threshold; it is not a mass
    calculation.
    """
    A = _hermitian(A)
    if A.shape != (3, 3) or np.max(abs(A.imag)) > 1e-12 or not np.isfinite(loss) or loss < 0:
        raise ValueError('Need a real 3 by 3 parameter and nonnegative density loss')
    values, U = np.linalg.eigh(A.real)
    U = U[:, ::-1]
    gaps = values[-1] - values[-2::-1]
    if loss == 0 or gaps[-1] == 0 or loss >= gaps[-1]:
        return []
    def coordinates(phi):
        g = gaps[0]*np.cos(phi)**2 + gaps[1]*np.sin(phi)**2
        squared = np.clip(loss/g, 0., 1.)
        radius = np.sqrt(squared)
        axis = np.sqrt(1-squared)
        return np.column_stack([axis, radius*np.cos(phi), radius*np.sin(phi)])

    if loss <= gaps[0]:
        phi = np.linspace(0., 2*np.pi, int(points))
        local = coordinates(phi)
        return [local @ U.T, -local @ U.T]
    alpha = float(np.arcsin(np.sqrt((loss-gaps[0])/(gaps[1]-gaps[0]))))
    curves = []
    for lower, upper in [(alpha, np.pi-alpha), (np.pi+alpha, 2*np.pi-alpha)]:
        phi = np.linspace(lower, upper, max(33, int(points)//2))
        positive = coordinates(phi)
        # The two branches meet at the equator, where the first coordinate
        # is exactly zero.  Keep that identity explicit at floating-point roots.
        positive[[0, -1], 0] = 0.
        negative = positive[::-1].copy()
        negative[:, 0] *= -1
        curves.append(np.vstack([positive, negative[1:]]) @ U.T)
    return curves


def _curve(ax, curves, R, *, color, linestyle='-', linewidth=1.8, label=None,
           visible_direction=None, alpha=1., zorder=12):
    for index, curve in enumerate(curves):
        X = (np.asarray(curve)*1.006) @ R.T
        if visible_direction is not None:
            X = X.copy()
            X[X @ visible_direction < 0] = np.nan
        ax.plot(X[:, 0], X[:, 1], X[:, 2], color=color, ls=linestyle,
                lw=linewidth, alpha=alpha, zorder=zorder,
                label=label if index == 0 else None)


def draw_density_sphere(ax, mesh, log_density, *, norm=None, rotation=None,
                        mode=None, A=None, hpd_loss=None, ball_radius=None,
                        truth=None, truth_label=r'$\mathcal X_t$',
                        title=None, axis_labels=(r'$e_1$', r'$e_2$', r'$e_3$'),
                        elev=22., azim=0., cmap=DENSITY_CMAP,
                        ball_label=r'$d_c=\rho_{t,\alpha}$',
                        hpd_label=r'$\mathrm{HPD}$', rasterized=True):
    """Color an exact unit sphere with supplied invariant-density values.

    The radius stays one everywhere.  Logarithmic colors are explicit in the
    colorbar generated by the caller.  ``mode`` and region geometry refer to
    real lines; do not use their antipodal cap semantics for a Bloch sphere.
    """
    if not hasattr(ax, 'plot_surface') or not isinstance(mesh, SphereMesh):
        raise ValueError('Use a Matplotlib 3D axis and a SphereMesh')
    log_density = np.asarray(log_density, float)
    if log_density.shape != mesh.shape or not np.isfinite(log_density).all():
        raise ValueError('One finite log density is required at every display node')
    norm = shared_density_norm([log_density]) if norm is None else norm
    R = _rotation(rotation)
    camera = np.array([np.cos(np.deg2rad(elev))*np.cos(np.deg2rad(azim)),
                       np.cos(np.deg2rad(elev))*np.sin(np.deg2rad(azim)),
                       np.sin(np.deg2rad(elev))])
    # Collection-level depth sorting can hide foreground contour lines behind
    # the whole sphere.  Explicitly cull hidden line vertices, then layer the
    # visible contours over the opaque unit-sphere surface.
    ax.computed_zorder = False
    ax.view_init(elev=elev, azim=azim)
    X = mesh.nodes @ R.T
    density = np.exp(np.clip(log_density, -745., 709.))
    surface = ax.plot_surface(X[..., 0], X[..., 1], X[..., 2],
        facecolors=cmap(norm(density)), rcount=mesh.shape[0], ccount=mesh.shape[1],
        linewidth=0., antialiased=False, shade=False, alpha=1., zorder=1)
    surface.set_rasterized(bool(rasterized))
    phi = np.linspace(0., 2*np.pi, 721)
    circles = [np.column_stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)]),
               np.column_stack([np.cos(phi), np.zeros_like(phi), np.sin(phi)]),
               np.column_stack([np.zeros_like(phi), np.cos(phi), np.sin(phi)])]
    _curve(ax, circles, R, color='#BAC6D9', linewidth=.75, alpha=.48, zorder=2,
           visible_direction=camera)
    if A is not None and hpd_loss is not None:
        _curve(ax, hpd_curves(A, float(hpd_loss)), R, color=HPD_COLOR,
               linewidth=2., label=hpd_label, visible_direction=camera)
    if mode is not None:
        h = np.asarray(mode, float)
        if h.shape != (3,) or not np.isclose(h @ h, 1., atol=1e-10):
            raise ValueError('Real-line mode must be a unit three-vector')
        locations = np.array([h, -h]) @ R.T
        visible = locations @ camera > 0
        ax.scatter(*locations[visible].T, color=ORANGE, s=26, depthshade=False, zorder=15)
        if ball_radius is not None:
            _curve(ax, ball_curves(h, float(ball_radius)), R, color=GREY,
                   linestyle='--', linewidth=1.7, label=ball_label,
                   visible_direction=camera)
    if truth is not None:
        h = np.asarray(truth, float)
        if h.shape != (3,) or not np.isclose(h @ h, 1., atol=1e-10):
            raise ValueError('A true real-line representative must be unit length')
        locations = np.array([h, -h]) @ R.T
        visible = locations @ camera > 0
        ax.scatter(*locations[visible].T, color='#222222', marker='x', s=38,
                   linewidths=1.5, depthshade=False, zorder=16, label=truth_label)
    for j, label in enumerate(axis_labels):
        direction = R[:, j]
        edge = 1.21*direction
        start = 1.025*direction
        ax.plot([start[0], edge[0]], [start[1], edge[1]], [start[2], edge[2]],
                color=GREY, lw=.6, alpha=.65, zorder=13)
        ax.text(*(1.33*direction), label, ha='center', va='center', fontsize=10,
                color=GREY, zorder=14,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.7, pad=.7))
    ax.set(xlim=(-1.30, 1.30), ylim=(-1.30, 1.30), zlim=(-1.30, 1.30))
    ax.set_box_aspect((1., 1., 1.), zoom=1.30)
    ax.set_proj_type('ortho')
    ax.set_axis_off()
    if title:
        ax.set_title(title, pad=2.)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def draw_real_bingham_sphere(ax, A, log_normalizer, *, mesh=None, **kwargs):
    """Convenience entry point for an actual real-line density on RP^2."""
    A = _hermitian(A)
    mesh = bingham_sphere_mesh(A) if mesh is None else mesh
    values, U = np.linalg.eigh(A)
    if values[-1]-values[-2] > 1e-10:
        kwargs.setdefault('mode', U[:, -1].real)
    return draw_density_sphere(ax, mesh,
        real_bingham_log_density(A, log_normalizer, mesh.nodes), A=A, **kwargs)


def draw_contact_normal_sphere(ax, A_plane, plane_log_normalizer, *, mesh=None,
                               **kwargs):
    """Draw q_{-A_t^+} for a tangent-plane posterior, including its gauge."""
    A_plane = _hermitian(A_plane)
    normalizer = float(plane_log_normalizer)-float(np.trace(A_plane).real)
    return draw_real_bingham_sphere(ax, -A_plane, normalizer, mesh=mesh, **kwargs)


def draw_bingham_section(ax, A, frame, log_normalizer, *, mesh=None,
                         relative=True, **kwargs):
    """Draw q_A(U_3 x) on a declared real great sphere through the mode.

    The first frame column must be a unit representative of the ambient modal
    line.  The other columns are orthonormal complement directions.  The sphere
    is not the full state space for N>3 (or for complex N>2).  An HPD contour is
    an intersection of an ambient state region, not a 95%-mass section region.
    If ``relative=True`` colors are q_A(U_3 x)/q_A(mode), not a sphere density.
    """
    A = _hermitian(A)
    E = np.asarray(frame)
    reduced = (E.conj().T @ A @ E).real
    mesh = bingham_sphere_mesh(reduced) if mesh is None else mesh
    logs = restriction_log_density(A, E, log_normalizer, mesh.nodes)
    maximum = float(np.linalg.eigvalsh(A)[-1])
    if not np.isclose(np.vdot(E[:, 0], A @ E[:, 0]).real, maximum,
                      atol=1e-8, rtol=1e-10):
        raise ValueError('The first section direction must be an ambient modal vector')
    if relative:
        logs = logs - (maximum-float(log_normalizer))
    kwargs.setdefault('mode', np.array([1., 0., 0.]))
    return draw_density_sphere(ax, mesh, logs, A=reduced, **kwargs)


def draw_bloch_bingham(ax, A, log_normalizer, *, mesh=None, hpd_loss=None,
                       rotation=None, elev=22., azim=0., **kwargs):
    """Draw the actual complex CP^1 density, whose Bloch contours are axial.

    A Hermitian 2 by 2 Bingham parameter is linear in Bloch coordinates.  This
    is a useful symmetry control, not a display of general tangent anisotropy.
    Antipodal Bloch points are orthogonal complex lines and are not identified.
    With no supplied display rotation, the modal cap is viewed face on so
    foreshortening does not masquerade as tangent anisotropy.  Bloch axes rotate
    consistently; pass a common rotation to compare fixed global orientations.
    """
    A = _hermitian(A)
    mesh = sphere_mesh() if mesh is None else mesh
    logs = bloch_log_density(A, log_normalizer, mesh.nodes)
    values, U = np.linalg.eigh(A)
    h = U[:, -1]
    cross = np.conj(h[0])*h[1]
    mode = np.array([2*cross.real, 2*cross.imag, abs(h[0])**2-abs(h[1])**2])
    camera = np.array([np.cos(np.deg2rad(elev))*np.cos(np.deg2rad(azim)),
                       np.cos(np.deg2rad(elev))*np.sin(np.deg2rad(azim)),
                       np.sin(np.deg2rad(elev))])
    if rotation is None:
        pivot = np.eye(3)[np.argmin(abs(mode))]
        t1 = pivot-mode*(mode @ pivot)
        t1 /= np.linalg.norm(t1)
        basis = np.column_stack([mode, t1, np.cross(mode, t1)])
        horizontal = np.array([-np.sin(np.deg2rad(azim)), np.cos(np.deg2rad(azim)), 0.])
        display = np.column_stack([camera, horizontal, np.cross(camera, horizontal)])
        rotation = display @ basis.T
    kwargs.setdefault('axis_labels', (r'$r_1$', r'$r_2$', r'$r_3$'))
    mapping = draw_density_sphere(ax, mesh, logs, rotation=rotation,
        elev=elev, azim=azim, **kwargs)
    R = _rotation(rotation)
    point = R @ mode
    gap = float(values[-1]-values[0])
    if gap > 1e-10 and point @ camera > 0:
        ax.scatter(*point, color=ORANGE, s=26, depthshade=False, zorder=15)
    if hpd_loss is not None and gap > 0 and 0 < hpd_loss < gap:
        cosine = 1-2*float(hpd_loss)/gap
        pivot = np.eye(3)[np.argmin(abs(mode))]
        e2 = pivot-mode*(mode @ pivot)
        e2 /= np.linalg.norm(e2)
        e3 = np.cross(mode, e2)
        phi = np.linspace(0., 2*np.pi, 1441)
        curve = cosine*mode[:, None] + np.sqrt(1-cosine*cosine)*(
            e2[:, None]*np.cos(phi)+e3[:, None]*np.sin(phi))
        _curve(ax, [curve.T], R, color=HPD_COLOR, linewidth=2.,
               visible_direction=camera)
    return mapping


def _finish(fig, path):
    if not plt.rcParams['text.usetex'] or fig.dpi != 200 or plt.rcParams['savefig.dpi'] != 200:
        raise AssertionError('Geometry figures require LaTeX and 200 dpi')
    if path is not None:
        save(fig, Path(path))
    return fig


def real_bingham_comparison(parameters, log_normalizers, *, titles=None,
                            hpd_losses=None, ball_radii=None, axis_labels=None,
                            path=None, decades=2., figsize=(13., 5.5), rotation=None,
                            colorbar_label=r'$q_A$', rasterized=True):
    """Two real RP^2 density maps with one shared invariant-density colorbar.

    Scientific normalizers and thresholds must be computed and verified by the
    caller.  Titles default to the draft's mathematical distribution notation.
    Surface rasterization keeps PDF size bounded; outlines, axes and TeX text
    remain vectors.  PNG exports use 200 dpi.
    """
    parameters = list(parameters)
    if len(parameters) != 2 or len(log_normalizers) != 2:
        raise ValueError('This figure compares exactly two declared beliefs')
    mesh = sphere_mesh()
    logs = [real_bingham_log_density(A, z, mesh.nodes)
            for A, z in zip(parameters, log_normalizers)]
    norm = shared_density_norm(logs, decades=decades)
    titles = titles or [r'$q_{A^{(1)}}$', r'$q_{A^{(2)}}$']
    hpd_losses = [None, None] if hpd_losses is None else hpd_losses
    ball_radii = [None, None] if ball_radii is None else ball_radii
    axis_labels = axis_labels or (r'$e_1$', r'$e_2$', r'$e_3$')
    fig = plt.figure(figsize=figsize, layout='constrained')
    axes = [fig.add_subplot(1, 2, j+1, projection='3d') for j in range(2)]
    for j, ax in enumerate(axes):
        values, U = np.linalg.eigh(parameters[j])
        mapping = draw_density_sphere(ax, mesh, logs[j], norm=norm, rotation=rotation,
            mode=U[:, -1], A=parameters[j], hpd_loss=hpd_losses[j],
            ball_radius=ball_radii[j], title=titles[j], axis_labels=axis_labels,
            rasterized=rasterized)
    colorbar = fig.colorbar(mapping, ax=axes, shrink=.64, pad=.025, aspect=25,
                           extend='min')
    colorbar.set_label(colorbar_label)
    handles = []
    if any(value is not None for value in hpd_losses):
        handles.append(Line2D([0], [0], color=HPD_COLOR, lw=2., label=r'$\mathrm{HPD}$'))
    if any(value is not None for value in ball_radii):
        handles.append(Line2D([0], [0], color=GREY, ls='--', label=r'$d_c=\rho_{t,\alpha}$'))
    handles.append(Line2D([0], [0], marker='o', color='none', markerfacecolor=ORANGE,
                          markersize=5, label=r'$\widehat{\mathcal X}$'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               bbox_to_anchor=(.47, .015), frameon=False)
    fig.density_geometry = dict(
        state_space='real Gr(1,3), represented antipodally on the unit sphere',
        measure='normalized invariant measure; uniform q_A=1',
        density_color_floor=float(norm.vmin), density_color_maximum=float(norm.vmax),
        density_color_decades=float(decades),
        density_colormap=DENSITY_CMAP.name,
        density_colormap_low='#245FA8', density_colormap_high='#FF94C2',
        color_floor_scope='Colors below the floor saturate; probability mass is not truncated',
        view='Front hemisphere visible; the antipodal rear lobe represents the same real line',
        normalization='Supplied verified log normalizer; no normalization from the display mesh',
        region_scope='Supplied HPD and chordal-ball thresholds; plotting does not calculate their mass',
        sphere_radius=1., figure_dpi=200, text_usetex=True)
    return _finish(fig, path)
