"""Adaptive reference integration for real rank-one credible regions in R^3.

The discrete nodes used to compute moments are not an accurate representation
of a continuous tail CDF at a fixed quadrature order. This independent helper
integrates polar caps directly and solves the continuous CDF equations from
Appendix I. It is a low-dimensional validation reference, not an array backend.
"""
from dataclasses import dataclass
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq
from scipy.special import dawsn


@dataclass(frozen=True)
class RankOneRegions:
    """Equal-mass radius and density-loss threshold, with reference CDF checks.

    ``integration_error`` is an estimated absolute error in normalized mass
    from QUADPACK. It is not a rigorous interval bound on its quadrature error.
    """
    radius: float
    hpd_loss: float
    ball_mass: float
    hpd_mass: float
    log_normalizer: float
    integration_error: float
    target_mass: float


def _polar_integral(g, lower):
    """Integral from lower to 1 of exp(-g*(1-q**2)) dq, g >= 0."""
    if lower >= 1:
        return 0.
    if g < 1e-5:
        # Three terms avoid subtractive cancellation at g=0. Remainder is
        # bounded by g**3 * (1-lower) / 6 on this branch.
        l = lower
        first = (1-l) - (1-l**3)/3
        second = (1-l) - 2*(1-l**3)/3 + (1-l**5)/5
        return (1-l) - g*first + g*g*second/2
    root = np.sqrt(g)
    value = (dawsn(root) - np.exp(-g*(1-lower*lower))*dawsn(root*lower))/root
    return max(0., float(value))


def rank_one_regions_gr13(A, mass=.95, epsabs=1e-11):
    """Adaptive equal-mass ball/HPD thresholds for a real 3x3 Bingham law.

    Integrates q=cos(theta) analytically with a stable Dawson expression and
    performs adaptive angular integration; root solving therefore uses a
    continuous CDF rather than weighted discrete quantiles. Both regions are
    centered on a leading eigenvector. Identity shifts are restored in the
    returned log normalizer. Repeated and uniform spectra are supported.
    """
    A = np.asarray(A)
    if (A.shape != (3, 3) or np.iscomplexobj(A) or
            not np.all(np.isfinite(A)) or
            not np.allclose(A, A.T, atol=1e-12)):
        raise ValueError('A must be a finite real symmetric 3x3 matrix')
    if not 0 < mass < 1 or not 0 < epsabs < 1:
        raise ValueError('require 0 < mass < 1 and 0 < epsabs < 1')
    lam = np.linalg.eigvalsh(A)[::-1]
    a, b = float(lam[0]-lam[1]), float(lam[0]-lam[2])
    errors = []

    def angular(lower, h=None):
        points = []
        if h is not None and a < h < b and b > a:
            points = [float(np.arcsin(np.sqrt((h-a)/(b-a))))]

        def integrand(phi):
            g = a*np.cos(phi)**2 + b*np.sin(phi)**2
            q0 = lower
            if h is not None:
                q0 = np.sqrt(max(0., 1-h/g)) if g > 0 else 0.
            return _polar_integral(g, q0)

        with warnings.catch_warnings():
            # Do not silently return apparently precise thresholds following
            # a failed adaptive integration.
            warnings.simplefilter('error', IntegrationWarning)
            value, error = quad(integrand, 0., np.pi/2, points=points,
                                epsabs=epsabs, epsrel=epsabs, limit=250)
        errors.append(error)
        return value

    normalizer = angular(0.)
    if normalizer <= 0:
        raise ArithmeticError('numerical normalizer is nonpositive')

    def ball(rho):
        return angular(np.sqrt(max(0., 1-rho*rho)))/normalizer

    def hpd(h):
        if b == 0:
            return 1.
        return angular(0., h=h)/normalizer

    radius = brentq(lambda rho: ball(rho)-mass, 0., 1., xtol=1e-12)
    threshold = (0. if b == 0 else
                 brentq(lambda h: hpd(h)-mass, 0., b, xtol=1e-11))
    achieved_ball, achieved_hpd = ball(radius), hpd(threshold)
    return RankOneRegions(
        float(radius), float(threshold), float(achieved_ball), float(achieved_hpd),
        float(lam[0]+np.log(2*normalizer/np.pi)),
        float(2*max(errors, default=0.)/normalizer), float(mass))
