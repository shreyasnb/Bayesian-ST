"""Independent continuous-CDF checks for the draft's anisotropic example."""
import unittest

import numpy as np

from lib.reference import rank_one_regions_gr13


class RankOneReferenceTests(unittest.TestCase):
    def test_appendix_i_thresholds(self):
        result = rank_one_regions_gr13(np.diag([0., -8., -80.]))
        self.assertAlmostEqual(result.radius, .519344, delta=1e-6)
        self.assertAlmostEqual(result.hpd_loss, 3.157974, delta=1e-6)
        self.assertAlmostEqual(result.ball_mass, .95, delta=1e-9)
        self.assertAlmostEqual(result.hpd_mass, .95, delta=1e-9)
        self.assertLess(result.integration_error, 1e-8)

    def test_uniform_and_identity_shift(self):
        uniform = rank_one_regions_gr13(np.zeros((3, 3)))
        self.assertAlmostEqual(uniform.radius, np.sqrt(1-.05**2), delta=1e-10)
        self.assertEqual(uniform.hpd_loss, 0.)
        # A constant-density HPD set has full mass, not exactly .95 mass.
        self.assertEqual(uniform.hpd_mass, 1.)
        A = np.diag([0., -8., -80.])
        original = rank_one_regions_gr13(A)
        shifted = rank_one_regions_gr13(A+7*np.eye(3))
        self.assertAlmostEqual(original.radius, shifted.radius, delta=1e-12)
        self.assertAlmostEqual(original.hpd_loss, shifted.hpd_loss, delta=1e-12)
        self.assertAlmostEqual(shifted.log_normalizer-original.log_normalizer,
                               7., delta=1e-12)

    def test_orthogonal_equivariance_and_repeated_spectrum(self):
        Q = np.linalg.qr(np.random.default_rng(61).standard_normal((3, 3)))[0]
        A = np.diag([0., 0., -8.])
        a = rank_one_regions_gr13(A)
        b = rank_one_regions_gr13(Q@A@Q.T)
        self.assertAlmostEqual(a.radius, b.radius, delta=1e-9)
        self.assertAlmostEqual(a.hpd_loss, b.hpd_loss, delta=1e-9)


if __name__ == '__main__':
    unittest.main()
