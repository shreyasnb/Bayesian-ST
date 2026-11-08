"""Independent deterministic and stochastic checks of moment backends."""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from lib.moments import (quadrature_gr13, quadrature_circle, sample_bingham,
                         chain_diagnostics, weighted_quantile, mean_bingham)


class MomentTests(unittest.TestCase):
    def test_paper_quadrature_and_accuracy(self):
        A = np.diag([0., -8., -80.])
        q = quadrature_gr13(A, 96)
        fine = quadrature_gr13(A, 144)
        assert_allclose(q.mean, fine.mean, atol=1e-10)
        assert_allclose(q.mean.diagonal(), [.92541154, .06829456, .00629390], atol=5e-9)
        self.assertAlmostEqual(q.weights.sum(), 1)

    def test_normalizer_derivative_identity_shift_and_equivariance(self):
        rng = np.random.default_rng(1)
        Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        A = np.diag([0., -3., -13.])
        B = np.array([[.2, .4, 0], [.4, -.1, .3], [0, .3, .5]])
        h = 1e-4
        q = quadrature_gr13(A)
        derivative = (quadrature_gr13(A+h*B).log_normalizer -
                      quadrature_gr13(A-h*B).log_normalizer) / (2*h)
        self.assertAlmostEqual(derivative, np.trace(q.mean@B), places=8)
        assert_allclose(quadrature_gr13(A+5*np.eye(3)).mean, q.mean, atol=1e-13)
        self.assertAlmostEqual(quadrature_gr13(A+5*np.eye(3)).log_normalizer,
                               q.log_normalizer+5, places=12)
        assert_allclose(quadrature_gr13(Q@A@Q.T).mean, Q@q.mean@Q.T, atol=1e-12)

    def test_circle_bessel_normalizer(self):
        from scipy.special import i0e
        # Uniform circle integral exp(a cos^2 theta+b sin^2 theta).
        a, b = 3., -7.
        q = quadrature_circle(np.diag([a, b]))
        exact = (a+b)/2 + abs(a-b)/2 + np.log(i0e((a-b)/2))
        self.assertAlmostEqual(q.log_normalizer, exact, places=12)

    def test_real_sampler_against_quadrature_and_chains(self):
        A = np.diag([0., -8., -80.])
        q = quadrature_gr13(A)
        results = [sample_bingham(A, 1, seed, draws=24000, burnin=3000)
                   for seed in (102, 203)]
        for result in results:
            error = max(abs((result.mean-q.mean).diagonal()))
            self.assertLess(error, max(.004, 5*result.mcse))
            self.assertGreater(result.ess, 100)
            self.assertGreater(result.acceptance, .1)
            assert_allclose(np.einsum('bni,bnj->bij', result.bases, result.bases),
                            np.ones((result.draws, 1, 1)), atol=1e-10)
        self.assertLess(chain_diagnostics(results)['max_rhat'], 1.1)

    def test_complex_uniform_and_hermitian_moments(self):
        result = sample_bingham(np.zeros((5, 5)), 2, 71, draws=4000,
                                burnin=500, complex_data=True)
        assert_allclose(result.mean, 2*np.eye(5)/5, atol=1e-12)
        assert_allclose(result.raw_mean, 2*np.eye(5)/5, atol=.04)
        assert_allclose(np.einsum('bni,bnj->bij', result.bases.conj(), result.bases),
                        np.broadcast_to(np.eye(2), (4000, 2, 2)), atol=1e-10)
        assert_allclose(mean_bingham(7*np.eye(5), 2), 2*np.eye(5)/5)

    def test_high_concentration_independent_chains(self):
        # n=16,k=3 includes spectral gaps up to 600, beyond the moment series
        # regime. Check independent chains and the leading Laplace limit;
        # agreement at this finite budget is not a global mixing guarantee.
        spectrum = np.r_[np.linspace(0, 180, 13), 400, 500, 600]
        for complex_data in (False, True):
            with self.subTest(complex_data=complex_data):
                results = [sample_bingham(np.diag(spectrum), 3, seed,
                           draws=12000, burnin=4000, complex_data=complex_data)
                           for seed in (51, 52, 53)]
                diagnostics = chain_diagnostics(results)
                self.assertLess(diagnostics['max_rhat'], 1.05)
                self.assertLess(diagnostics['max_standardized_difference'], 4)
                self.assertGreater(diagnostics['min_ess'], 40)
                factor = 1 if complex_data else 2
                tangent = 1 / (factor * (spectrum[None, 13:] - spectrum[:13, None]))
                approximation = np.r_[tangent.sum(axis=1), 1-tangent.sum(axis=0)]
                for result in results:
                    self.assertLess(max(abs(result.mean.diagonal()-approximation)),
                                    .002 + 4*result.mcse)
                    self.assertGreater(result.acceptance, .15)

    def test_input_and_weighted_quantile(self):
        self.assertEqual(weighted_quantile([1., 3., 2.], .5, [.2, .2, .6]), 2.)
        with self.assertRaises(ValueError):
            sample_bingham(np.eye(3), 3)
        with self.assertRaises(ValueError):
            quadrature_gr13(np.ones((2, 3)))


if __name__ == '__main__':
    unittest.main()
