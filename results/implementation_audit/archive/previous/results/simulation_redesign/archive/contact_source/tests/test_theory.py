"""Independent checks of information, motion transport and regional duality."""
import unittest
import numpy as np
from scipy.linalg import null_space, expm
from lib.theory import fisher_information, tracking_bound, regional_bound


class TheoryTests(unittest.TestCase):
    def test_fisher_matches_finite_difference_mean_jacobian(self):
        rng = np.random.default_rng(430)
        X = np.linalg.qr(rng.normal(size=(5, 2)))[0]
        Q = null_space(X.T)
        V = rng.normal(size=(5, 8))
        s = .37
        columns = []
        h = 1e-6
        for i in range(2):
            for a in range(3):
                Z = np.zeros((3, 2))
                Z[a, i] = h
                plus = np.linalg.qr(X+Q@Z)[0]
                minus = np.linalg.qr(X-Q@Z)[0]
                derivative = ((plus@plus.T-minus@minus.T)@V)/(2*h)
                columns.append(derivative.ravel())
        J = np.column_stack(columns)
        np.testing.assert_allclose(fisher_information(X, V, s), J.T@J/s, rtol=2e-8, atol=1e-7)

    def test_global_ambiguity_and_worst_orientation(self):
        n, k = 5, 2
        V = np.eye(n)[:, :n-2]
        X = np.eye(n)[:, [0, 3]]
        Y = np.eye(n)[:, [0, 4]]
        np.testing.assert_allclose(X@X.T@V, Y@Y.T@V)
        self.assertGreater(np.linalg.norm(X@X.T-Y@Y.T), 1)
        # The two least excited directions lie on opposite sides of X.
        C = np.diag([9., 6., 3., 1., 0.])
        X = np.eye(n)[:, [0, 4]]
        F = fisher_information(X, np.diag(np.sqrt(np.diag(C))), .5)
        self.assertAlmostEqual(np.linalg.eigvalsh(F)[0], 2.)

    def test_exact_motion_zero_drift_and_direct_C(self):
        rng = np.random.default_rng(9)
        n, k, t = 4, 2, 7
        G = []
        Xs = []
        X = np.eye(n)[:, :k]
        for _ in range(t):
            A = rng.normal(size=(n,n))*.06
            g = expm(A-A.T)
            G.append(g)
            X = g@X
            Xs.append(X)
        V = rng.normal(size=(t,n))
        b = tracking_bound(Xs, V, .02, beta=.93, G=G, A0=np.diag([4.,1.,-1.,-2.]))
        self.assertLess(b.drift, 1e-10)
        direct = sum(.93**(t-1-j)*np.outer(b.transports[j]@V[j], b.transports[j]@V[j]) for j in range(t))
        np.testing.assert_allclose(b.C, direct)
        self.assertAlmostEqual(b.initialization, .93**t*3.)

    def test_fixed_time_bound_becomes_nonvacuous(self):
        # Thousands of preassigned equal-norm observations give a large gap.
        V = np.tile(np.eye(3), (1000,1))
        Xs = np.tile(np.array([[1.],[0.],[0.]])[None], (len(V),1,1))
        b = tracking_bound(Xs, V, .01)
        self.assertTrue(b.valid)
        self.assertLess(b.bound, b.ceiling)
        self.assertAlmostEqual(b.gamma, 100000.)
        b2 = tracking_bound(Xs[:1], V[:1], 1.)
        self.assertFalse(b2.valid)
        self.assertTrue(np.isnan(b2.bound))

    def test_dual_tie_requires_rotation(self):
        # B+mu*C is identically zero at mu=1, and an arbitrary leading
        # eigenvector violates the active constraint or misses the optimum.
        B = np.diag([0., 1.])
        C = np.diag([1., 0.])
        result = regional_bound(B,C,.3,1)
        self.assertTrue(result.feasible)
        self.assertAlmostEqual(result.constraint,.3,places=8)
        self.assertAlmostEqual(result.lower,.7,places=8)
        self.assertLess(result.gap,1e-7)

    def test_complex_dual_against_dense_rank_one_search(self):
        B = np.array([[1.,.7+.4j],[.7-.4j,-.2]])
        C = np.diag([0.,1.])
        result = regional_bound(B,C,.6,1)
        theta = np.linspace(0,np.pi/2,400)
        phi = np.linspace(0,2*np.pi,500)
        u = np.stack(np.broadcast_arrays(np.cos(theta)[:,None], np.sin(theta)[:,None]*np.exp(1j*phi)[None]),axis=-1)
        task = np.einsum('...i,ij,...j->...',u.conj(),B,u).real
        allowed = np.abs(u[...,1])**2 >= .6
        grid = np.max(task[allowed])
        self.assertTrue(result.feasible)
        self.assertGreaterEqual(result.upper+1e-10, grid)
        self.assertLess(result.gap,1e-6)
        self.assertLess(result.lower-grid,.01)

    def test_region_endpoints_and_scalar_C(self):
        B = np.diag([2.,7.,3.])
        C = np.diag([1.,1.,0.])
        result = regional_bound(B,C,2.,2)
        self.assertTrue(result.feasible)
        self.assertAlmostEqual(result.lower,9.)
        self.assertLess(result.gap,1e-10)
        result = regional_bound(B,np.eye(3),1.,2)
        self.assertAlmostEqual(result.lower,10.)
        with self.assertRaises(ValueError):
            regional_bound(B,np.eye(3),3.,2)

    def test_random_real_and_complex_regional_certificates(self):
        rng = np.random.default_rng(99)
        for complex_data in (False, True):
            for n in range(2, 8):
                for k in range(1, n):
                    B = rng.normal(size=(n,n))
                    C = rng.normal(size=(n,n))
                    if complex_data:
                        B = B+1j*rng.normal(size=(n,n))
                        C = C+1j*rng.normal(size=(n,n))
                    B = (B+B.conj().T)/2
                    C = (C+C.conj().T)/2
                    e = np.linalg.eigvalsh(C)
                    threshold = .8*np.sum(e[-k:])+.2*np.sum(e[:k])
                    result = regional_bound(B,C,threshold,k)
                    self.assertTrue(result.feasible)
                    self.assertLess(result.gap, 1e-5)
                    np.testing.assert_allclose(result.basis.conj().T@result.basis,np.eye(k),atol=1e-10)


if __name__ == '__main__':
    unittest.main()
