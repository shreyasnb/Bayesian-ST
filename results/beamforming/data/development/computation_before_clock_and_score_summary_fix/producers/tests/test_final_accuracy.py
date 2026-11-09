import numpy as np
from scipy.special import logsumexp
from lib.final_accuracy import complex_reference, complex_response_second_moment
from lib.guide_information import independent_moments


def test_uniform_and_repeated_eigenvalues_have_dirichlet_moments():
    for n in (2, 3, 8):
        ref = complex_reference(np.full(n, 7.), full_moments=True)
        assert ref['numerical_ok']
        assert abs(ref['log_normalizer']-7) < 1e-12
        assert np.allclose(ref['selected_mean'], 1/n)
        expected = (np.ones((n, n))+np.eye(n))/(n*(n+1))
        assert np.allclose(ref['second_squared_coordinates'], expected)
    v = np.array([.3+.2j, -.5j, .7])
    second, _ = complex_response_second_moment(np.zeros((3, 3)), v)
    assert np.allclose(second, (np.vdot(v, v)*np.eye(3)+np.outer(v, v.conj()))/12)


def test_divided_differences_match_independent_continuous_simplex_quadrature():
    A = np.diag([-13., -1.7, .4])
    mean, logz = independent_moments(A, complex_data=True, order=64)
    ref = complex_reference(np.diag(A), full_moments=True)
    assert ref['numerical_ok']
    assert abs(ref['log_normalizer']-logz) < 1e-11
    assert np.allclose(ref['selected_mean'], np.diag(mean), atol=1e-11)
    v = np.array([.2+.5j, -.3j, .8])
    actual, _ = complex_response_second_moment(A, v)
    # Independently integrate squared radii over the simplex and both free
    # phases. This checks the complete response matrix, including off-diagonals.
    g, w = np.polynomial.legendre.leggauss(30)
    x, y = np.meshgrid((g+1)/2, (g+1)/2, indexing='ij')
    squared = np.stack([x, (1-x)*y, (1-x)*(1-y)], axis=-1).reshape(-1, 3)
    prior = (w[:, None]*w[None, :]/4*2*(1-x)).ravel()
    logits = squared@np.diag(A)+np.log(prior)
    weights = np.exp(logits-logsumexp(logits))
    phases = np.arange(7)*2*np.pi/7
    reference = np.zeros((3, 3), complex)
    for a in phases:
        for b in phases:
            h = np.sqrt(squared)*np.exp(1j*np.array([0., a, b]))
            response = h*(h.conj()@v)[:, None]
            reference += np.einsum('t,ti,tj->ij', weights, response, response.conj())/49
    assert np.allclose(actual, reference, atol=2e-10)


def test_reference_selected_moments_respect_unsorted_input_and_trace_shifts():
    values = np.array([2., -8., .1, -3.])
    first = complex_reference(values, full_moments=True)
    shifted = complex_reference(values+19., full_moments=True)
    assert first['numerical_ok'] and shifted['numerical_ok']
    assert abs(shifted['log_normalizer']-first['log_normalizer']-19) < 1e-12
    assert np.allclose(shifted['selected_mean'], first['selected_mean'])
    assert abs(first['selected_mean'].sum()-1) < 1e-12
