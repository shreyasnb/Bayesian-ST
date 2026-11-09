"""Independent high-precision rank-one complex integral checks.

The squared eigen-coordinates of a uniform complex line are Dirichlet(1,...,1).
Its exponential integral is (n-1)! times a confluent divided difference of exp.
This reference is independent of the native positive-series recurrence. It
does not certify eigensolver rounding, arbitrary-rank integration, or tails.
"""
import math
import numpy as np
import mpmath as mp


def _cluster(values):
    values = np.sort(np.asarray(values, float))
    tolerance = 32*np.finfo(float).eps*max(1., float(np.max(abs(values))))
    groups, start = [], 0
    for index in range(1, len(values)+1):
        if index == len(values) or values[index]-values[start] > tolerance:
            groups.append((start, index))
            start = index
    clustered = values.copy()
    for start, end in groups:
        clustered[start:end] = values[start:end].mean()
    return clustered, float(np.max(abs(clustered-values)))


def _exp_divided_difference(nodes):
    nodes = sorted(nodes)
    values = [mp.exp(x) for x in nodes]
    for order in range(1, len(nodes)):
        updated = []
        for index in range(len(nodes)-order):
            if nodes[index+order] == nodes[index]:
                value = mp.exp(nodes[index])/mp.factorial(order)
            else:
                value = (values[index+1]-values[index])/(nodes[index+order]-nodes[index])
            updated.append(value)
        values = updated
    return values[0]


def _reference(values, digits, indices, second):
    with mp.workdps(digits):
        shift = float(np.max(values))
        nodes = [mp.mpf(float(x))-mp.mpf(shift) for x in values]
        z = mp.factorial(len(nodes)-1)*_exp_divided_difference(nodes)
        if z <= 0 or not mp.isfinite(z):
            raise ArithmeticError('High-precision exponential divided difference unresolved')
        logz = mp.log(z)+mp.mpf(shift)
        means = {i: mp.factorial(len(nodes)-1)*_exp_divided_difference(nodes+[nodes[i]])/z
                 for i in indices}
        pairs = {}
        if second:
            for i in indices:
                for j in indices:
                    pairs[i, j] = ((2 if i == j else 1)*mp.factorial(len(nodes)-1)
                        *_exp_divided_difference(nodes+[nodes[i], nodes[j]])/z)
        return logz, means, pairs


def complex_reference(eigenvalues, *, digits=(100, 200), full_moments=False,
                      tolerance=1e-10):
    """Independent log normalizer, selected means and optional second moments.

    Near-equal double eigenvalues are made exactly confluent. The maximum
    displacement bounds the log-integral change for a rank-one projector.
    Cross-precision agreement is measured, not asserted as a rounding theorem.
    """
    original = np.asarray(eigenvalues, float)
    if (original.ndim != 1 or len(original) < 2 or not np.isfinite(original).all()
            or len(digits) != 2 or not 10 <= digits[0] < digits[1]):
        raise ValueError('Finite eigenvalues and increasing decimal precision are required')
    order = np.argsort(original)
    values, cluster_bound = _cluster(original)
    indices = (list(range(len(values))) if full_moments
               else sorted(set([0, len(values)//2, len(values)-1])))
    low = _reference(values, digits[0], indices, full_moments)
    high = _reference(values, digits[1], indices, full_moments)
    with mp.workdps(digits[1]):
        error = abs(low[0]-high[0])
        for index in indices:
            error = max(error, abs(low[1][index]-high[1][index]))
        for pair in high[2]:
            error = max(error, abs(low[2][pair]-high[2][pair]))
        error = float(error)
    inverse = np.argsort(order)
    means_sorted = np.full(len(values), np.nan)
    for index, value in high[1].items():
        means_sorted[index] = float(value)
    result = dict(log_normalizer=float(high[0]), selected_mean=means_sorted[inverse],
        selected_indices=order[np.array(indices)], cross_precision_error=error,
        cluster_logz_perturbation_bound=cluster_bound, digits=list(digits),
        numerical_ok=bool(error+cluster_bound < tolerance),
        scope='Confluent high-precision Dirichlet integral; double-input eigenvalues; '
              'cross-precision agreement is not a rigorous floating-point enclosure.')
    if full_moments:
        second_sorted = np.empty((len(values), len(values)))
        for (i, j), value in high[2].items():
            second_sorted[i, j] = float(value)
        result['second_squared_coordinates'] = second_sorted[np.ix_(inverse, inverse)]
    return result


def complex_response_second_moment(A, v, *, digits=(100, 200), tolerance=1e-10):
    """E[P vv^H P] for a complex line, independently of a native mean backend."""
    eigenvalues, Q = np.linalg.eigh(A)
    reference = complex_reference(eigenvalues, digits=digits, full_moments=True,
                                  tolerance=tolerance)
    C = reference['second_squared_coordinates']
    coordinates = Q.conj().T@v
    result = C*np.outer(coordinates, coordinates.conj())
    np.fill_diagonal(result, C@abs(coordinates)**2)
    return Q@result@Q.conj().T, reference
