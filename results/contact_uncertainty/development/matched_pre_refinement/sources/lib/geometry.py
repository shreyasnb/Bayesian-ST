"""Basis-independent real and complex Grassmann operations."""
import numpy as np


def hermitian(A):
    A = np.asarray(A)
    return (A + A.conj().T) / 2


def center(A):
    A = hermitian(A)
    return A - np.trace(A).real / len(A) * np.eye(len(A))


def projector(X):
    return X @ X.conj().T


def orth(A):
    return np.linalg.qr(A)[0]


def mode(A, k):
    return np.linalg.eigh(hermitian(A))[1][:, -k:]


def chordal(X, Y):
    return np.sqrt(max(0., X.shape[1] - np.linalg.norm(X.conj().T @ Y, 'fro') ** 2))


def gaussian(rng, shape, complex_data=False):
    z = rng.normal(size=shape)
    return (z + 1j * rng.normal(size=shape)) / np.sqrt(2) if complex_data else z


def risk(M, v):
    return float(np.real(v.conj() @ (M - M @ M) @ v))
