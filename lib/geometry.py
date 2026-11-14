"""Basis-independent real and complex Grassmann operations."""
import numpy as np


def hermitian(A):
    A = np.asarray(A)
    return (A + A.conj().T) / 2


def center(A):
    A = hermitian(A)
    return A - np.trace(A).real / len(A) * np.eye(len(A))
