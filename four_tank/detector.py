"""Detector: short filtering vectors in the q-ary kernel lattice and the two-sided test (Theorem 1).

Given N encrypted residual samples (P_k, b_k) with P_k in Z_q^v, a filtering vector d in Z^N with
d^T P = 0 (mod q) removes the secret:  d^T b = r d^T rho + d^T e (mod q).
We find d by building a basis of {d : d^T P = 0 mod q} and reducing it with LLL (exact, via sympy).
For the paper's parameters (v = 64) the same step is done with BKZ (g6k / fpylll) instead of LLL.
"""
import numpy as np
from sympy import ZZ
from sympy.polys.matrices import DomainMatrix
from scipy.stats import norm


from _kernel import kernel_mod_q as _kernel_mod_q


def short_filtering_vector(P, q):
    """Shortest vector of the LLL-reduced q-ary kernel lattice (centered coefficients)."""
    B = DomainMatrix([[ZZ(x) for x in row] for row in _kernel_mod_q(P, q)], (P.shape[0],) * 2, ZZ)
    R = B.lll().to_Matrix().tolist()
    vecs = [np.array([int(x) for x in row], dtype=object) for row in R]
    d = min(vecs, key=lambda w: sum(int(x) ** 2 for x in w))
    return np.array([int(x) for x in d], dtype=np.int64)


def centered(x, q):
    x = np.asarray(x, dtype=object) % q
    return np.array([int(t) - q if int(t) > q // 2 else int(t) for t in x], dtype=object)


def two_sided_test(d, b, q, s2, alpha):
    """Theorem 1: X = d^T b mod q.  Reject H0 when |X| >= gamma, where gamma is the (1 - alpha/2)-quantile
    of N(0, s2) and s2 = d^T Sigma_rho d."""
    X = int(centered([int(np.dot(d.astype(object), np.asarray(b, dtype=object))) % q], q)[0])
    gamma = norm.ppf(1 - alpha / 2) * np.sqrt(s2)
    return X, gamma, abs(X) >= gamma
