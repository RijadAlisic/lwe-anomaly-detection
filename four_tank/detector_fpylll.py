"""Short filtering vectors with fpylll (LLL and BKZ), for the paper's parameters (v = 64, N = 192, q = 16381).

Same kernel lattice as detector.py, but reduced with fpylll, which is orders of magnitude faster than the exact
sympy LLL.  Needs fpylll (and optionally g6k); run it with a Python that has them.
"""
import time
import numpy as np
from fpylll import IntegerMatrix, LLL, BKZ
from fpylll.algorithms.bkz2 import BKZReduction

KERNEL = None


def kernel_basis(P, q):
    global KERNEL
    if KERNEL is None:
        import importlib.util, os
        spec = importlib.util.spec_from_file_location("detector_plain", os.path.join(os.path.dirname(__file__), "_kernel.py"))
        KERNEL = importlib.util.module_from_spec(spec); spec.loader.exec_module(KERNEL)
    return KERNEL.kernel_mod_q(P, q)


def reduce_basis(basis, block_size=0, tours=8):
    """LLL, then BKZ with the given block size (0 = LLL only).  Returns the reduced rows as numpy arrays."""
    A = IntegerMatrix.from_matrix([[int(x) for x in row] for row in basis])
    LLL.reduction(A)
    if block_size > 2:
        bkz = BKZReduction(A)
        for _ in range(tours):
            bkz(BKZ.Param(block_size=block_size, strategies=BKZ.DEFAULT_STRATEGY))
    return np.array([[A[i, j] for j in range(A.ncols)] for i in range(A.nrows)], dtype=np.int64)


def short_vectors(P, q, k=1, block_size=0, tours=8):
    """The k shortest rows of the reduced kernel lattice of P (shape (N, v)) and the time spent."""
    t = time.time()
    R = reduce_basis(kernel_basis(P, q), block_size, tours)
    norms = np.linalg.norm(R.astype(float), axis=1)
    idx = np.argsort(norms)[:k]
    return R[idx], norms[idx], time.time() - t
