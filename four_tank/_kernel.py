"""Kernel lattice basis mod q (pure Python)."""


def kernel_mod_q(P, q):
    """Basis (rows) of {d in Z_q^N : d^T P = 0 mod q}, q prime.  P has shape (N, v)."""
    N, v = P.shape
    M = [[int(x) % q for x in row] for row in P.T]          # v x N
    piv, r = [], 0
    for c in range(N):
        if r == v:
            break
        p = next((i for i in range(r, v) if M[i][c] % q), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = pow(M[r][c], -1, q)
        M[r] = [(x * inv) % q for x in M[r]]
        for i in range(v):
            if i != r and M[i][c]:
                f = M[i][c]
                M[i] = [(a - f * b) % q for a, b in zip(M[i], M[r])]
        piv.append(c)
        r += 1
    free = [c for c in range(N) if c not in piv]
    basis = []
    for j in free:                                           # d_j = 1, pivots solved
        d = [0] * N
        d[j] = 1
        for i, pc in enumerate(piv):
            d[pc] = (-M[i][j]) % q
        basis.append(d)
    for pc in piv:                                           # q * e_pivot
        d = [0] * N
        d[pc] = q
        basis.append(d)
    return basis
