"""Rough bit security of the demo parameters against the two standard lattice attacks, sieving cost model, classical.

LWE with uniform secret in Z_q^n, N <= 192 samples (one key epoch), fresh noise sigma.  No Lattice Estimator (needs Sage) is used,
so these are the textbook estimates, good to a few bits and not a security claim:
  primal (Alkim et al. 2016): BKZ-beta solves uSVP on the (m+n+1)-dim embedding if  sigma sqrt(beta) <= delta0^(2 beta - d - 1) q^(m/d)
  dual   (distinguishing):    short w with w^T A = 0, |w| = delta0^m q^(n/m), advantage eps = exp(-2 pi^2 (sigma |w| / q)^2),
                              cost of BKZ-beta / eps^2 repetitions
delta0(beta) from Chen-Nguyen (beta >~ 50; smaller beta is an extrapolation).   Cost per BKZ call: 2^(0.292 beta + 16.4 + log2(8 d)) (the sieving model of the Lattice Estimator); beta is searched from 40 up.
Usage: python lwe_security.py
"""
import numpy as np

n = 64
d0 = lambda b: ((b / (2 * np.pi * np.e)) * (np.pi * b) ** (1 / b)) ** (1 / (2 * (b - 1)))


def primal(q, sig, M):
    best = None
    for m in range(40, M + 1, 4):
        d = m + n + 1
        for b in range(40, 400):
            if sig * np.sqrt(b) <= d0(b) ** (2 * b - d - 1) * q ** (m / d):
                bits = 0.292 * b + 16.4 + np.log2(8 * d)
                if best is None or bits < best[2]:
                    best = (b, m, bits)
                break
    return best


def dual(q, sig, M):
    best = None
    for m in range(40, M + 1, 4):
        for b in range(40, 400):
            ell = d0(b) ** m * q ** (n / m)
            eps = np.exp(-2 * np.pi ** 2 * (sig * ell / q) ** 2)
            bits = 0.292 * b + 16.4 + np.log2(8 * m) + 2 * np.log2(1 / max(eps, 1e-300))
            if best is None or bits < best[0]:
                best = (bits, b, m, eps)
    return best


print(f"n = {n}, uniform secret; bits = BKZ cost model above; the dual bits include the 1/eps^2 repetitions")
print(f"{'q':>10} {'sigma':>6} {'samples':>7} | {'primal beta':>11} {'bits':>5} | {'dual beta':>9} {'bits':>5} {'|w|':>7} {'eps':>7}")
for q, sig, label in ((16381, 26, "paper"), (2 ** 23 + 9, 50, "COMPACT"), (2 ** 23 + 9, 100, ""), (2 ** 23 + 9, 200, ""),
                      (2 ** 23 + 9, 300, ""), (2 ** 23 + 9, 400, ""), (2 ** 23 + 9, 600, "")):
    for M in (128, 192):
        p = primal(q, sig, M); du = dual(q, sig, M)
        ell = d0(du[1]) ** du[2] * q ** (n / du[2])
        pb = f"{p[0]:11d} {p[2]:5.0f}" if p else f"{'>400':>11} {'':>5}"
        print(f"{q:10d} {sig:6g} {M:7d} | {pb} | {du[1]:9d} {du[0]:5.0f} {ell:7.0f} {du[3]:7.2f}   {label if M == 128 else ''}")
