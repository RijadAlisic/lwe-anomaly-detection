"""Block size and cost of BKZ for a target filtering-vector length (Section VII of the paper, sweep paragraph).

Model: BKZ with block size kappa reaches the root Hermite factor delta0(kappa) (Chen-Nguyen, valid for kappa >~ 50), so
||d||_2 ~ delta0^N q^(v/N) for N samples; the cost of one call grows as 2^(0.292 kappa).  LLL (delta0 ~ 1.019) is outside the BKZ range.
Parameters of the numerical sweep: v = 64, q = 2^23 + 9, N = 192 samples per key epoch.   Run:  python bkz_cost.py
"""
import numpy as np
from scipy.optimize import brentq

q, v, N = 2 ** 23 + 9, 64, 192
delta0 = lambda k: ((k / (2 * np.pi * np.e)) * (np.pi * k) ** (1 / k)) ** (1 / (2 * (k - 1)))
base = q ** (v / N)
gh = np.sqrt(N / (2 * np.pi * np.e)) * base
print(f"q^(v/N) = {base:.1f};  Gaussian heuristic of the kernel lattice: {gh:.0f}")
print("LLL (delta0 = 1.019):  ||d|| ~ %.1e" % (base * 1.019 ** N))
print(f"{'||d||':>8} {'delta0':>8} {'kappa':>7} {'cost 2^':>8}")
for target in (2000, 1500, 1400, 1000, 800, 680):
    d = (target / base) ** (1 / N)
    k = brentq(lambda k: delta0(k) - d, 50, 2000)
    print(f"{target:8d} {d:8.4f} {k:7.0f} {0.292 * k:8.1f}")
print("kappa = 80 gives ||d|| ~ %.2e at cost 2^%.1f" % (base * delta0(80) ** N, 0.292 * 80))
