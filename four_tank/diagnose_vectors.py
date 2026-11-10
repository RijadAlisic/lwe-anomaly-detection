"""Look at the filtering vectors of one run: their norm, support, and how large the signal r*d.rho is next to q/2.
Usage:  OMP_NUM_THREADS=1 python diagnose_vectors.py [SIGMA] [L]      (random-sign attack of length L in epoch 2)"""
import sys
import numpy as np
import encrypted
from quantized import QuantizedLinear
from _kernel import kernel_mod_q
from blaster import reduce as blaster_lll

T = 192
sg = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
L = int(sys.argv[2]) if len(sys.argv) > 2 else 192
spec = QuantizedLinear()
spec.lwe = dict(spec.lwe, sigma=sg, r=int(round(4096 * sg / 50)))
q, r = spec.lwe["q"], spec.lwe["r"]
rng = np.random.default_rng(1000 + 3)
sgn = {2 * T + i: int(rng.choice([-1, 1])) for i in range(L)}
out = encrypted.run(spec, 3 * T, sensor_attack=lambda k: sgn.get(k, 0), seed=3, keep_ciphertexts=True)
res = out["res_ct"]
rho = np.rint(out["rho"] * spec.cs).astype(int)
Tmap = encrypted.noise_map(spec, T)
cen = lambda x: ((x + q // 2) % q) - q // 2
for w in (1, 2):
    seg = res[w * T:(w + 1) * T]
    P, b, rh = seg[:, :-1], seg[:, -1], rho[w * T:(w + 1) * T]
    B = np.array(kernel_mod_q(P, q), dtype=np.int64)
    R = blaster_lll(B.T, delta=0.99999, cores=1, use_seysen=False, depth=0)[1].T
    D = R[np.argsort(np.linalg.norm(R.astype(float), axis=1))[:6]]
    S = sg ** 2 * (D.astype(float) @ (Tmap @ Tmap.T) @ D.astype(float).T)
    print(f"\nepoch {w}: nonzero residual samples {np.count_nonzero(rh)}, q/(2 sigma) = {q / 2 / sg:.0f}")
    X = []
    for i in range(6):
        d = D[i]
        sig = int(d.astype(object) @ rh.astype(object))
        x = cen(int(d.astype(object) @ b.astype(object)))
        X.append(x)
        print(f"  vec{i}: |d|={np.linalg.norm(d.astype(float)):7.1f} entries!=0 {np.count_nonzero(d):3d}  r*d.rho={r * sig:10d} (q/2={q // 2})  X={x:9d}  s_ii={np.sqrt(S[i, i]):8.1f}")
    X = np.array(X, float)
    print("  Z =", round(float(X @ np.linalg.solve(S, X)), 1))
