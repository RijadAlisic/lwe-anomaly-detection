"""Why is Z flat in the number of nonzero residual samples?  Test of the wrap-around explanation.
For one key and several attack lengths L (random-sign attack, first L samples of epoch 2) take the six LLL vectors d_i of the
attacked epoch and split  X_i = cen( r d_i.rho + N_i ),  N_i = d_i^T e  (the noise part, = cen(X_i - r d_i.rho)).
Then recompute Z with the signal scaled by f:  X_i(f) = cen( f * r d_i.rho + N_i ).   f = 1 is the real detector.
If the flatness is caused by wrap-around, Z(f) must grow with L for small f.
Usage: OMP_NUM_THREADS=1 python signal_scaling_check.py [SIGMA]"""
import sys
import numpy as np
import encrypted
from quantized import QuantizedLinear
from _kernel import kernel_mod_q
from blaster import reduce as blaster_lll

T = 192
sg = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
spec = QuantizedLinear()
spec.lwe = dict(spec.lwe, sigma=sg, r=int(round(4096 * sg / 50)))
q, r = spec.lwe["q"], spec.lwe["r"]
Tmap = encrypted.noise_map(spec, T)
cen = lambda x: ((x + q // 2) % q) - q // 2
FS = [1.0, 0.1, 0.01, 0.001]
print(f"sigma={sg}, r={r}, q={q}.  columns: nonzero rho, mean |r d.rho| / (q/2)  (over the 6 vectors, unwrapped integer), then Z at signal scale f = {FS}")
for L in (0, 1, 2, 5, 20, 50, 192):
    rng = np.random.default_rng(1003)
    sgn = {2 * T + i: int(rng.choice([-1, 1])) for i in range(L)}
    out = encrypted.run(spec, 3 * T, sensor_attack=lambda k: sgn.get(k, 0), seed=3, keep_ciphertexts=True)
    seg = out["res_ct"][2 * T:3 * T]
    rho = np.rint(out["rho"][2 * T:3 * T] * spec.cs).astype(int)
    P, b = seg[:, :-1], seg[:, -1]
    B = np.array(kernel_mod_q(P, q), dtype=np.int64)
    R = blaster_lll(B.T, delta=0.99999, cores=1, use_seysen=False, depth=0)[1].T
    D = R[np.argsort(np.linalg.norm(R.astype(float), axis=1))[:6]]
    S = sg ** 2 * (D.astype(float) @ (Tmap @ Tmap.T) @ D.astype(float).T)
    sig = np.array([int(D[i].astype(object) @ rho.astype(object)) for i in range(6)], dtype=object)   # d_i . rho (exact integer)
    X = np.array([cen(int(D[i].astype(object) @ b.astype(object))) for i in range(6)], dtype=object)
    N = np.array([cen(int(X[i] - r * sig[i])) for i in range(6)], dtype=float)                          # noise part d^T e
    zs = []
    for f in FS:
        Xf = np.array([cen(int(round(f * r * float(sig[i]))) + int(N[i])) for i in range(6)], dtype=float)
        zs.append(float(Xf @ np.linalg.solve(S, Xf)))
    big = np.mean([abs(r * float(sig[i])) for i in range(6)]) / (q / 2)
    print(f"L={L:3d}  nz={np.count_nonzero(rho):3d}  |r d.rho|/(q/2) = {big:7.2f}   Z(f) = " + "  ".join(f"{z:8.1f}" for z in zs), flush=True)
