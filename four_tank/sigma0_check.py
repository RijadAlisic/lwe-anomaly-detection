"""sigma = 0 (no LWE noise; insecure, only a sanity check of the implementation).
Then b = <P,s> + r*rho exactly, so X_i = d_i^T b = r d_i.rho mod q exactly, and Sigma = 0 so Z is undefined.
We print X_i and compare it with cen(r d_i.rho): they must agree.  A test then alarms iff some X_i != 0.
Usage: OMP_NUM_THREADS=1 python sigma0_check.py"""
import numpy as np
import encrypted
from quantized import QuantizedLinear
from _kernel import kernel_mod_q
from blaster import reduce as blaster_lll

T = 192
spec = QuantizedLinear()
spec.lwe = dict(spec.lwe, sigma=0.0)
q, r = spec.lwe["q"], spec.lwe["r"]
cen = lambda x: ((x + q // 2) % q) - q // 2
print(f"sigma=0, r={r}, q={q}")
for L in (0, 1, 2, 5, 20, 192):
    rng = np.random.default_rng(1003)
    sgn = {2 * T + i: int(rng.choice([-1, 1])) for i in range(L)}
    out = encrypted.run(spec, 3 * T, sensor_attack=lambda k: sgn.get(k, 0), seed=3, keep_ciphertexts=True)
    for w in ((1, 2) if L == 0 else (2,)):
        seg = out["res_ct"][w * T:(w + 1) * T]
        rho = np.rint(out["rho"][w * T:(w + 1) * T] * spec.cs).astype(int)
        P, b = seg[:, :-1], seg[:, -1]
        B = np.array(kernel_mod_q(P, q), dtype=np.int64)
        R = blaster_lll(B.T, delta=0.99999, cores=1, use_seysen=False, depth=0)[1].T
        D = R[np.argsort(np.linalg.norm(R.astype(float), axis=1))[:6]]
        X = [cen(int(D[i].astype(object) @ b.astype(object))) for i in range(6)]
        pred = [cen(r * int(D[i].astype(object) @ rho.astype(object))) for i in range(6)]
        print(f"L={L:3d} epoch {w}: nonzero rho {np.count_nonzero(rho):3d}  decryption errors {out['dec_errors']}  X = {X}")
        print(f"        cen(r d.rho) = {pred}   equal: {X == pred}   alarm (any X != 0): {any(x != 0 for x in X)}", flush=True)
