"""Probe: four-tank window at larger sigma, LLL then BKZ with growing block size (fpylll).  Reports time and detector statistic.
Usage: python bkz_probe.py SIGMA R WINDOW BETA [BETA ...]"""
import sys, time
import numpy as np
from scipy.stats import chi2
import experiment as ex, encrypted
from quantized import QuantizedFourTank, COMPACT
from _kernel import kernel_mod_q
from detect import centered

sigma, rr, w = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); betas = [int(b) for b in sys.argv[4:]]
sc = ex.default_scenario("fourtank")
spec = QuantizedFourTank(); spec.lwe = dict(COMPACT, sigma=sigma, r=rr); spec.offset_h2 = sc.offset_h2
sens, act = sc.draws()
out = encrypted.run(spec, sc.n if w >= 8 else 9 * 192, sensor_attack=lambda k: sens.get(k, 0),
                    actuator_attack=lambda k: act.get(k, 0) * spec.c ** (k % spec.kp + 1), seed=ex.LWE_SEED["fourtank"],
                    keep_ciphertexts=True)
q, T = out["q"], 192
seg = out["res_ct"][w * T:(w + 1) * T]; P, b = seg[:, :-1], seg[:, -1]
Tm = encrypted.noise_map(spec, T)
print(f"sigma={sigma} r={rr} max|U|,|yq| range q/2r={COMPACT['q']/2/rr:.0f}  window {w}  decryption errors {out['dec_errors']}  nonzero residuals {int(np.count_nonzero(out['rho'][w*T:(w+1)*T]))}", flush=True)
B = np.array(kernel_mod_q(P, q), dtype=np.int64)
from blaster import reduce as blaster_reduce
def stat(R, K=6):
    R = np.asarray(R, dtype=np.int64)
    idx = np.argsort(np.linalg.norm(R.astype(float), axis=1))[:K]; D = R[idx]
    X = np.array([centered(int(D[i].astype(object) @ b.astype(object)), q) for i in range(K)], float)
    S = sigma ** 2 * (D.astype(float) @ (Tm @ Tm.T) @ D.astype(float).T)
    return float(X @ np.linalg.solve(S, X)), np.linalg.norm(D.astype(float), axis=1)[0]
for beta in [0] + betas:
    t0 = time.perf_counter()
    kw = dict(beta=beta, bkz_tours=2) if beta else {}
    R = blaster_reduce(B.T, delta=0.99999, cores=1, use_seysen=False, depth=0, **kw)[1].T
    dt = time.perf_counter() - t0; Z, n0 = stat(R)
    print(f"{'LLL' if not beta else 'BKZ-%d' % beta:8s} {dt:7.1f} s   shortest |d| {n0:8.0f}   Z={Z:8.1f}  (alarm if > {chi2.isf(1e-3, 6):.1f})", flush=True)
