"""Which block size does the detector need at larger noise?  Four-tank loop, COMPACT q, sigma varied, r = 40 sigma (decrypts without errors).
For each sigma: 8 windows with a sensor attack (windows 8-15) and 8 calm windows (3-6, 17-20), each reduced with LLL, BKZ-40 and
BKZ-50 (blaster, 2 tours, one core); reports the alarm fraction of the 6-vector chi-square test (alpha = 1e-3) and the time per window.
Usage: python bkz_sigma_sweep.py SIGMA       ->  det_results/bkz_sigma_sweep_<SIGMA>.json   (needs blaster-python)"""
import sys, json, time
import numpy as np
from scipy.stats import chi2
import experiment as ex, encrypted
from quantized import QuantizedFourTank, COMPACT
from _kernel import kernel_mod_q
from detect import centered
from blaster import reduce as blaster_reduce

K, ALPHA, T = 6, 1e-3, 192
BETAS = (0, 40, 50)                                      # 0 = LLL
ATT, CALM = list(range(8, 16)), [3, 4, 5, 6, 17, 18, 19, 20]


def stat(R, b, q, sigma, Tm):
    R = np.asarray(R, dtype=np.int64)
    idx = np.argsort(np.linalg.norm(R.astype(float), axis=1))[:K]; D = R[idx]
    X = np.array([centered(int(D[i].astype(object) @ b.astype(object)), q) for i in range(K)], float)
    S = sigma ** 2 * (D.astype(float) @ (Tm @ Tm.T) @ D.astype(float).T)
    return float(X @ np.linalg.solve(S, X)), float(np.linalg.norm(D.astype(float), axis=1)[0])


if __name__ == "__main__":
    sigma = float(sys.argv[1]); rr = int(round(40 * sigma))
    sc = ex.default_scenario("fourtank")
    spec = QuantizedFourTank(); spec.lwe = dict(COMPACT, sigma=sigma, r=rr); spec.offset_h2 = sc.offset_h2
    sens, act = sc.draws()
    out = encrypted.run(spec, 21 * T, sensor_attack=lambda k: sens.get(k, 0), actuator_attack=lambda k: act.get(k, 0) * spec.c ** (k % spec.kp + 1),
                        seed=ex.LWE_SEED["fourtank"], keep_ciphertexts=True)
    q = out["q"]; Tm = encrypted.noise_map(spec, T)
    print(f"sigma={sigma:g} r={rr} decryption errors {out['dec_errors']}", flush=True)
    rows = []
    for w in ATT + CALM:
        seg = out["res_ct"][w * T:(w + 1) * T]; P, b = seg[:, :-1], seg[:, -1]
        B = np.array(kernel_mod_q(P, q), dtype=np.int64)
        row = dict(window=w, attacked=w in ATT, nz=int(np.count_nonzero(out["rho"][w * T:(w + 1) * T])))
        for beta in BETAS:
            t0 = time.perf_counter()
            kw = dict(beta=beta, bkz_tours=2) if beta else {}
            R = blaster_reduce(B.T, delta=0.99999, cores=1, use_seysen=False, depth=0, **kw)[1].T
            row[f"t{beta}"] = time.perf_counter() - t0
            row[f"Z{beta}"], row[f"n{beta}"] = stat(R, b, q, sigma, Tm)
        rows.append(row); print(row, flush=True)
    json.dump(dict(sigma=sigma, r=rr, dec=int(out["dec_errors"]), rows=rows), open(f"det_results/bkz_sigma_sweep_{int(sigma)}.json", "w"))
