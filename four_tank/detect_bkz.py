"""Detector with BKZ on the four-tank loop at a larger noise, all 36 key epochs (windows), in parallel on one core each.
Same output as detect.py (det_results/fourtank_bkz<BETA>.npz, used by make_gifs.py --det-tag), but the reduction is BKZ-BETA
(blaster, 2 tours) instead of LLL, so the measured time per window is the cost of the harder detector.
Usage: python detect_bkz.py SIGMA BETA [WORKERS]    (r = 40 SIGMA; needs blaster-python)"""
import sys, os, time
import numpy as np
from multiprocessing import Pool
from scipy.stats import chi2, norm
import experiment as ex, encrypted
from quantized import QuantizedFourTank, COMPACT
from _kernel import kernel_mod_q
from detect import centered
from blaster import reduce as blaster_reduce

K, ALPHA = 6, 0.001


def job(args):
    P, b, q, sigma, Tm, beta = args
    t0 = time.perf_counter()
    B = np.array(kernel_mod_q(P, q), dtype=np.int64)
    R = blaster_reduce(B.T, delta=0.99999, cores=1, use_seysen=False, depth=0, beta=beta, bkz_tours=2)[1].T
    idx = np.argsort(np.linalg.norm(R.astype(float), axis=1))[:K]; D = R[idx]
    X = np.array([centered(int(D[i].astype(object) @ b.astype(object)), q) for i in range(K)], float)
    S = sigma ** 2 * (D.astype(float) @ (Tm @ Tm.T) @ D.astype(float).T)
    Z = float(X @ np.linalg.solve(S, X)); z1 = abs(X[0]) / np.sqrt(S[0, 0])
    return dict(Z=Z, z1=float(z1), seconds=time.perf_counter() - t0, norm=float(np.linalg.norm(D.astype(float), axis=1)[0]))


if __name__ == "__main__":
    sigma, beta = float(sys.argv[1]), int(sys.argv[2]); workers = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    lw = dict(COMPACT, sigma=sigma, r=int(round(40 * sigma)))
    sc = ex.default_scenario("fourtank")
    log = ex.run("fourtank", "encrypted", sc, keep_ciphertexts=True, lwe=lw)
    spec = QuantizedFourTank(); spec.lwe = lw; spec.offset_h2 = sc.offset_h2
    T, q = lw["T"], lw["q"]; res = log["res_ct"]; nw = len(res) // T
    Tm = encrypted.noise_map(spec, T)
    print(f"sigma={sigma:g} r={lw['r']} decryption errors {log['dec_errors']}; BKZ-{beta} on {nw} windows, {workers} workers", flush=True)
    jobs = [(res[w * T:(w + 1) * T, :-1], res[w * T:(w + 1) * T, -1], q, sigma, Tm, beta) for w in range(nw)]
    out = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap(job, jobs), 1):
            out.append(r); print(f"window {i - 1:2d}  Z={r['Z']:9.1f}  |d|={r['norm']:.0f}  {r['seconds']:.0f} s", flush=True)
    a0, a1 = sc.sensor_window, sc.actuator_window
    att = [(w * T < a0[1] and (w + 1) * T > a0[0]) or (w * T < a1[1] and (w + 1) * T > a1[0]) for w in range(nw)]
    Z = np.array([r["Z"] for r in out]); th = chi2.isf(ALPHA, K)
    np.savez(f"det_results/fourtank_bkz{beta}.npz", window=np.arange(nw), attacked=att,
             rho_nonzero=[int(np.count_nonzero(log["rho"][w * T:(w + 1) * T])) for w in range(nw)],
             flag_chi=Z > th, flag_single=[r["z1"] > norm.isf(ALPHA / 2) for r in out], Z=Z, seconds=[r["seconds"] for r in out],
             T=T, K=K, alpha=ALPHA, Ts=log["Ts"], n=len(res), z1=[r["z1"] for r in out], sensor_window=sc.sensor_window,
             actuator_window=sc.actuator_window, machine=f"{workers} windows in parallel, 1 core each; seconds = measured compute time per window",
             method=f"kernel lattice + BKZ-{beta}", sigma=sigma, r=lw["r"])
    att = np.array(att)
    print(f"\nattacked windows with an alarm: {int((Z > th)[att].sum())}/{att.sum()}; calm windows with an alarm: {int((Z > th)[~att].sum())}/{(~att).sum()}; mean time {np.mean([r['seconds'] for r in out]):.0f} s")
