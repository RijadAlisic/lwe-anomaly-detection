"""Step 4: anomaly detection on the encrypted residual, per key epoch (detection window), for the Aero2 parameters.

For every window of T residual ciphertexts (P_k, b_k) the detector, which never sees the key:
  1. builds the q-ary kernel lattice {d : d^T P = 0 mod q} and reduces it with LLL (blaster, as in the Aero2
     detector; fpylll as a slower fallback),
  2. takes the K shortest vectors d_i and forms X_i = d_i^T b mod q (the secret cancels),
  3. whitens, Z = X^T (D^T Sigma D)^-1 X ~ chi^2_K under the null hypothesis, and raises an alarm if Z exceeds the
     chi^2_K quantile (the multi-vector test of the paper; the single-vector test of Theorem 1 is reported too).
The time spent per window is recorded.  With the Aero2 parameters (q/sigma about 1.7e5) LLL alone gives vectors
short enough, so no BKZ is needed.

Run with a Python that has fpylll (for example the research virtual environment):
    python detect.py linear      or      python detect.py fourtank       ->  det_results/<plant>_aero2.npz
"""
import os, sys, time
import numpy as np
from scipy.stats import chi2, norm
from fpylll import IntegerMatrix, LLL
try:
    from blaster import reduce as blaster_lll
except ImportError:
    blaster_lll = None
import experiment as ex
import encrypted
from _kernel import kernel_mod_q

K, ALPHA = 6, 0.001


def centered(x, q):
    x = int(x) % q
    return x - q if x > q // 2 else x


def detect_window(P, b, q, sigma, Tmap, K=K, alpha=ALPHA):
    t0 = time.perf_counter()
    B = np.array(kernel_mod_q(P, q), dtype=np.int64)
    if blaster_lll is not None:                                                  # fast LLL (as in the Aero2 detector)
        R = blaster_lll(B.T, delta=0.99999, cores=1, use_seysen=False, depth=0)[1].T
    else:                                                                        # fallback: fpylll (slower)
        A = IntegerMatrix.from_matrix([[int(x) for x in row] for row in B])
        LLL.reduction(A)
        R = np.array([[A[i, j] for j in range(A.ncols)] for i in range(A.nrows)], dtype=np.int64)
    idx = np.argsort(np.linalg.norm(R.astype(float), axis=1))[:K]
    D = R[idx]                                                                  # K x N
    X = np.array([centered(int(D[i].astype(object) @ b.astype(object)), q) for i in range(K)], dtype=float)
    S = sigma ** 2 * (D.astype(float) @ (Tmap @ Tmap.T) @ D.astype(float).T)      # D Sigma D^T  (K x K)
    Z = float(X @ np.linalg.solve(S, X))
    z1 = abs(X[0]) / np.sqrt(S[0, 0])
    return dict(flag_chi=bool(Z > chi2.isf(alpha, K)), flag_single=bool(z1 > norm.isf(alpha / 2)), Z=Z,
                z1=float(z1), norms=np.linalg.norm(D.astype(float), axis=1), seconds=time.perf_counter() - t0,
                ok=bool(all(not ((D[i].astype(object) @ P.astype(object)) % q).any() for i in range(K))))


def main(plant):
    from quantized import QuantizedLinear, QuantizedFourTank
    sc = ex.default_scenario(plant)
    log = ex.run(plant, "encrypted", sc, keep_ciphertexts=True, lwe="aero2")
    spec = QuantizedLinear() if plant == "linear" else QuantizedFourTank()
    T, v, q, sigma = spec.lwe["T"], spec.lwe["v"], spec.lwe["q"], spec.lwe["sigma"]
    res = log["res_ct"]
    nw = len(res) // T
    Tmap = encrypted.noise_map(spec, T)
    rows = []
    for w in range(nw):
        seg = res[w * T:(w + 1) * T]
        r = detect_window(seg[:, :-1], seg[:, -1], q, sigma, Tmap)
        a0, a1 = sc.sensor_window, sc.actuator_window
        lo, hi = w * T, (w + 1) * T
        attacked = (lo < a0[1] and hi > a0[0]) or (lo < a1[1] and hi > a1[0])
        rho_nz = int(np.count_nonzero(log["rho"][lo:hi]))
        rows.append((w, attacked, rho_nz, r))
        print(f"window {w:2d}  attacked={str(attacked):5s}  nonzero residual samples {rho_nz:3d}  "
              f"Z={r['Z']:11.1f}  alarm(chi2_{K})={r['flag_chi']}  alarm(single)={r['flag_single']}  "
              f"max|d|={r['norms'].max():.0f}  {r['seconds']:.1f} s  d^T P=0: {r['ok']}", flush=True)
    os.makedirs("det_results", exist_ok=True)
    np.savez(f"det_results/{plant}_aero2.npz", window=np.arange(nw), attacked=[x[1] for x in rows],
             rho_nonzero=[x[2] for x in rows], flag_chi=[x[3]["flag_chi"] for x in rows],
             flag_single=[x[3]["flag_single"] for x in rows], Z=[x[3]["Z"] for x in rows],
             seconds=[x[3]["seconds"] for x in rows], T=T, K=K, alpha=ALPHA)
    calm = [x for x in rows if not x[1]]
    att = [x for x in rows if x[1]]
    print(f"\ncalm windows with an alarm: {sum(x[3]['flag_chi'] for x in calm)}/{len(calm)}   "
          f"(of these with strictly zero residual: {sum(x[3]['flag_chi'] for x in calm if x[2]==0)}/"
          f"{sum(1 for x in calm if x[2]==0)})")
    print(f"attacked windows with an alarm: {sum(x[3]['flag_chi'] for x in att)}/{len(att)};  mean time per window "
          f"{np.mean([x[3]['seconds'] for x in rows]):.1f} s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fourtank")
