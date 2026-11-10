"""Reviewer 4, comment 2: how does the encrypted detector compare with detectors that may decrypt, and with
detectors that are fed the ciphertext?  A sensor attack of density p (a random +-1 quantization step at a fraction p of
the samples of one key epoch) tunes the residual amplitude; p = 0 is no attack.

Per run (5 key epochs of T = 192 steps, linear plant, COMPACT parameters, encrypted loop):
    epoch 0  start-up transient           epoch 1  calm        epoch 2  attacked (density p)
    epoch 3  transient after the attack   epoch 4  calm
Detectors (alpha = 1e-3 per epoch, except where stated):
    lattice    the proposed detector on ciphertexts only (q-ary kernel, LLL, 6 shortest vectors, chi2_6)
    oracle     plaintext residual, alarm on any nonzero value (the best possible test, nominal residual is exactly 0)
    chi2/CUSUM/EWMA/SPRT   textbook tests on the decrypted residual, tuned for a null noise level of S0 quantization steps
    enc-CUSUM  the same CUSUM fed the ciphertext residual (b-component, centered mod q) without / with recalibration
Usage:  OMP_NUM_THREADS=1 python compare_detectors.py [seeds]  ->  det_results/detector_comparison.json
"""
import sys, json, time, os
import numpy as np
from multiprocessing import Pool
from scipy.stats import chi2
from quantized import QuantizedLinear, COMPACT as quantized_COMPACT
import encrypted
from detect import detect_window

T, NW, ATT = 192, 5, 2
MODE = os.environ.get("ATTACK", "sparse")
BIAS = MODE in ("bias", "duration", "random")
SIGMA = float(os.environ.get("SIGMA", 50))          # LWE noise; r is scaled with sigma so that decryption keeps working
RATIO = float(os.environ.get("R_OVER_SIGMA", 0))     # r = RATIO * sigma (default: the COMPACT ratio 82)
PLANT = os.environ.get("PLANT", "")                      # "stablezero": plant of the zero-dynamics study with z0 = 0.7
SKIP = os.environ.get("DETECT_ALL") is None          # detect only epochs 1, 2, 4 (the others are transients)     # bias: constant sensor bias of p quantization steps over epoch 2
P_LIST = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 192] if os.environ.get("FINE") else [0, 1, 2, 5, 10, 20, 50, 100, 192] if MODE in ("duration", "random") else [0, 1, 2, 3, 5, 8, 12, 20] if BIAS else [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
S0 = 0.5            # null residual std assumed by the plaintext tests [quantization steps]
ALPHA = 1e-3        # per-epoch false-alarm target


# ---------- textbook tests on one epoch of residual samples r (integer units) ----------
def stat_chi2(r, s0=S0):
    return float(np.sum(r ** 2) / s0 ** 2)


def stat_cusum(r, s0=S0):                      # one-sided CUSUM on the variance statistic r^2/s0^2 - 1.5
    g, m = 0.0, 0.0
    for x in r:
        g = max(0.0, g + x * x / s0 ** 2 - 1.5)
        m = max(m, g)
    return m


def stat_ewma(r, s0=S0, lam=0.2):              # EWMA of r^2/s0^2, maximum over the epoch
    z, m = 1.0, 0.0
    for x in r:
        z = lam * x * x / s0 ** 2 + (1 - lam) * z
        m = max(m, z)
    return m


def stat_sprt(r, s0=S0, ratio=2.0):            # Wald SPRT (with restart) H0: N(0, s0^2) vs H1: N(0, (ratio s0)^2)
    a = 0.5 * (1 - 1 / ratio ** 2) / s0 ** 2
    c = np.log(ratio)
    g, m = 0.0, 0.0
    for x in r:
        g = max(0.0, g + a * x * x - c)
        m = max(m, g)
    return m


STATS = dict(chi2=stat_chi2, cusum=stat_cusum, ewma=stat_ewma, sprt=stat_sprt)


def null_thresholds(rng, n=40000):
    """Threshold per test: the 1-alpha quantile of the statistic over simulated null epochs N(0, S0^2)."""
    th = {"chi2": float(chi2.isf(ALPHA, T))}
    R = rng.normal(0, S0, (n, T))
    for name in ("cusum", "ewma", "sprt"):
        f = STATS[name]
        th[name] = float(np.quantile([f(R[i]) for i in range(n // 8)], 1 - ALPHA))
    return th


def attack_fn(p, seed):
    if MODE == "random":                                    # random-sign +-1 step at each of the first p samples of epoch 2
        rng = np.random.default_rng(1000 + seed)
        sg = {ATT * T + i: int(rng.choice([-1, 1])) for i in range(int(p))}
        return lambda k: sg.get(k, 0)
    if MODE == "duration":                                  # bias of one step during the first p samples of epoch 2
        return lambda k: 1 if ATT * T <= k < ATT * T + int(p) else 0
    if BIAS:
        return lambda k: int(p) if ATT * T <= k < (ATT + 1) * T else 0
    rng = np.random.default_rng(1000 + seed)
    lo, hi = ATT * T, (ATT + 1) * T
    on = {k: int(rng.choice([-1, 1])) for k in range(lo, hi) if rng.random() < p}
    return lambda k: on.get(k, 0)


def one_run(args):
    p, seed = args
    spec = QuantizedLinear()
    if PLANT == "stablezero":                                # the paper's plant with B(1) changed so that it has the zero z0 = 0.7, controller (2, 1, 3), cu = 10
        import zero_attack as za
        spec = za.ZeroSpec("stable", 0.0)
        spec.make_plant = lambda: za.ZeroPlant(0.7, (0.1, 0.0))
        spec.lwe = dict(quantized_COMPACT)
    spec.lwe = dict(spec.lwe, sigma=SIGMA, r=int(round(RATIO * SIGMA)) if RATIO else int(round(spec.lwe["r"] * SIGMA / 50)))
    out = encrypted.run(spec, NW * T, sensor_attack=attack_fn(p, seed), seed=seed, keep_ciphertexts=True)
    q, sigma = spec.lwe["q"], spec.lwe["sigma"]
    Tmap = encrypted.noise_map(spec, T)
    rho = np.rint(out["rho"] * spec.cs).astype(int)
    res = out["res_ct"]
    rec = dict(p=p, seed=seed, dec_errors=int(out["dec_errors"]), epochs=[])
    for w in range(NW):
        seg = res[w * T:(w + 1) * T]
        r = rho[w * T:(w + 1) * T]
        if SKIP and w not in (1, ATT, 4):
            d = dict(flag_chi=False, Z=0.0, seconds=0.0)
        else:
            d = detect_window(seg[:, :-1], seg[:, -1], q, sigma, Tmap, alpha=ALPHA)
        rec["epochs"].append(dict(nz=int(np.count_nonzero(r)), lattice=bool(d["flag_chi"]), Z=d["Z"], sec=d["seconds"],
                                  stats={k: f(r) for k, f in STATS.items()},
                                  enc=[int(x) for x in out["enc_rho"][w * T:(w + 1) * T]]))
    return rec


if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    SEED0 = int(os.environ.get("SEED0", 0))
    jobs = [(p, s) for s in range(SEED0, SEED0 + seeds) for p in P_LIST]
    t0 = time.time()
    res = []
    with Pool(int(os.environ.get("WORKERS", 5))) as pool:
        for i, rec in enumerate(pool.imap_unordered(one_run, jobs)):
            res.append(rec)
            print(f"{i + 1}/{len(jobs)}  p={rec['p']} seed={rec['seed']}  nz(att)={rec['epochs'][ATT]['nz']}  "
                  f"lattice(att)={rec['epochs'][ATT]['lattice']}  {time.time() - t0:.0f}s", flush=True)
    json.dump(dict(T=T, NW=NW, ATT=ATT, S0=S0, ALPHA=ALPHA, p_list=P_LIST, runs=res), open("det_results/detector_comparison%s%s.json" % ("_" + MODE if BIAS else "", ("_sigma%d" % SIGMA if SIGMA != 50 else "") + ("_r%d" % round(RATIO * SIGMA) if RATIO else "") + ("_fine" if os.environ.get("FINE") else "") + ("_" + PLANT if PLANT else "") + ("_s%d" % int(os.environ.get("SEED0", 0)) if os.environ.get("SEED0") else "")), "w"))
