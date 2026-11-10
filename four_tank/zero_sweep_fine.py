"""Zero-dynamics counterpart of the attack-length sweep (compare_detectors.py, ATTACK=random).
A disturbance and a zero-dynamics attack start 4 steps into key epoch 1 (stealth_example.py).  The attacker builds the attack from a guess
xhat = (x1, g*x2) of the hidden state: g = 1 matched, g < 1 mismatched by |1-g|; g = None: disturbance only (no attack).
Plants: stable zero (xi = 3) and unstable zero (xi = 0.7).  Encrypted loop with sigma = 50, r = 2000 (r/sigma = 40), 6-vector chi2 detector.  The unstable-zero plant uses the lower-gain controller
(-1, -2, -1), cu = 5 (design_unstable.py); the stable-zero plant keeps (2, 1, 3), cu = 10.
Per run it keeps Z of epoch 0 (calm) and epoch 1, the number of nonzero residual samples, decryption errors, and the ciphertext residual of epoch 1.
Usage:  OMP_NUM_THREADS=1 python zero_sweep_fine.py [KEYS] [SEED0]  ->  det_results/zero_sweep_fine[_sSEED0].json"""
import sys, json, os, time
import numpy as np
from multiprocessing import Pool
import quantized
import zero_attack as za
import stealth_example as se

za.ZeroSpec.lwe = dict(quantized.COMPACT, sigma=float(os.environ.get("SIGMA", 50)), r=2000)       # r/sigma = 40: decrypts without errors for both plants
za.CASES["unstable"] = dict(za.CASES["unstable"], gains=(-1, -2, -1), cu=5.0)   # lower-gain controller (noise gain 2.4 instead of 8.8, see design_unstable.py)
GS = [None, 1.0, 0.99, 0.98, 0.95, 0.9, 0.8, 0.5, 0.0]
KINDS = ("stable", "unstable")
T = se.T


def job(c):
    kind, g, seed = c
    r = se.run(kind, g, None, seed)
    return dict(kind=kind, g=g, seed=seed, Z0=r["Z0"], Z1=r["Z"], flag0=r["flag0"], flag1=r["flag"], nz1=r["nz"], dec=r["dec"],
                max_x2=float(np.abs(r["x2"]).max()), max_y=float(np.abs(r["y"][T:]).max()),
                enc=[int(v) for v in r["enc"][T:2 * T]])


if __name__ == "__main__":
    keys = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    s0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    gs_by = json.loads(os.environ["GS_JSON"]) if os.environ.get("GS_JSON") else {k: GS for k in KINDS}      # per-plant list of guesses
    jobs = [(k, g, s) for s in range(s0, s0 + keys) for k in KINDS for g in gs_by[k]]
    print(len(jobs), "runs", flush=True)
    t0 = time.time(); out = []
    with Pool(int(os.environ.get("WORKERS", 5))) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
            out.append(r)
            print(f"{i}/{len(jobs)} {r['kind']} g={r['g']} seed={r['seed']} nz={r['nz1']} Z={r['Z1']:.0f} dec={r['dec']} {time.time() - t0:.0f}s", flush=True)
    json.dump(out, open("det_results/zero_sweep_fine%s%s.json" % ("_s%d" % s0 if s0 else "", os.environ.get("TAG", "")), "w"))
