"""Systematic sweep of zero-dynamics attacks, encrypted (AERO2 parameters), detection per key epoch of 192 steps.

A  attack from the first step on the true zero-dynamics state (0, xi): attacker guesses g*xi, runs one window
   (192 steps), then stops; 3 windows in total (window 0 attacked, 1 and 2 aftermath).
B  attack starts from rest at step 192 (state 0): attacker uses amplitude eps, runs 192 steps (= window 1); 4 windows
   in total (0 calm, 1 attacked, 2 aftermath, 3 later).
Each configuration with 3 LWE key seeds, stable (z0 = 0.7) and unstable (z0 = 1.02) zero.
Writes det_results/zero_sweep.json.  Run with the research venv python:  python zero_sweep.py
"""
import json, itertools, sys
import numpy as np
from multiprocessing import Pool
import encrypted, detect
import zero_attack as za

T = 192
SEEDS = (0, 1, 2)


def attack_fn(spec, xi_att, k0, length):
    gain = -0.4 / spec.b1
    def f(k):
        if xi_att is None or k < k0 or k >= k0 + length:
            return 0
        return int(np.round(spec.cu * gain * xi_att * spec.z0 ** (k - k0)))
    return f


def job(cfg):
    kind, scen, xi, par, seed = cfg
    if scen == "A":                 # par = g (None = no attack)
        n, x0 = 3 * T, xi
        spec = za.ZeroSpec(kind, x0, None if par is None else par * xi)
        act = attack_fn(spec, None if par is None else par * xi, 0, T)
    else:                           # B: par = eps (None = no attack), plant at rest
        n, x0 = 4 * T, 0.0
        spec = za.ZeroSpec(kind, x0, par)
        act = attack_fn(spec, par, T, T)
    out = encrypted.run(spec, n, actuator_attack=act, seed=seed, keep_ciphertexts=True)
    res = out["res_ct"]; Tm = encrypted.noise_map(spec, T); q = out["q"]
    pl = spec.make_plant(); xs = []
    for k in range(n):
        xs.append(pl.x[:, 0].copy()); pl.apply(out["u"][k])
    xs = np.array(xs)
    wins = []
    for w in range(n // T):
        seg = res[w * T:(w + 1) * T]
        d = detect.detect_window(seg[:, :-1], seg[:, -1], q, spec.lwe["sigma"], Tm)
        sl = slice(w * T, (w + 1) * T)
        wins.append(dict(Z=d["Z"], flag=bool(d["flag_chi"]), nz=int(np.count_nonzero(out["rho"][sl])),
                         max_x2=float(np.abs(xs[sl, 1]).max()), max_y=float(np.abs(out["y"][sl]).max())))
    return dict(kind=kind, scen=scen, xi=xi, par=par, seed=seed, dec_errors=int(out["dec_errors"]), wins=wins)


def configs():
    c = []
    for kind in ("stable", "unstable"):
        for xi, gs in ((0.7, (None, 1.0, 0.95, 0.9, 0.8, 0.5, 0.0)), (0.35, (None, 1.0, 0.8, 0.5, 0.0))):
            for g in gs:
                c += [(kind, "A", xi, g, s) for s in SEEDS]
        for eps in (None, 0.05, 0.1, 0.2, 0.4):
            c += [(kind, "B", 0.0, eps, s) for s in SEEDS]
    return c


if __name__ == "__main__":
    cs = configs(); print(len(cs), "runs", flush=True)
    res = []
    with Pool(6) as p:
        for i, r in enumerate(p.imap_unordered(job, cs, chunksize=1), 1):
            res.append(r)
            if i % 10 == 0:
                print(f"{i}/{len(cs)}", flush=True)
    json.dump(res, open("det_results/zero_sweep.json", "w"))
    print("done")
