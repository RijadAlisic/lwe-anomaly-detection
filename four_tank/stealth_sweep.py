"""Systematic test around the example figure (stealth_example.py): disturbance + zero-dynamics attack 4 steps into window 1.
kind {stable, unstable} x kick amplitude xi x attacker's state guess g, 5 key seeds.  Window 0 is calm (false alarms),
window 1 is where the disturbance and the attack start.  Writes det_results/stealth_sweep.json.  Run with a Python that has blaster or fpylll; set OMP_NUM_THREADS=1
before starting (six worker processes each use one thread)."""
import json
import numpy as np
from multiprocessing import Pool
import stealth_example as se

XIS = {"stable": (3.0, 0.7), "unstable": (0.7, 0.35)}
GS = (None, 1.0, 0.98, 0.95, 0.9, 0.8, 0.5, 0.0)
SEEDS = range(5)


def job(c):
    kind, xi, g, seed = c
    r = se.run(kind, g, xi, seed)
    return dict(kind=kind, xi=xi, g=g, seed=seed, flag0=r["flag0"], flag1=r["flag"], Z0=r["Z0"], Z1=r["Z"], nz1=r["nz"],
                max_x2=float(np.abs(r["x2"]).max()), max_y=float(np.abs(r["y"][se.T:]).max()), dec=r["dec"])


if __name__ == "__main__":
    cs = [(k, xi, g, s) for k in XIS for xi in XIS[k] for g in GS for s in SEEDS]
    print(len(cs), "runs", flush=True)
    out = []
    with Pool(6) as p:
        for i, r in enumerate(p.imap_unordered(job, cs), 1):
            out.append(r)
            if i % 20 == 0:
                print(i, flush=True)
    json.dump(out, open("det_results/stealth_sweep.json", "w")); print("done")
