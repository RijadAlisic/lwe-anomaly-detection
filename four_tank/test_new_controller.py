"""Does a lower-gain controller for the unstable zero keep (a) the matched zero-dynamics attack stealthy and (b) decryption correct at a small r?
For each candidate: matched attack (g = 1) and disturbance only (g = None), 3 keys, r in (1200, 2000); prints nonzero residual samples in epoch 1,
max |y| in epoch 1, decryption errors, Z of epoch 1."""
import sys
import numpy as np
import quantized
import zero_attack as za
import stealth_example as se

CANDS = [("current", (-5, -6, 4), 10.0), ("cu5", (-1, -2, -1), 5.0), ("cu4", (-2, -1, 0), 4.0), ("cu3", (-2, -1, 1), 3.0)]
for name, gains, cu in CANDS:
    za.CASES["unstable"] = dict(za.CASES["unstable"], gains=gains, cu=cu)
    for r in ((1200, 2000) if name != "current" else (4096,)):
        za.ZeroSpec.lwe = dict(quantized.COMPACT, sigma=50.0, r=r)
        res = {}
        for g in (1.0, None):
            rows = [se.run("unstable", g, None, s) for s in range(3)]
            res[g] = (sum(x["dec"] for x in rows), [x["nz"] for x in rows], [round(float(np.abs(x["y"][se.T:]).max()), 2) for x in rows], [round(x["Z"]) for x in rows])
        print(f"{name:8s} gains {gains} cu {cu:4.1f} r {r:4d}: matched: dec {res[1.0][0]} nz {res[1.0][1]} max|y| {res[1.0][2]} Z {res[1.0][3]} | disturbance only: dec {res[None][0]} nz {res[None][1]} Z {res[None][3]}", flush=True)
