"""How many samples with rho != 0 does the zero-dynamics attack leave, as a function of the attacker's state guess g?
Plaintext quantized loop (no encryption, the residual does not depend on it).  Same plants and controllers as zero_sweep_fine.py."""
import numpy as np
import quantized
import zero_attack as za
import stealth_example as se

za.CASES["unstable"] = dict(za.CASES["unstable"], gains=(-1, -2, -1), cu=5.0)
T = se.T
GRID = [1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.88, 0.86, 0.84, 0.82, 0.8, 0.78, 0.76, 0.74, 0.72, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.3, 0.2, 0.1, 0.0]

def nz_of(kind, g):
    spec, Ua = se.plan(kind, g, se.XI[kind])
    out = quantized.run(spec, se.N, actuator_attack=lambda k: Ua.get(k, 0))
    return int(np.count_nonzero(out["rho_unscaled"][T:]))

if __name__ == "__main__":
    for kind in ("stable", "unstable"):
        print(kind, {g: nz_of(kind, g) for g in GRID}, flush=True)
