"""Smallest r that decrypts without errors?  Encrypted loop (3 key epochs, no attack and with a random-sign attack of length 192),
sigma = 50, several r.  Counts steps where the decrypted input u or residual differs from the plaintext shadow (dec_errors)."""
import sys
import numpy as np
import encrypted
from quantized import QuantizedLinear

T = 192
sg = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
print(f"sigma={sg}, q=8388617;  r/sigma, r, decryption errors out of {3 * T} steps (no attack | attack length 192)")
for r in (150, 200, 300, 400, 800, 1600, 4096):
    spec = QuantizedLinear(); spec.lwe = dict(spec.lwe, sigma=sg, r=r)
    errs = []
    for atk in (False, True):
        rng = np.random.default_rng(1003)
        sgn = {2 * T + i: int(rng.choice([-1, 1])) for i in range(T)} if atk else {}
        errs.append(encrypted.run(spec, 3 * T, sensor_attack=lambda k: sgn.get(k, 0), seed=3)["dec_errors"])
    print(f"  r/sigma={r / sg:5.1f}  r={r:5d}   errors: {errs[0]:3d} | {errs[1]:3d}", flush=True)
