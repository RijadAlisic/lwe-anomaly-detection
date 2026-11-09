"""Does a larger noise make the detector's job harder?  Linear plant, AERO2 q and r, sigma varied, 3 key epochs
(attack in the middle one).  Reports decryption errors of the control loop and the detector's time and decision.
Run with the research venv python:  python sigma_sweep.py"""
import numpy as np
import encrypted, detect
from quantized import QuantizedLinear, AERO2

rng = np.random.default_rng(0)
att = {k: int(rng.integers(-5, 6)) for k in range(192, 384)}
print("sigma  q/sigma   dec.errors  window: Z (alarm) seconds        max|d|")
for sigma in (50, 200, 800, 3200, 12800):
    spec = QuantizedLinear(); spec.lwe = dict(AERO2, sigma=float(sigma))
    out = encrypted.run(spec, 576, sensor_attack=lambda k: att.get(k, 0), seed=4, keep_ciphertexts=True)
    Tm = encrypted.noise_map(spec, 192); res = out["res_ct"]; row = []
    for w in range(3):
        seg = res[w * 192:(w + 1) * 192]
        try:
            d = detect.detect_window(seg[:, :-1], seg[:, -1], out["q"], float(sigma), Tm)
            row.append(f"{d['Z']:8.1f} ({'A' if d['flag_chi'] else '-'}) {d['seconds']:5.1f}s")
        except Exception as e:
            row.append(f"fail:{type(e).__name__}")
    print(f"{sigma:5d}  {AERO2['q']/sigma:8.0f}  {int(out['dec_errors']):6d}      " + "  |  ".join(row), flush=True)
