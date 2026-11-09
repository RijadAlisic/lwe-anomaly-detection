"""Patches for the zero-dynamics study: (1) how long a matched attack stays invisible (plain integer loop, no
ciphertext range limit), (2) detection versus the attacker's guess g*xi of the state, encrypted, 5 key seeds.
Run with the research venv python:  python zero_attack_sweep.py"""
import numpy as np
import encrypted, quantized, detect
import zero_attack as za
from zero_attack_study import XI

print("(1) first step with a nonzero residual under the matched attack (plain loop, 600 steps)")
for kind, z0s in (("stable", (0.7,)), ("unstable", (1.005, 1.01, 1.02))):
    for z0 in z0s:
        za.CASES[kind]["z0"] = z0
        spec = za.ZeroSpec(kind, XI[kind])
        out = quantized.run(spec, 600, actuator_attack=spec.attack(None))
        nz = np.nonzero(out["rho"])[0]
        print(f"  {kind:8s} z0={z0:5.3f}: first nonzero residual at k={nz[0] if len(nz) else None}, "
              f"max|y|={np.abs(out['y']).max():.2f}, max|U attack|={abs(spec.attack(None)(int(nz[0]) if len(nz) else 599))}")
za.CASES["unstable"]["z0"] = 1.02

print("\n(2) detection versus the attacker's guess g (attack sized for g*xi, true state xi), 5 key seeds")
print("kind      g      nonzero  Z (median)   alarms/5")
for kind in ("stable", "unstable"):
    for g in (1.0, 0.98, 0.95, 0.9, 0.8, 0.5, 0.0):
        Zs, flags, nzs = [], [], []
        for seed in range(5):
            spec = za.ZeroSpec(kind, XI[kind], g * XI[kind])
            act = spec.attack(None)
            out = encrypted.run(spec, 192, actuator_attack=act, seed=seed, keep_ciphertexts=True)
            res = out["res_ct"]
            r = detect.detect_window(res[:, :-1], res[:, -1], out["q"], spec.lwe["sigma"], encrypted.noise_map(spec, 192))
            Zs.append(r["Z"]); flags.append(r["flag_chi"]); nzs.append(int(np.count_nonzero(out["rho"])))
        print(f"{kind:8s} {g:5.2f}  {int(np.median(nzs)):5d}  {np.median(Zs):9.1f}   {sum(flags)}/5", flush=True)
