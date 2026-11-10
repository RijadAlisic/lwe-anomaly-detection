"""Does the sweep's zero-dynamics attack equal the closed form  u_k = -(0.4/b1) g xi0 z0^(k-k0)  (k0 = first attacked step)?
Plaintext quantized loop with the attack planned by stealth_example.plan (as in zero_sweep_fine.py); compares the applied plant input
with the closed form.  The difference should be at most the input rounding 1/(2 cu)."""
import numpy as np
import quantized
import zero_attack as za
import stealth_example as se

za.CASES["unstable"] = dict(za.CASES["unstable"], gains=(-1, -2, -1), cu=5.0)
K0, N = se.K0, se.N
for kind in ("stable", "unstable"):
    cfg = za.CASES[kind]; z0 = cfg["z0"]; b1 = -0.4 * za.B2 / (z0 - 0.1); xi = se.XI[kind]; cu = cfg["cu"]
    for g in (1.0, 0.9, 0.5):
        spec, Ua = se.plan(kind, g, xi)
        out = quantized.run(spec, N, actuator_attack=lambda k: Ua.get(k, 0))
        u = np.asarray(out["u"])
        k = np.arange(K0, K0 + 40)                         # first 40 attacked steps
        formula = -(0.4 / b1) * g * xi * z0 ** (k - K0)
        err = np.abs(u[k] - formula)
        inj = np.array([Ua[kk] for kk in k]) / cu                       # the injected attack u^a_k = U_a/cu
        err_inj = np.abs(inj - formula)
        print(f"{kind:8s} g={g:3.1f}: applied input vs formula {err.max():.3f}; INJECTED attack vs formula {err_inj.max():.3f} "
              f"(first 3 steps {np.round(err_inj[:3],3).tolist()}, from step 3 on {err_inj[3:].max():.3f}); rounding bound {1/(2*cu):.3f}; max|y| {np.abs(np.asarray(out['y'])[K0:K0+40]).max():.3f}")
