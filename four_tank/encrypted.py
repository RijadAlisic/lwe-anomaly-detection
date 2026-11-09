"""Step 3: encrypted control.  The integer controller of step 2 evaluated on LWE ciphertexts.

Sensor encrypts yq, the controller works on ciphertexts only (homomorphic integer linear maps), the actuator
decrypts the input.  The key is refreshed every T steps (a multiple of the reset period kp).  Attacks act on the
ciphertexts, with the same integer conventions as quantized.run: the sensor attack adds r*a to the measurement
ciphertext (a is added to yq), the actuator attack adds r*a to the input ciphertext (a is added to U).  We also log the 'encrypted values' (second component b of each ciphertext, centered
mod q) next to the plaintext, as in Fig. 7 of the paper.
"""
import time
import numpy as np
import lwe


def centered(x, q):
    x = np.asarray(x, dtype=np.int64) % q
    return np.where(x > q // 2, x - q, x)


def run(spec, n_steps, sensor_attack=None, actuator_attack=None, seed=1, keep_ciphertexts=False):
    P = spec.lwe
    L = lwe.LWE(P["v"], P["q"], P["r"], P["sigma"], np.random.default_rng(seed))
    M = {k: np.array(v, dtype=np.int64) for k, v in spec.M.items()}
    c, kp, cs, cu, q, r = spec.c, spec.kp, spec.cs, spec.cu, P["q"], P["r"]
    pl = spec.make_plant()
    n = M["A"].shape[0]
    X = np.zeros((n, P["v"] + 1), dtype=np.int64)                   # trivial encryption of the reset state
    log = {k: [] for k in ("y", "yq", "u", "rho", "enc_y", "enc_u", "enc_rho")}
    Xp = np.zeros((n, 1), dtype=np.int64)                           # plaintext shadow, only to count decryption errors
    dec_errors, error_steps = 0, []
    res_ct = []
    tm = dict(encrypt=0.0, evaluate=0.0, decrypt=0.0)
    for k in range(n_steps):
        kb = k % kp
        if k % P["T"] == 0:
            L.new_key()                                             # key refresh (aligned with a reset)
        if kb == 0:
            X = np.zeros_like(X)                                    # periodic reset of the controller state
            Xp = np.zeros_like(Xp)
        y = pl.y()
        yq = int(np.round(cs * y))
        t0 = time.perf_counter()
        Ye = L.encrypt([yq])
        tm['encrypt'] += time.perf_counter() - t0
        t0 = time.perf_counter()
        if sensor_attack and sensor_attack(k):
            Ye = Ye.copy(); Ye[0, -1] = (Ye[0, -1] + r * sensor_attack(k)) % q
        Ys = (c ** kb * Ye) % q
        U = (L.lin(M["C"], X) + L.lin(M["D"], Ys)) % q
        Yhat = (L.lin(M["Co"], X) + L.lin(M["Do"], Ys)) % q
        Xn = (L.lin(M["A"], X) + L.lin(M["B"], Ys)) % q
        Rho = (c ** (kb + 1) * Ye - Yhat) % q
        tm['evaluate'] += time.perf_counter() - t0
        if actuator_attack and actuator_attack(k):
            U = U.copy(); U[0, -1] = (U[0, -1] + r * actuator_attack(k)) % q
        ya = yq + (sensor_attack(k) if sensor_attack else 0)        # shadow: what the controller should see
        Ysp = c ** kb * ya
        Up = int((M["C"] @ Xp)[0, 0] + M["D"][0, 0] * Ysp) + (actuator_attack(k) if actuator_attack else 0)
        Rp = c ** (kb + 1) * ya - int((M["Co"] @ Xp)[0, 0] + M["Do"][0, 0] * Ysp)
        bad = int(L.decrypt(U)[0] != Up or L.decrypt(Rho)[0] != Rp)
        dec_errors += bad
        if bad:
            error_steps.append(k)
        Xp = M["A"] @ Xp + M["B"] * Ysp
        t0 = time.perf_counter()
        u = L.decrypt(U)[0] / (c ** (kb + 1) * cu)
        tm['decrypt'] += time.perf_counter() - t0
        rho = L.decrypt(Rho)[0] / c ** (kb + 1) / cs                # residual in output units
        for key, val in zip(log, (y, yq, u, rho, centered(Ye[:, -1], q)[0], centered(U[:, -1], q)[0],
                                  centered(Rho[:, -1], q)[0])):
            log[key].append(val)
        if keep_ciphertexts:
            res_ct.append(Rho[0].copy())                              # full residual ciphertext (P | b)
        X = Xn
        pl.apply(u)
    out = {k: np.array(v) for k, v in log.items()}
    out["q"] = q
    out["timing"] = {k: 1e3 * v / n_steps for k, v in tm.items()}          # ms per sample (Python, one core)
    if keep_ciphertexts:
        out["res_ct"] = np.array(res_ct)
    out["error_steps"] = error_steps
    out["dec_errors"] = dec_errors                                   # steps where noise broke the decryption
    return out
