"""One entry point for all controller versions on both plants, with the same attack scenario.

    log = run(plant="linear" | "fourtank", mode="nominal" | "quantized", scenario=None)

Modes:
    nominal    floating point controller (step 1)
    quantized  integer controller on quantized signals (step 2)
    encrypted  the integer controller evaluated on LWE ciphertexts (step 3); the log also holds the encrypted
               values enc_y, enc_u, enc_rho (second ciphertext component, centered mod q), q and dec_errors
    detected   the encrypted run plus the detector's flag, statistic and measured compute time per window (step 4,
               see detect.py); replay.py turns the measured times into a parallel plant/detector timeline
The attack is defined in physical units once, so every mode sees exactly the same attack.
The storyline is fixed: calm start, a sensor attack in the second quarter of the first n_attack samples, recovery,
an actuator attack in the last quarter of them, then a few calm windows (n > n_attack) (fixed random seed, so the gifs of all modes show the same attack).
The log has the same keys for every mode: t, y, u, rho (residual in output units) and the attack windows.
"""
import os
from dataclasses import dataclass
import numpy as np
from systems import LinearPaper, FourTank, simulate
from quantized import QuantizedLinear, QuantizedFourTank, PAPER, COMPACT, run as run_quantized
from encrypted import run as run_encrypted

PLANTS = {"linear": LinearPaper, "fourtank": FourTank}
MODES = ("nominal", "quantized", "encrypted", "detected")
LWE_SEED = {"linear": 4, "fourtank": 1}   # with the PAPER preset the linear plant decrypts wrongly in a few steps of most
                                           # runs; seed 4 has none (see README).  COMPACT has none for any seed.


@dataclass
class Scenario:
    plant: str
    n: int                    # number of samples
    unit_s: float             # sensor attack unit (output units); one quantization step
    a_s: int                  # sensor attack: random integer in [-a_s, a_s] times unit_s
    unit_a: float             # actuator attack unit (input units); one input quantization step
    a_a: int
    seed: int = 0
    offset_h2: float = 0.0    # four-tank initial offset of h2 [cm]
    n_attack: int = None      # the attack windows are placed within the first n_attack samples (default n)

    @property
    def sensor_window(self):
        m = self.n_attack or self.n
        return m // 4 + 1, m // 2

    @property
    def actuator_window(self):
        m = self.n_attack or self.n
        return 3 * m // 4 + 1, m

    def draws(self):
        """Integer draws of the two attacks (identical for every mode)."""
        rng = np.random.default_rng(self.seed)
        s0, s1 = self.sensor_window
        a0, a1 = self.actuator_window
        sens = {k: int(rng.integers(-self.a_s, self.a_s + 1)) for k in range(s0, s1)}
        act = {k: int(rng.integers(-self.a_a, self.a_a + 1)) for k in range(a0, a1)}
        return sens, act


def default_scenario(plant):
    # 32 key epochs of 192 steps as in the paper, attacks of +-5 quantization steps, then 4 calm epochs
    if plant == "linear":
        return Scenario("linear", 36 * 192, 1 / 5.0, 5, 1 / 5.0, 5, n_attack=32 * 192)
    return Scenario("fourtank", 36 * 192, 1 / 5.0, 5, 1 / 10.0, 10, offset_h2=1.0, n_attack=32 * 192)  # +-1 cm, +-1 V


def run(plant, mode, scenario=None, keep_ciphertexts=False, lwe="compact", det_tag="compact"):
    """lwe: "compact", "paper" or a dict of LWE parameters; det_tag: which det_results/<plant>_<tag>.npz holds the detector's flags."""
    if mode == "detected":      # the encrypted run plus the detector's flags (computed by detect.py, one npz per plant)
        import os
        path = os.path.join(os.path.dirname(__file__), "det_results", f"{plant}_{det_tag}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} is missing: run  python detect.py {plant}  (needs fpylll or blaster)")
        log = run(plant, "encrypted", scenario, lwe=lwe)
        d = np.load(path)
        log.update(mode="detected", det={k: d[k] for k in d.files})
        return log
    sc = scenario or default_scenario(plant)
    sens, act = sc.draws()
    if mode == "nominal":
        pl = PLANTS[plant]() if plant == "linear" else FourTank(offset_h2=sc.offset_h2)
        ctrl = PLANTS[plant].controller()
        out = simulate(pl, ctrl, sc.n,
                       sensor_attack=lambda k: sc.unit_s * sens.get(k, 0),
                       actuator_attack=lambda k: sc.unit_a * act.get(k, 0))
        y, u, rho = out["y"], out["u_applied"], out["rho"]
        extra = {"h": out["h"]} if "h" in out else {}
    elif mode == "quantized":
        spec = QuantizedLinear() if plant == "linear" else QuantizedFourTank()
        if plant == "fourtank":
            spec.offset_h2 = sc.offset_h2
        out = run_quantized(spec, sc.n, sensor_attack=lambda k: sens.get(k, 0),
                            actuator_attack=lambda k: act.get(k, 0) * spec.c ** (k % spec.kp + 1))
        y, u, rho = out["y"], out["u"], out["rho_unscaled"] / spec.cs          # residual in output units
        extra = {"h": out["h"]} if "h" in out else {}
    elif mode == "encrypted":
        spec = QuantizedLinear() if plant == "linear" else QuantizedFourTank()
        spec.lwe = lwe if isinstance(lwe, dict) else {"compact": COMPACT, "paper": PAPER}[lwe]
        if plant == "fourtank":
            spec.offset_h2 = sc.offset_h2
        out = run_encrypted(spec, sc.n, sensor_attack=lambda k: sens.get(k, 0),
                            actuator_attack=lambda k: act.get(k, 0) * spec.c ** (k % spec.kp + 1), seed=LWE_SEED[plant],
                            keep_ciphertexts=keep_ciphertexts)
        y, u, rho = out["y"], out["u"], out["rho"]
        extra = {k: out[k] for k in ("enc_y", "enc_u", "enc_rho", "q", "dec_errors", "timing")}
        extra["lwe"] = spec.lwe
        if "h" in out:
            extra["h"] = out["h"]
        if keep_ciphertexts:
            extra["res_ct"] = out["res_ct"]
    else:
        raise ValueError(f"unknown mode {mode!r}; available: {MODES}")
    Ts = PLANTS[plant].Ts
    log = dict(plant=plant, mode=mode, t=np.arange(sc.n) * Ts, y=np.asarray(y), u=np.asarray(u),
               rho=np.asarray(rho), sensor_window=sc.sensor_window, actuator_window=sc.actuator_window, Ts=Ts)
    log.update(extra)
    if "h" in log:                                      # four-tank: equilibrium levels and input for the illustration
        import plant as _plant
        log["h0"], log["u0"] = _plant.operating_point()
    return log
