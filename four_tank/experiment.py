"""One entry point for all controller versions on both plants, with the same attack scenario.

    log = run(plant="linear" | "fourtank", mode="nominal" | "quantized", scenario=None)

Modes (more to come: "detected"):
    nominal    floating point controller (step 1)
    quantized  integer controller on quantized signals (step 2)
    encrypted  the integer controller evaluated on LWE ciphertexts (step 3); the log also holds the encrypted
               values enc_y, enc_u, enc_rho (second ciphertext component, centered mod q), q and dec_errors
The attack is defined in physical units once, so every mode sees exactly the same attack.
The storyline is fixed: calm start, a sensor attack in the second quarter of the run, recovery, then an
actuator attack in the last quarter (fixed random seed, so the gifs of all modes show the same attack).
The log has the same keys for every mode: t, y, u, rho (residual in output units) and the attack windows.
"""
from dataclasses import dataclass
import numpy as np
from systems import LinearPaper, FourTank, simulate
from quantized import QuantizedLinear, QuantizedFourTank, run as run_quantized
from encrypted import run as run_encrypted

PLANTS = {"linear": LinearPaper, "fourtank": FourTank}
MODES = ("nominal", "quantized", "encrypted")
LWE_SEED = {"linear": 4, "fourtank": 1}   # linear: the paper's LWE parameters decrypt wrongly in a few steps of most runs;
                                           # seed 4 has none (see README); encrypted.run reports dec_errors


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

    @property
    def sensor_window(self):
        return self.n // 4 + 1, self.n // 2

    @property
    def actuator_window(self):
        return 3 * self.n // 4 + 1, self.n

    def draws(self):
        """Integer draws of the two attacks (identical for every mode)."""
        rng = np.random.default_rng(self.seed)
        s0, s1 = self.sensor_window
        a0, a1 = self.actuator_window
        sens = {k: int(rng.integers(-self.a_s, self.a_s + 1)) for k in range(s0, s1)}
        act = {k: int(rng.integers(-self.a_a, self.a_a + 1)) for k in range(a0, a1)}
        return sens, act


def default_scenario(plant):
    if plant == "linear":   # the paper: 32 key epochs of 192 steps, attacks of +-5 quantization steps
        return Scenario("linear", 6144, 1 / 5.0, 5, 1 / 5.0, 5)
    return Scenario("fourtank", 1024, 1.0, 1, 0.01, 100, offset_h2=1.0)   # +-1 cm sensor, +-1 V actuator


def run(plant, mode, scenario=None, keep_ciphertexts=False):
    sc = scenario or default_scenario(plant)
    sens, act = sc.draws()
    if mode == "nominal":
        pl = PLANTS[plant]() if plant == "linear" else FourTank(offset_h2=sc.offset_h2)
        ctrl = PLANTS[plant].controller()
        out = simulate(pl, ctrl, sc.n,
                       sensor_attack=lambda k: sc.unit_s * sens.get(k, 0),
                       actuator_attack=lambda k: sc.unit_a * act.get(k, 0))
        y, u, rho = out["y"], out["u_applied"], out["rho"]
    elif mode == "quantized":
        spec = QuantizedLinear() if plant == "linear" else QuantizedFourTank()
        if plant == "fourtank":
            spec.offset_h2 = sc.offset_h2
        out = run_quantized(spec, sc.n, sensor_attack=lambda k: sens.get(k, 0),
                            actuator_attack=lambda k: act.get(k, 0) * spec.c ** (k % spec.kp + 1))
        y, u, rho = out["y"], out["u"], out["rho_unscaled"] / spec.cs          # residual in output units
    elif mode == "encrypted":
        spec = QuantizedLinear() if plant == "linear" else QuantizedFourTank()
        if plant == "fourtank":
            spec.offset_h2 = sc.offset_h2
        out = run_encrypted(spec, sc.n, sensor_attack=lambda k: sens.get(k, 0),
                            actuator_attack=lambda k: act.get(k, 0) * spec.c ** (k % spec.kp + 1), seed=LWE_SEED[plant],
                            keep_ciphertexts=keep_ciphertexts)
        y, u, rho = out["y"], out["u"], out["rho"]
        extra = {k: out[k] for k in ("enc_y", "enc_u", "enc_rho", "q", "dec_errors")}
        extra["lwe"] = spec.lwe
        if keep_ciphertexts:
            extra["res_ct"] = out["res_ct"]
    else:
        raise ValueError(f"unknown mode {mode!r}; available: {MODES}")
    Ts = PLANTS[plant].Ts
    log = dict(plant=plant, mode=mode, t=np.arange(sc.n) * Ts, y=np.asarray(y), u=np.asarray(u),
               rho=np.asarray(rho), sensor_window=sc.sensor_window, actuator_window=sc.actuator_window, Ts=Ts)
    log.update(extra if mode == "encrypted" else {})
    return log
