"""Step 2: quantized signals and the integer controller of eq. (23), still in plain integers (no encryption).

  measurement   yq = round(cs * y)                      (integer)
  controller    X+   = [cA] X + [cB] (c^kbar yq)        (integer matrices, reset to xss every kp steps)
                U    = [cC] X + [cD] (c^kbar yq)        = c^(kbar+1) * (integer input)
                Yhat = [cCo] X + [cDo] (c^kbar yq)
                rho  = c^(kbar+1) yq - Yhat             (integer residual)
  actuator      u = U / (c^(kbar+1) cu)
Attacks act on the integers: sensor attack added to yq, actuator attack added to U.
"""
import numpy as np
import plant
from systems import LinearPaper, FourTank


class QuantizedLinear:
    """The linear plant of the paper with the paper's integer controller (c = 1, cs = 5, kp = 64)."""
    name = "Linear plant of the paper"
    plant_cls = LinearPaper
    cs, cu, c, kp = 5.0, 5.0, 1, 64
    M = dict(A=[[0, 0], [1, 0]], B=[[1], [-1]], C=[[-1, 1]], D=[[0]], Co=[[1, 0]], Do=[[0]])
    Ts = LinearPaper.Ts
    lwe = dict(v=64, q=16381, r=400, sigma=26.0, T=192)      # the paper's parameters (key refresh every T steps)

    def make_plant(self):
        return LinearPaper()


class QuantizedFourTank:
    """The nonlinear four-tank with an integer controller (c = 16, kp = 3), designed in loop.py."""
    name = "Nonlinear four-tank process"
    Ts = FourTank.Ts

    def __init__(self):
        import loop
        _, _, _, M = loop.design()
        self.M = {k: v.tolist() for k, v in M.items()}
        self.cs, self.cu, self.c, self.kp = loop.CY, loop.CU, loop.C_SCALE, loop.KP
        # toy LWE parameters (NOT secure): large q so that the scale growth c^kbar fits; T is a multiple of kp
        self.lwe = dict(v=loop.V, q=loop.Q_MOD, r=loop.R_SCALE, sigma=loop.SIGMA, T=33)

    offset_h2 = 0.0

    def make_plant(self):
        return FourTank(offset_h2=self.offset_h2)


def run(spec, n_steps, sensor_attack=None, actuator_attack=None):
    """Integer closed loop.  sensor_attack(k) -> int added to yq; actuator_attack(k) -> int added to U."""
    M = {k: np.array(v, dtype=np.int64) for k, v in spec.M.items()}
    c, kp, cs, cu = spec.c, spec.kp, spec.cs, spec.cu
    pl = spec.make_plant()
    X = np.zeros((M["A"].shape[0], 1), dtype=np.int64)
    log = {k: [] for k in ("y", "yq", "u", "rho", "Yhat", "rho_unscaled")}
    for k in range(n_steps):
        kb = k % kp
        if kb == 0:
            X = np.zeros_like(X)                                 # periodic reset to xss = 0
        y = pl.y()
        yq = int(np.round(cs * y)) + (sensor_attack(k) if sensor_attack else 0)
        Ys = c ** kb * yq
        U = int((M["C"] @ X)[0, 0] + M["D"][0, 0] * Ys) + (actuator_attack(k) if actuator_attack else 0)
        Yhat = int((M["Co"] @ X)[0, 0] + M["Do"][0, 0] * Ys)
        rho = c ** (kb + 1) * yq - Yhat
        u = U / (c ** (kb + 1) * cu)
        X = M["A"] @ X + M["B"] * Ys
        for key, val in zip(log, (y, yq, u, rho, Yhat, rho / c ** (kb + 1))):
            log[key].append(val)
        pl.apply(u)
    return {k: np.array(v) for k, v in log.items()}
