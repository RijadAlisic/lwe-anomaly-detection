"""Step 1: the two plants with their observer-based controllers, in plain floating point.

No quantization, no encryption: only feedback and attack injections.
  * LinearPaper : the linear plant of the numerical section of the paper (unstable, integral action).
  * FourTank    : the nonlinear four-tank process of the KTH lab (plant.py), h2 controlled by u1.
Common interface: sys.y() measures, sys.apply(u) advances one sample.  The controller is
    xhat+ = Ac xhat + Bc y,   u = Cc xhat,   yhat = Co xhat + Do y,   residual rho = y - yhat.
"""
import numpy as np
import plant


class LinearPaper:
    name = "Linear plant of the paper"
    Ts = 1.0
    cs = 5.0                                      # quantization gain of the paper (used from step 2 on)
    A = np.array([[1.001, 0.4], [0.0, 0.1]])
    B = np.array([[0.0], [0.4722]])
    C = np.array([[1.0, 0.0]])

    def __init__(self):
        self.x = np.array([[0.1], [0.0]])         # initial state of the paper's example

    def y(self):
        return float((self.C @ self.x)[0, 0])

    def apply(self, u):
        self.x = self.A @ self.x + self.B * u

    @classmethod
    def controller(cls):
        """The controller actually used in the paper's run (research code, V1_enc_loop.py): a small integer
        controller on the quantized signals, X+ = [[0,0],[1,0]] X + [1,-1]^T yq, U = [-1,1] X, yhat = [1,0] X,
        with c = 1.  Because yq = cs*y and u = U/cs, the scale cs cancels and the same matrices act on the
        unquantized signals (step 1); step 2 adds the quantization."""
        return dict(A=np.array([[0.0, 0.0], [1.0, 0.0]]), B=np.array([[1.0], [-1.0]]),
                    C=np.array([[-1.0, 1.0]]), Co=np.array([[1.0, 0.0]]), Do=np.zeros((1, 1)))


class FourTank:
    name = "Nonlinear four-tank process"
    Ts = 2.0

    def __init__(self, offset_h2=1.0):
        self.h0, self.u0 = plant.operating_point()
        self.h = self.h0.copy()
        self.h[1] += offset_h2                    # start 1 cm above the equilibrium

    def y(self):
        return float(self.h[1] - self.h0[1])

    def apply(self, u):
        self.h = plant.step(self.h, self.u0 + np.array([u, 0.0]), self.Ts)

    cs, cu = 5.0, 10.0                            # output gain (0.2 cm steps) and input gain (0.1 V steps)

    @classmethod
    def controller(cls):
        """Same structure as the paper's integer controller (c = 1): u = (y[k-2] - 2 y[k-1]) / cu, acting on
        yq = cs * y, so on the real signals the gain is cs/cu.  Stabilizes the four-tank with a settling time
        of about 7 steps."""
        return dict(A=np.array([[0.0, 0.0], [1.0, 0.0]]), B=np.array([[1.0], [-1.0]]),
                    C=(cls.cs / cls.cu) * np.array([[-1.0, 1.0]]), Co=np.array([[1.0, 0.0]]), Do=np.zeros((1, 1)))


def simulate(system, ctrl, n_steps, sensor_attack=None, actuator_attack=None):
    """Closed loop.  sensor_attack(k) is added to the measurement, actuator_attack(k) to the control signal."""
    n = ctrl["A"].shape[0]
    xh = np.zeros((n, 1))
    log = {k: [] for k in ("y", "y_attacked", "u", "u_applied", "yhat", "rho")}
    if hasattr(system, "h"):
        log["h"] = []                                     # tank levels [cm], for the four-tank illustration
    for k in range(n_steps):
        if "h" in log:
            log["h"].append(system.h.copy())
        y = system.y()
        ya = y + (sensor_attack(k) if sensor_attack else 0.0)
        u = float((ctrl["C"] @ xh)[0, 0])
        yhat = float((ctrl["Co"] @ xh)[0, 0] + ctrl["Do"][0, 0] * ya)
        ua = u + (actuator_attack(k) if actuator_attack else 0.0)
        for key, val in zip(log, (y, ya, u, ua, yhat, ya - yhat)):
            log[key].append(val)
        system.apply(ua)
        xh = ctrl["A"] @ xh + ctrl["B"] * ya
    return {k: np.array(v) for k, v in log.items()}
