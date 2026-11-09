"""Step 1: the two plants with their observer-based controllers, in plain floating point.

No quantization, no encryption: only feedback and attack injections.
  * LinearPaper : the linear plant of the numerical section of the paper (unstable, integral action).
  * FourTank    : the nonlinear four-tank process of the KTH lab (plant.py), h2 controlled by u1.
Common interface: sys.y() measures, sys.apply(u) advances one sample.  The controller is
    xhat+ = Ac xhat + Bc y,   u = Cc xhat,   yhat = Co xhat + Do y,   residual rho = y - yhat.
"""
import numpy as np
from scipy.signal import place_poles
from scipy.linalg import solve_discrete_are
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
    def controller(cls, blend=0.5):
        """Pole placement as in the research code: state feedback + integrator on y, observer, and a
        blend of measurement and prediction in the residual."""
        A, B, C, Ts = cls.A, cls.B, cls.C, cls.Ts
        n = A.shape[0]
        Aa = np.block([[A, np.zeros((n, 1))], [-Ts * C, np.eye(1)]])
        Ba = np.vstack([B, np.zeros((1, 1))])
        K = place_poles(Aa, Ba, [0.85, 0.75, 0.8]).gain_matrix
        Kx, Ki = K[:, :n], K[:, -1:]
        L = place_poles(A.T, C.T, [0.01, 0.05]).gain_matrix.T
        Ac = np.block([[A - L @ C - B @ Kx, -B @ Ki], [np.zeros((1, n)), np.eye(1)]])
        Bc = np.vstack([L, [[-Ts]]])
        Cc = -np.hstack([Kx, Ki])
        Co = (1 - blend) * np.hstack([C, np.zeros((1, 1))])
        return dict(A=Ac, B=Bc, C=Cc, Co=Co, Do=np.array([[blend]]))


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

    @classmethod
    def controller(cls):
        h0, u0 = plant.operating_point()
        A, B, C = plant.linearize_siso(h0, u0, cls.Ts)
        R = np.array([[20.0]])
        X = solve_discrete_are(A, B, np.diag([1.0, 0.1]), R)
        K = np.linalg.solve(R + B.T @ X @ B, B.T @ X @ A)
        Y = solve_discrete_are(A.T, C.T, 0.05 * np.eye(2), np.array([[0.25]]))
        L = A @ Y @ C.T @ np.linalg.inv(C @ Y @ C.T + 0.25)
        return dict(A=A - L @ C - B @ K, B=L, C=-K, Co=C, Do=np.zeros((1, 1)))


def simulate(system, ctrl, n_steps, sensor_attack=None, actuator_attack=None):
    """Closed loop.  sensor_attack(k) is added to the measurement, actuator_attack(k) to the control signal."""
    n = ctrl["A"].shape[0]
    xh = np.zeros((n, 1))
    log = {k: [] for k in ("y", "y_attacked", "u", "u_applied", "yhat", "rho")}
    for k in range(n_steps):
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
