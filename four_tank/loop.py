"""Encrypted control loop around the nonlinear four-tank process (Section III-VI of the paper).

Sensor side : y -> yq = round(cy * (h1 - h1_0)) -> LWE encryption.
Controller  : integer controller of eq. (23) on ciphertexts, with periodic state reset every kp steps;
              outputs the encrypted input U and the encrypted prediction Yhat; residual R = c^(kbar+1) Y - Yhat.
Actuator    : decrypts U and applies v1 = v1_0 + u / (cu * c^(kbar+1)).
The key is refreshed every T steps; one detection window = one key epoch.
"""
import numpy as np
from scipy.linalg import solve_discrete_are
from sympy import nextprime
import plant, lwe

Ts = 2.0                     # sample time [s]
CY, CU = 1.0, 100.0          # output scaling (deadband +-0.5 cm) and input scaling (0.01 V steps)
C_SCALE, KP = 16, 3          # controller scaling factor c and reset period kp
V, Q_MOD, R_SCALE, SIGMA = 8, int(nextprime(1 << 40)), 1 << 18, 2.0   # toy LWE parameters (NOT secure)
T_EPOCH = 32                 # key refresh period = detection window N (multiple of KP is not required)


def design():
    h0, u0 = plant.operating_point()
    A, B, C = plant.linearize_siso(h0, u0, Ts)       # states (h2, h4), output h2
    Rr = np.array([[20.0]])
    X = solve_discrete_are(A, B, np.diag([1.0, 0.1]), Rr)
    K = np.linalg.solve(Rr + B.T @ X @ B, B.T @ X @ A)
    Y = solve_discrete_are(A.T, C.T, 0.05 * np.eye(2), np.array([[0.25]]))
    L = A @ Y @ C.T @ np.linalg.inv(C @ Y @ C.T + 0.25)
    real = dict(A=A - L @ C - B @ K, B=L, C=-(CU / CY) * K, D=np.zeros((1, 1)), Co=C, Do=np.zeros((1, 1)))
    ints = {k: np.rint(C_SCALE * M).astype(np.int64) for k, M in real.items()}
    return h0, u0, real, ints


class Loop:
    def __init__(self, seed=0, sensor_noise=0.0):
        self.rng = np.random.default_rng(seed)
        self.h0, self.u0, self.real, self.M = design()
        self.n = self.M['A'].shape[0]
        self.lwe = lwe.LWE(V, Q_MOD, R_SCALE, SIGMA, self.rng)
        self.h = self.h0.copy()
        self.sensor_noise = sensor_noise
        self.k = 0
        self.X = np.zeros((self.n, V + 1), dtype=np.int64)       # trivial encryption of the reset state
        self.Xp = np.zeros((self.n, 1))                          # plaintext shadow of the state
        self.epoch = []                                          # residual ciphertexts of the current epoch

    def step(self, sensor_attack=0, actuator_attack=0.0):
        """One sampling period.  sensor_attack is added to the plaintext measurement (integer),
        actuator_attack to the applied voltage [V].  Returns a dict of what happened."""
        kb = self.k % KP
        L = self.lwe
        if kb == 0:                                              # periodic reset of the controller state
            self.X = np.zeros((self.n, V + 1), dtype=np.int64)
            self.Xp = np.zeros((self.n, 1))
        y = self.h[1] - self.h0[1] + self.rng.normal(0.0, self.sensor_noise)
        yq = int(np.rint(CY * y))
        Ye = L.encrypt([yq])
        if sensor_attack:                                        # additive attack on the ciphertext
            Ye = Ye.copy(); Ye[0, -1] = (Ye[0, -1] + L.r * sensor_attack) % L.q
        M, c = self.M, C_SCALE
        Ys = (c ** kb * Ye) % L.q                                # c^kbar * Y
        U = (L.lin(M['C'], self.X) + L.lin(M['D'], Ys)) % L.q
        Yhat = (L.lin(M['Co'], self.X) + L.lin(M['Do'], Ys)) % L.q
        Xn = (L.lin(M['A'], self.X) + L.lin(M['B'], Ys)) % L.q
        Rho = (c ** (kb + 1) * Ye - Yhat) % L.q                  # encrypted residual
        u = L.decrypt(U)[0] / c ** (kb + 1)                      # actuator decrypts the input
        # plaintext shadow of the controller (checks the homomorphic evaluation)
        ysp = c ** kb * (yq + sensor_attack)
        Yhat_p = (M['Co'] @ self.Xp + M['Do'] * ysp)[0, 0]
        rho_p = c ** (kb + 1) * (yq + sensor_attack) - Yhat_p
        self.Xp = M['A'] @ self.Xp + M['B'] * ysp
        self.X = Xn
        u1 = self.u0[0] + u / CU + actuator_attack
        self.h = plant.step(self.h, np.array([u1, self.u0[1]]), Ts)
        self.epoch.append(Rho)
        self.k += 1
        return dict(y=y, yq=yq, u=u, rho_dec=int(L.decrypt(Rho)[0]), rho_plain=float(rho_p))

    def end_epoch(self):
        """Return the encrypted residuals of the finished epoch and refresh the key."""
        R = np.concatenate(self.epoch, axis=0)
        self.epoch = []
        self.lwe.new_key()
        return R

    def noise_map(self, n):
        """Matrix T (n x n) with e_rho = T e_y: how the sensors' encryption noises e_y reach the residual
        noise of one epoch (this is T_c^T of the paper).  Sigma_rho = SIGMA^2 T T^T."""
        A, B, Co = (self.M[k].astype(float) for k in ('A', 'B', 'Co'))
        T = np.zeros((n, n))
        for j in range(n):
            x = np.zeros((self.n, 1))
            for k in range(n):
                kb = k % KP
                if kb == 0:
                    x = np.zeros((self.n, 1))
                e = 1.0 if k == j else 0.0
                T[k, j] = C_SCALE ** (kb + 1) * e - (Co @ x)[0, 0]
                x = A @ x + B * (C_SCALE ** kb * e)
        return T
