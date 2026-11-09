"""Zero-dynamics attack (a stealthy attack) on a modified linear plant with a stable or an unstable zero.

The paper's linear plant has relative degree 2 and no zero.  Here the input matrix is changed to B = [b1, 0.4722]^T,
b1 = -0.4*0.4722/(z0 - 0.1), which puts one transmission zero at z0 (C = [1 0] as before):
    A = [[1.001, 0.4], [0, 0.1]],  zero z0 < 1: stable (harmless attack),  z0 > 1: unstable (the hidden state grows).
A zero-dynamics attack starts from the state (0, xi) and injects u_a(k) = -(0.4/b1) xi z0^k; the output stays exactly
zero, so the controller and the residual never see it.  Matched: the plant really is in the state (0, xi) when the
attack starts.  Mismatched: the attacker guessed another amplitude, the output leaves zero, the residual becomes nonzero.
Without the attack the same initial state produces a transient (y = 0 at k = 0, then y(1) = 0.4 xi).
The controller is an integer FIR output feedback of the same structure as in the paper (X = past outputs, kp = 64).
"""
import numpy as np
from quantized import COMPACT

B2 = 0.4722
CASES = {   # zero, controller gains (D, a, b) on yq, cs, cu
    "stable": dict(z0=0.7, gains=(2, 1, 3), cs=5.0, cu=10.0),
    "unstable": dict(z0=1.02, gains=(-5, -6, 4), cs=5.0, cu=10.0),
}


class ZeroPlant:
    A = np.array([[1.001, 0.4], [0.0, 0.1]])
    C = np.array([[1.0, 0.0]])
    Ts = 1.0

    def __init__(self, z0, x0):
        self.z0 = z0
        self.B = np.array([[-0.4 * B2 / (z0 - 0.1)], [B2]])
        self.x = np.array([[float(x0[0])], [float(x0[1])]])

    def y(self):
        return float((self.C @ self.x)[0, 0])

    def apply(self, u):
        self.x = self.A @ self.x + self.B * u


class ZeroSpec:
    c, kp = 1, 64
    Ts = 1.0
    lwe = COMPACT

    def __init__(self, kind, xi_true, xi_attack=None, attack_len=None, amp=None):
        cfg = CASES[kind]
        self.kind, self.z0, self.cs, self.cu = kind, cfg["z0"], cfg["cs"], cfg["cu"]
        D, a, b = cfg["gains"]
        self.M = dict(A=[[0, 0], [1, 0]], B=[[1], [0]], C=[[a, b]], D=[[D]], Co=[[1, 0]], Do=[[0]])
        self.xi_true = xi_true
        self.xi_attack = xi_true if xi_attack is None else xi_attack
        self.b1 = -0.4 * B2 / (self.z0 - 0.1)

    def make_plant(self):
        return ZeroPlant(self.z0, (0.0, self.xi_true))

    def attack(self, n_on):
        """Integer actuator injection U_a(k) (units of 1/cu volts), zero after n_on steps (None: never stops)."""
        def f(k):
            if n_on is not None and k >= n_on:
                return 0
            return int(np.round(self.cu * (-0.4 / self.b1) * self.xi_attack * self.z0 ** k))
        return f
