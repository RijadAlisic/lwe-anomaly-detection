"""Nonlinear four-tank process, minimum-phase setting of the KTH water-tank lab.

Parameters and equations are taken from the lab's MATLAB sources
(water_tank_lab_2019/src/SimPara.m, +tankFcn/FreshPara.m and tankClass/Model.m):
  states h1..h4 [cm]; inputs u1, u2 [V, 0..3] (the pump power module multiplies by 15/3);
  dh1 = -alpha1 sqrt(2g h1) + alpha13 sqrt(2g h3) + beta1  u1
  dh2 = -alpha2 sqrt(2g h2) + alpha24 sqrt(2g h4) + beta2  u2
  dh3 = -alpha3 sqrt(2g h3)                        + beta32 u2
  dh4 = -alpha4 sqrt(2g h4)                        + beta41 u1
We control h2 with u1 (SISO, second order: u1 -> h4 -> h2); u2 is held at its nominal value.
"""
import numpy as np
from scipy.optimize import fsolve
from scipy.signal import cont2discrete

A_T = 15.5179                                        # tank cross section [cm^2]
K1, K2 = 4.32, 3.74                                  # pump gains [cm^3/(V s)]
A1, A2, A3, A4 = 0.1678, 0.1542, 0.06743, 0.06504    # outlet areas [cm^2]
GAM1 = GAM2 = 0.625                                  # valve settings
G = 981.0
GAIN = 15.0 / 3.0                                    # power module: 0..3 V in, 0..15 V pump voltage
al = np.array([A1, A2, A3, A4]) / A_T
b1, b2 = GAM1 * K1 / A_T * GAIN, GAM2 * K2 / A_T * GAIN
b32, b41 = (1 - GAM2) * K2 / A_T * GAIN, (1 - GAM1) * K1 / A_T * GAIN
H_NOM = np.array([18.0, 18.0, 14.0, 19.0])           # linearization point of the lab [cm]


def f(h, u):
    s = np.sqrt(2 * G * np.maximum(h, 0.0))
    return np.array([-al[0] * s[0] + al[2] * s[2] + b1 * u[0],
                     -al[1] * s[1] + al[3] * s[3] + b2 * u[1],
                     -al[2] * s[2] + b32 * u[1],
                     -al[3] * s[3] + b41 * u[0]])


def step(h, u, Ts, n_sub=20):
    """Advance the plant by Ts seconds with RK4 (zero-order-hold input, voltage limited to 0..3 V)."""
    u = np.clip(u, 0.0, 3.0)
    dt = Ts / n_sub
    for _ in range(n_sub):
        c1 = f(h, u); c2 = f(h + dt / 2 * c1, u); c3 = f(h + dt / 2 * c2, u); c4 = f(h + dt * c3, u)
        h = h + dt / 6 * (c1 + 2 * c2 + 2 * c3 + c4)
    return h


def operating_point():
    """Exact equilibrium near the lab's linearization point: inputs from h3, h4, then h solved from f(h, u) = 0."""
    s = np.sqrt(2 * G * H_NOM)
    u = np.array([al[3] * s[3] / b41, al[2] * s[2] / b32])
    h = fsolve(lambda x: f(x, u), H_NOM)
    return h, u


def linearize_siso(h0, u0, Ts):
    """Discretized model of u1 -> h2 on the controllable and observable states (h2, h4)."""
    n = 4
    A = np.zeros((n, n)); B = np.zeros((n, 1)); eps = 1e-6
    for i in range(n):
        dh = np.zeros(n); dh[i] = eps
        A[:, i] = (f(h0 + dh, u0) - f(h0 - dh, u0)) / (2 * eps)
    B[:, 0] = (f(h0, u0 + np.array([eps, 0])) - f(h0, u0 - np.array([eps, 0]))) / (2 * eps)
    idx = [1, 3]                                      # states (h2, h4)
    A, B = A[np.ix_(idx, idx)], B[idx]
    C = np.array([[1.0, 0.0]])
    Ad, Bd, Cd, _, _ = cont2discrete((A, B, C, np.zeros((1, 1))), Ts)
    return Ad, Bd, Cd
