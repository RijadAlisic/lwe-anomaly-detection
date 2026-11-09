"""Step 1: both plants in closed loop, nominal and under sensor / actuator attack injections (no quantization,
no encryption).  The attack windows follow the paper: sensor attack in the second quarter of the run and
actuator attack in the last quarter, each a random integer in [-5, 5] times a unit (four-tank actuator: 0.2 V).
Run:  python step1_feedback_attacks.py      writes step1_feedback_attacks.pdf / .png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from systems import LinearPaper, FourTank, simulate

CASES = [  # plant class, controller kwargs, steps, sensor unit, actuator unit
    (LinearPaper, dict(), 6144, 1 / 5.0, 1 / 5.0),      # unit = 1/cs: one quantization step
    (FourTank, dict(), 1024, 0.2, 0.2),                            # cm and V
]


def windows(n):
    return (n // 4, n // 2), (3 * n // 4, n)


def run(cls, ckw, n, su, au, seed=0):
    rng = np.random.default_rng(seed)
    (s0, s1), (a0, a1) = windows(n)
    sens = {k: su * rng.integers(-5, 6) for k in range(s0, s1)}
    act = {k: au * rng.integers(-5, 6) for k in range(a0, a1)}
    ctrl = cls.controller(**ckw)
    nominal = simulate(cls(), ctrl, n)
    attacked = simulate(cls(), ctrl, n, sensor_attack=lambda k: sens.get(k, 0.0), actuator_attack=lambda k: act.get(k, 0.0))
    return nominal, attacked, (s0, s1), (a0, a1)


if __name__ == "__main__":
    fig, ax = plt.subplots(3, 2, figsize=(7.2, 5.6))
    for j, (cls, ckw, n, su, au) in enumerate(CASES):
        nom, att, sw, aw = run(cls, ckw, n, su, au)
        t = np.arange(n) * cls.Ts
        print(f"{cls.name}: max |y| nominal {np.abs(nom['y']).max():.3f}, attacked {np.abs(att['y']).max():.3f}; "
              f"max |residual| nominal {np.abs(nom['rho']).max():.4f}, attacked {np.abs(att['rho']).max():.3f}")
        for i, (key, lab) in enumerate((("y", "output y"), ("u_applied", "applied input"), ("rho", "residual"))):
            a = ax[i, j]
            a.plot(t, nom[key], color="#888888", lw=0.7, label="nominal")
            a.plot(t, att[key], color="#0072b2", lw=0.7, label="attacked")
            a.axvspan(sw[0] * cls.Ts, sw[1] * cls.Ts, color="#e69f00", alpha=0.15, lw=0)
            a.axvspan(aw[0] * cls.Ts, aw[1] * cls.Ts, color="#d55e00", alpha=0.15, lw=0)
            a.set_ylabel(lab); a.grid(alpha=0.3)
        ax[0, j].set_title(cls.name, fontsize=9); ax[2, j].set_xlabel("time [s]")
    ax[0, 0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.text(0.5, 0.005, "shaded: sensor attack (orange), actuator attack (red)", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig("step1_feedback_attacks.pdf"); fig.savefig("step1_feedback_attacks.png", dpi=150)
