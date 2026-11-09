"""Step 2: both plants with quantized signals and the integer controller, with attack injections.

For the linear plant of the paper the nominal part (first quarter of the run) is checked against the saved run
of the research code: the plant output, the prediction and the residual must agree exactly.
Run:  python step2_quantization.py     writes step2_quantization.pdf / .png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from quantized import QuantizedLinear, QuantizedFourTank, run

CASES = [  # spec, steps, sensor attack range (integers), actuator attack range (integers in U)
    (QuantizedLinear(), 6144, 5, 5),
    (QuantizedFourTank(), 1024, 1, 25),
]


def attack_windows(spec, n, s_rng, a_rng, seed=0):
    rng = np.random.default_rng(seed)
    sens = {k: int(rng.integers(-s_rng, s_rng + 1)) for k in range(n // 4 + 1, n // 2)}
    act = {k: int(rng.integers(-a_rng, a_rng + 1)) * spec.c ** (k % spec.kp + 1) for k in range(3 * n // 4 + 1, n)}
    return sens, act


def check_against_saved(log):
    path = os.path.join(os.path.dirname(__file__), "..", "fig7", "windowed_det_data.npz")
    D = np.load(path)
    m = 1536                                       # first quarter: no attack
    dy = np.abs(log["y"][:m] - np.array(D["y"]).ravel()[:m]).max()
    dr = np.abs(log["rho"][:m] - np.array(D["rdec"]).ravel()[:m]).max()
    dh = np.abs(log["Yhat"][:m] - np.array(D["yhat_e"]).ravel()[:m]).max()
    print(f"check against the saved run (first {m} steps): max |dy| = {dy:g}, max |drho| = {dr:g}, max |dyhat| = {dh:g}")
    assert dy == 0 and dr == 0 and dh == 0, "quantized linear loop does not reproduce the saved run"


if __name__ == "__main__":
    fig, ax = plt.subplots(3, 2, figsize=(7.2, 5.6))
    for j, (spec, n, s_rng, a_rng) in enumerate(CASES):
        sens, act = attack_windows(spec, n, s_rng, a_rng)
        nom = run(spec, n)
        att = run(spec, n, sensor_attack=lambda k: sens.get(k, 0), actuator_attack=lambda k: act.get(k, 0))
        if j == 0:
            check_against_saved(nom)
        t = np.arange(n) * spec.Ts
        nz = np.nonzero(nom["rho"])[0]
        print(f"{spec.name}: nominal nonzero residual samples {len(nz)} (first steps {nz[:6].tolist()}); "
              f"attacked: max |rho| (unscaled) = {np.abs(att['rho_unscaled']).max():.3g}")
        for i, (key, lab) in enumerate((("y", "output y"), ("u", "applied input"), ("rho_unscaled", "residual (unscaled)"))):
            a = ax[i, j]
            a.plot(t, nom[key], color="#888888", lw=0.7, label="nominal")
            a.plot(t, att[key], color="#0072b2", lw=0.7, label="attacked")
            a.axvspan((n // 4) * spec.Ts, (n // 2) * spec.Ts, color="#e69f00", alpha=0.15, lw=0)
            a.axvspan((3 * n // 4) * spec.Ts, n * spec.Ts, color="#d55e00", alpha=0.15, lw=0)
            a.set_ylabel(lab); a.grid(alpha=0.3)
        ax[0, j].set_title(spec.name, fontsize=9); ax[2, j].set_xlabel("time [s]")
    ax[0, 0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.text(0.5, 0.005, "shaded: sensor attack (orange), actuator attack (red)", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig("step2_quantization.pdf"); fig.savefig("step2_quantization.png", dpi=150)
