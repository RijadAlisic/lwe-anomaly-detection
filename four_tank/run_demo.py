"""Detection demo on the encrypted four-tank loop: false alarms and detection rate versus attack size.

Every detection window is one key epoch of T_EPOCH samples.  For each window the script
  1. runs the plant with the encrypted controller (loop.py) under a given scenario,
  2. finds a short filtering vector in the q-ary kernel lattice of the residual ciphertexts (detector.py),
  3. applies the two-sided test of Theorem 1 at false-alarm level ALPHA.
Usage:  python run_demo.py [--windows 12] [--alpha 0.01] [--jobs 4]      (about 6 s of lattice reduction per window)
"""
import argparse, csv, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import loop, detector

SENSOR_BIAS = [1, 2, 3, 5]                       # added to the quantized measurement (units of cy^-1 cm)
ACTUATOR = [0.05, 0.1, 0.2, 0.3, 0.5]            # added to the pump voltage u1 [V]
ATTACK_START = 4                                  # attack active from this sample of the window


def run_window(args):
    kind, amp, seed, alpha = args
    lp = loop.Loop(seed=seed, sensor_noise=0.1)
    for k in range(loop.T_EPOCH):
        on = k >= ATTACK_START
        lp.step(sensor_attack=int(amp) if (kind == "sensor" and on) else 0,
                actuator_attack=float(amp) if (kind == "actuator" and on) else 0.0)
    Tm = lp.noise_map(loop.T_EPOCH)
    R = lp.end_epoch()
    P, b = R[:, :-1], R[:, -1]
    d = detector.short_filtering_vector(P, loop.Q_MOD)
    s2 = loop.SIGMA ** 2 * float(np.sum((Tm.T @ d.astype(float)) ** 2))
    X, gamma, alarm = detector.two_sided_test(d, b, loop.Q_MOD, s2, alpha)
    return dict(kind=kind, amp=amp, seed=seed, alarm=int(alarm), X=X, gamma=gamma, norm_d=float(np.linalg.norm(d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=12, help="windows per scenario (nominal gets 3x)")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    jobs = [("nominal", 0, s, a.alpha) for s in range(3 * a.windows)]
    jobs += [("sensor", v, 1000 + 100 * i + s, a.alpha) for i, v in enumerate(SENSOR_BIAS) for s in range(a.windows)]
    jobs += [("actuator", v, 5000 + 100 * i + s, a.alpha) for i, v in enumerate(ACTUATOR) for s in range(a.windows)]
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(run_window, jobs))
    print(f"{len(rows)} windows in {time.time() - t0:.0f} s")
    with open("results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    def rate(kind, amp):
        sel = [r["alarm"] for r in rows if r["kind"] == kind and r["amp"] == amp]
        return float(np.mean(sel)), len(sel)

    fa, n_nom = rate("nominal", 0)
    print(f"false-alarm rate (alpha = {a.alpha}): {fa:.3f} over {n_nom} nominal windows")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for axi, kind, amps, xl in ((ax[0], "sensor", SENSOR_BIAS, "sensor bias (quantization steps)"),
                                (ax[1], "actuator", ACTUATOR, "actuator injection [V]")):
        r = [rate(kind, v)[0] for v in amps]
        for v, x in zip(amps, r):
            print(f"{kind:9s} {v:5}: detection rate {x:.2f}")
        axi.plot(amps, r, "o-", color="#0072b2")
        axi.axhline(fa, color="gray", ls="--", lw=0.8, label="false-alarm rate")
        axi.set_xlabel(xl); axi.grid(alpha=0.3)
    ax[0].set_ylabel("alarm rate"); ax[0].legend(frameon=False); ax[0].set_ylim(-0.02, 1.02)
    fig.tight_layout(); fig.savefig("detection_vs_attack.pdf"); fig.savefig("detection_vs_attack.png", dpi=150)


if __name__ == "__main__":
    main()
