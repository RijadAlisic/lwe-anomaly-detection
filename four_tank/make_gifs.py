"""Animated plots, one gif per controller version, in the order nominal -> quantized -> (encrypted -> detected).

Every gif has the same layout and the same attack storyline (same scenario, same axis limits per plant), so
consecutive gifs can be compared by flipping through them.  Usage:  python make_gifs.py [--frames 100] [--seconds 10]
Writes gifs/<plant>_<number>_<mode>.gif
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import experiment as ex

TITLES = {"linear": "Linear plant of the paper", "fourtank": "Nonlinear four-tank process"}
MODE_LABEL = {"nominal": "nominal control (floating point)", "quantized": "quantized control (integer controller)",
              "encrypted": "encrypted control (LWE ciphertexts)",
              "detected": "encrypted control with anomaly detection"}
ENC_KEY = {"y": "enc_y", "u": "enc_u", "rho": "enc_rho"}
ROWS = (("y", "output y"), ("u", "applied input u"), ("rho", "residual"))
UNITS = {"linear": ("", "", ""), "fourtank": (" [cm]", " [V]", " [cm]")}
TIME = {"linear": "time [s]", "fourtank": "time [s]"}


def shared_limits(plant):
    """Axis limits from all modes, so that the gifs of one plant are directly comparable."""
    logs = [ex.run(plant, m) for m in ex.MODES if m != "detected"]
    lim = {}
    for key, _ in ROWS:
        lo = min(l[key].min() for l in logs); hi = max(l[key].max() for l in logs)
        pad = 0.08 * (hi - lo + 1e-9)
        lim[key] = (lo - pad, hi + pad)
    return lim


def make_gif(plant, mode, number, lim, frames, outdir, seconds=10.0):
    L = ex.run(plant, mode)
    t, Ts = L["t"], L["Ts"]
    detected = mode == "detected"
    if detected:                                         # a fourth row for the detection flags
        fig, axes = plt.subplots(4, 1, figsize=(6.4, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 3, 3, 1.1]))
        sax, axes = axes[3], axes[:3]
    else:
        fig, axes = plt.subplots(3, 1, figsize=(6.4, 5.4), sharex=True)
    lines, enc_lines = [], []
    encrypted = "enc_y" in L
    for ax, (key, lab), unit in zip(axes, ROWS, UNITS[plant]):
        ax.axvspan(L["sensor_window"][0] * Ts, L["sensor_window"][1] * Ts, color="#e69f00", alpha=0.15, lw=0)
        ax.axvspan(L["actuator_window"][0] * Ts, L["actuator_window"][1] * Ts, color="#d55e00", alpha=0.15, lw=0)
        ax.set_xlim(t[0], t[-1] + Ts); ax.grid(alpha=0.3)
        pax = ax
        if encrypted:                                   # encrypted values as a gray background (left), plaintext on top (right)
            half = L["q"] / 2
            (eln,) = ax.plot([], [], color="#8c8c8c", lw=0.5, alpha=0.4)
            ax.set_ylim(-half, half); ax.set_yticks([]); ax.set_ylabel("encrypted", color="#8c8c8c", fontsize=8)
            enc_lines.append((eln, ENC_KEY[key]))
            pax = ax.twinx(); pax.set_zorder(ax.get_zorder() + 1); pax.patch.set_visible(False)
        (ln,) = pax.plot([], [], color="#0072b2", lw=0.8)
        (dot,) = pax.plot([], [], "o", color="#0072b2", ms=3.5)
        pax.set_ylim(*lim[key]); pax.set_ylabel(lab + unit)
        if encrypted:
            pax.yaxis.set_label_position("right")
        lines.append((ln, dot, key))
    fig.suptitle(f"{TITLES[plant]}: {MODE_LABEL[mode]}", fontsize=10, weight="bold", x=0.02, ha="left")
    tr = axes[0].get_xaxis_transform()                  # x in data, y as fraction of the axis height
    axes[0].text(0.5 * (L["sensor_window"][0] + L["sensor_window"][1]) * Ts, 0.97, "sensor attack", ha="center",
                 va="top", fontsize=7, color="#a06a00", transform=tr, zorder=10)
    axes[0].text(0.5 * (L["actuator_window"][0] + L["actuator_window"][1]) * Ts, 0.97, "actuator attack", ha="center",
                 va="top", fontsize=7, color="#a03a00", transform=tr, zorder=10)
    if detected:
        d = L["det"]; Tw = int(d["T"]) * Ts; nw = len(d["window"])
        patches = []
        for w in range(nw):
            patch = plt.Rectangle((w * Tw, 0.12), Tw, 0.76, facecolor="#ececec", edgecolor="white", lw=1.0)
            sax.add_patch(patch); patches.append(patch)
        sax.set_ylim(0, 1); sax.set_yticks([]); sax.set_ylabel("detection", rotation=0, ha="right", va="center", fontsize=9)
        sax.set_xlabel(TIME[plant]); sax.tick_params(axis="x", labelbottom=True)
        sax.text(1.01, 0.72, "alarm", transform=sax.transAxes, color="#d55e00", fontsize=8, va="center")
        sax.text(1.01, 0.28, "no alarm", transform=sax.transAxes, color="#009e73", fontsize=8, va="center")
        sax.text(1.01, 0.5, "", transform=sax.transAxes)
    else:
        axes[2].set_xlabel(TIME[plant])
    if encrypted:                                       # what one sample costs, for this parameter set
        tm = L["timing"]
        fig.text(0.13, 0.012, f"LWE n=64, q=2^23+9, sigma=50 | per sample (Python, 1 core): encrypt {tm['encrypt']:.2f} ms, "
                 f"control {tm['evaluate']:.2f} ms, decrypt {tm['decrypt']:.2f} ms", fontsize=6, color="#555555")
    if detected:
        fig.text(0.13, 0.038, f"detection per window of {int(d['T'])} samples (kernel lattice + LLL, Python, 1 core): "
                 f"{np.mean(d['seconds']):.0f} s on average, {int(d['K'])}-vector chi-square test at alpha = {float(d['alpha']):g}",
                 fontsize=6, color="#555555")
    fig.subplots_adjust(left=0.13, right=0.86, top=0.93, bottom=0.17 if detected else 0.13, hspace=0.12)
    idx = np.unique(np.linspace(1, len(t), frames).astype(int))
    seq = [("draw", int(i)) for i in idx]
    if detected:     # after the signals are drawn, the detection flags fill in window by window
        per = max(1, int(round(30 / nw)))
        seq += [("det", w) for w in range(nw) for _ in range(per)] + [("det", nw - 1)] * 15

    def draw(f):
        kind, i = f
        if kind == "draw":
            n = i
        else:
            n = len(t)
            for w in range(i + 1):
                patches[w].set_facecolor("#d55e00" if d["flag_chi"][w] else "#009e73")
        for ln, dot, key in lines:
            ln.set_data(t[:n], L[key][:n]); dot.set_data([t[n - 1]], [L[key][n - 1]])
        for eln, ek in enc_lines:
            eln.set_data(t[:n], L[ek][:n])
        return [x for ln, dot, _ in lines for x in (ln, dot)] + [eln for eln, _ in enc_lines]

    anim = FuncAnimation(fig, draw, frames=seq, blit=not detected)
    path = os.path.join(outdir, f"{plant}_{number}_{mode}.gif")
    anim.save(path, writer=PillowWriter(fps=len(idx) / seconds))
    plt.close(fig)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=10.0, help="duration of each gif")
    a = ap.parse_args()
    os.makedirs("gifs", exist_ok=True)
    for plant in ("linear", "fourtank"):
        lim = shared_limits(plant)
        for number, mode in enumerate(ex.MODES, start=1):
            p = make_gif(plant, mode, number, lim, a.frames, "gifs", a.seconds)
            print(p, f"{os.path.getsize(p) / 1e3:.0f} kB")
