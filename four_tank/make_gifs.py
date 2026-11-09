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
MODE_LABEL = {"nominal": "nominal control (floating point)", "quantized": "quantized control (integer controller)"}
ROWS = (("y", "output y"), ("u", "applied input u"), ("rho", "residual"))
UNITS = {"linear": ("", "", ""), "fourtank": (" [cm]", " [V]", " [cm]")}
TIME = {"linear": "time [s]", "fourtank": "time [s]"}


def shared_limits(plant):
    """Axis limits from all modes, so that the gifs of one plant are directly comparable."""
    logs = [ex.run(plant, m) for m in ex.MODES]
    lim = {}
    for key, _ in ROWS:
        lo = min(l[key].min() for l in logs); hi = max(l[key].max() for l in logs)
        pad = 0.08 * (hi - lo + 1e-9)
        lim[key] = (lo - pad, hi + pad)
    return lim


def make_gif(plant, mode, number, lim, frames, outdir, seconds=10.0):
    L = ex.run(plant, mode)
    t, Ts = L["t"], L["Ts"]
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 5.4), sharex=True)
    lines = []
    for ax, (key, lab), unit in zip(axes, ROWS, UNITS[plant]):
        ax.axvspan(L["sensor_window"][0] * Ts, L["sensor_window"][1] * Ts, color="#e69f00", alpha=0.15, lw=0)
        ax.axvspan(L["actuator_window"][0] * Ts, L["actuator_window"][1] * Ts, color="#d55e00", alpha=0.15, lw=0)
        (ln,) = ax.plot([], [], color="#0072b2", lw=0.8)
        (dot,) = ax.plot([], [], "o", color="#0072b2", ms=3.5)
        ax.set_xlim(t[0], t[-1] + Ts); ax.set_ylim(*lim[key]); ax.set_ylabel(lab + unit); ax.grid(alpha=0.3)
        lines.append((ln, dot, key))
    fig.suptitle(f"{TITLES[plant]}: {MODE_LABEL[mode]}", fontsize=10, weight="bold", x=0.02, ha="left")
    axes[0].text(0.5 * (L["sensor_window"][0] + L["sensor_window"][1]) * Ts, lim["y"][1], "sensor attack", ha="center",
                 va="top", fontsize=7, color="#a06a00")
    axes[0].text(0.5 * (L["actuator_window"][0] + L["actuator_window"][1]) * Ts, lim["y"][1], "actuator attack", ha="center",
                 va="top", fontsize=7, color="#a03a00")
    axes[2].set_xlabel(TIME[plant])
    fig.subplots_adjust(left=0.13, right=0.975, top=0.93, bottom=0.09, hspace=0.12)   # identical plot area in every gif
    idx = np.unique(np.linspace(1, len(t), frames).astype(int))

    def draw(i):
        for ln, dot, key in lines:
            ln.set_data(t[:i], L[key][:i]); dot.set_data([t[i - 1]], [L[key][i - 1]])
        return [x for ln, dot, _ in lines for x in (ln, dot)]

    anim = FuncAnimation(fig, draw, frames=idx, blit=True)
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
