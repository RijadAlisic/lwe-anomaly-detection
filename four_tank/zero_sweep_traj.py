"""Trajectories of the zero-dynamics sweep (seed 0): output y, applied input u, residual (encrypted residual as gray
background, red ticks = decryption errors), hidden state x2, and the detector's alarm per window (from zero_sweep.json).
Writes traj_A_<kind>.png and traj_B_<kind>.png.  Run with the research venv python."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import encrypted
import zero_attack as za
from zero_sweep import attack_fn, T

R = json.load(open("det_results/zero_sweep.json"))
FLAGS = {(r["kind"], r["scen"], r["xi"], r["par"]): [w["flag"] for w in r["wins"]] for r in R if r["seed"] == 0}
COL = "#0072b2"


def run(kind, scen, par):
    if scen == "A":
        n, x0, xi_att, k0 = 3 * T, 0.7, None if par is None else par * 0.7, 0
        spec = za.ZeroSpec(kind, x0, xi_att)
    else:
        n, x0, xi_att, k0 = 4 * T, 0.0, par, T
        spec = za.ZeroSpec(kind, x0, xi_att)
    act = attack_fn(spec, xi_att, k0, T)
    out = encrypted.run(spec, n, actuator_attack=act, seed=0)
    pl = spec.make_plant(); xs = []
    for k in range(n):
        xs.append(pl.x[:, 0].copy()); pl.apply(out["u"][k])
    return out, np.array(xs), n, k0, xi_att is not None


def figure(kind, scen, pars, fname, title):
    fig, axes = plt.subplots(5, len(pars), figsize=(3.4 * len(pars), 9), sharex="col",
                             gridspec_kw=dict(height_ratios=[3, 3, 3, 3, 0.8]))
    for j, par in enumerate(pars):
        out, x, n, k0, attacked = run(kind, scen, par)
        t = np.arange(n); q = out["q"]
        xi = 0.7 if scen == "A" else 0.0
        flags = FLAGS[(kind, scen, xi, par)]
        for i, (sig, lab) in enumerate(((out["y"], "output y"), (out["u"], "applied input u"), (None, "residual"), (x[:, 1], "hidden state x2"))):
            ax = axes[i, j]; ax.grid(alpha=0.3)
            if attacked:
                ax.axvspan(k0, k0 + T, color="#d55e00", alpha=0.08, lw=0)
            if i == 2:
                ax.plot(t, out["enc_rho"], color="#8c8c8c", lw=0.4, alpha=0.5); ax.set_ylim(-q / 2, q / 2); ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel("encrypted", color="#8c8c8c", fontsize=8)
                pax = ax.twinx(); pax.set_zorder(ax.get_zorder() + 1); pax.patch.set_visible(False)
                pax.plot(t, out["rho"], color=COL, lw=0.9); pax.set_ylim(-1.6, 1.6)
                es = np.asarray(out["error_steps"], dtype=int)
                if len(es):
                    pax.plot(es, np.full(len(es), -1.45), "|", color="red", ms=5)
                if j == len(pars) - 1:
                    pax.set_ylabel("residual (plaintext)")
            else:
                ax.plot(t, sig, color=COL, lw=0.9)
                if j == 0:
                    ax.set_ylabel(lab)
        sax = axes[4, j]; sax.set_ylim(0, 1); sax.set_yticks([]); sax.set_xlim(0, n)
        for w, f in enumerate(flags):
            sax.add_patch(plt.Rectangle((w * T, 0.1), T, 0.8, facecolor="#d55e00" if f else "#009e73", edgecolor="white"))
        if j == 0:
            sax.set_ylabel("alarm", fontsize=8)
        sax.set_xlabel("time step")
        lab = "no attack" if par is None else (f"guess g = {par}" if scen == "A" else f"eps = {par}")
        axes[0, j].set_title(lab, fontsize=10)
    fig.suptitle(title, fontsize=10, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(fname, dpi=120); plt.close(fig)


for kind in ("stable", "unstable"):
    figure(kind, "A", [None, 1.0, 0.8, 0.5], f"traj_A_{kind}.png",
           f"A, {kind} zero: zero-dynamics attack on window 0 (shaded), attacker guesses g*xi, xi = 0.7; windows 1-2 after the attack")
    figure(kind, "B", [None, 0.1, 0.2, 0.4], f"traj_B_{kind}.png",
           f"B, {kind} zero: attack from rest, window 1 (shaded), amplitude eps; windows 2-3 after the attack")
print("ok")
