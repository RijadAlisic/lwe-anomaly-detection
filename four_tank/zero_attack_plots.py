"""Plots of the zero-dynamics study (reads det_results/zero_attack.npz, written by zero_attack_study.py).
One figure per zero: columns none / matched / mismatched attack; rows output y, applied input u, residual (encrypted
residual as gray background, plaintext on top, as in the gifs), hidden state x2.  Writes zero_attack_<kind>.png/pdf."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import zero_attack as za

D = np.load("det_results/zero_attack.npz")
COL = {"none": "#7f7f7f", "matched": "#d55e00", "mismatched": "#0072b2"}
TITLE = {"none": "no attack (initial transient only)", "matched": "matched zero-dynamics attack",
         "mismatched": "mismatched attack (guess = 0.5 x true state)"}


def get(kind, att, key):
    return D[f"{kind}_{att}_{key}"]


for kind in ("stable", "unstable"):
    fig, axes = plt.subplots(4, 3, figsize=(11, 8), sharex=True)
    for j, att in enumerate(("none", "matched", "mismatched")):
        y, u, rho, enc, x = (get(kind, att, k) for k in ("y", "u", "rho", "enc_rho", "x"))
        q = float(get(kind, att, "q")); t = np.arange(len(y))
        for i, (sig, lab) in enumerate(((y, "output y"), (u, "applied input u"), (None, "residual"), (x[:, 1], "hidden state x2"))):
            ax = axes[i, j]; ax.grid(alpha=0.3)
            if att != "none":
                ax.axvspan(0, len(y), color="#d55e00", alpha=0.07, lw=0)
            if i == 2:
                ax.plot(t, enc, color="#8c8c8c", lw=0.5, alpha=0.5); ax.set_ylim(-q / 2, q / 2); ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel("encrypted", color="#8c8c8c", fontsize=8)
                pax = ax.twinx(); pax.set_zorder(ax.get_zorder() + 1); pax.patch.set_visible(False)
                pax.plot(t, rho, color=COL[att], lw=1); pax.set_ylim(-1.6, 1.6)
                if j == 2:
                    pax.set_ylabel("residual (plaintext)")
            else:
                ax.plot(t, sig, color=COL[att], lw=1)
                if j == 0:
                    ax.set_ylabel(lab)
        flag, Z = bool(get(kind, att, "flag")), float(get(kind, att, "Z"))
        axes[0, j].set_title(f"{TITLE[att]}\nwindow test: {'ALARM' if flag else 'no alarm'} (Z = {Z:.0f})", fontsize=9,
                             color="#b00000" if flag else "#006400")
        axes[3, j].set_xlabel("time step")
    # common y limits per row, so the three cases are comparable
    for i in (0, 1, 3):
        lo = min(ax.get_ylim()[0] for ax in axes[i]); hi = max(ax.get_ylim()[1] for ax in axes[i])
        for ax in axes[i]:
            ax.set_ylim(lo, hi)
    fig.suptitle(f"Zero-dynamics attack, {kind} zero (z0 = {za.CASES[kind]['z0']}), encrypted control, AERO2 parameters",
                 fontsize=11, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"zero_attack_{kind}.png", dpi=130); fig.savefig(f"zero_attack_{kind}.pdf")
    plt.close(fig)
print("ok")
