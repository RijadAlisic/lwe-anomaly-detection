"""Single-column figure for the paper (unstable zero): output and hidden state in one panel, residual below;
reads det_results/stealth_example.npz."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

D = np.load("det_results/stealth_example.npz")
T, K0 = 192, 196
lo, hi = K0 - 4, K0 + 100
t = np.arange(lo, hi) - K0
cases = (("none", "#7f7f7f", "disturbance only"), ("mismatched", "#0072b2", "attack, state guessed (0.5x)"),
         ("matched", "#d55e00", "attack, state known"))
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6})
fig, ax = plt.subplots(2, 1, figsize=(3.5, 3.2), sharex=True, gridspec_kw=dict(height_ratios=[1.35, 1]))
handles = []
for name, col, lab in cases:
    ax[0].plot(t, D[f"unstable_{name}_y"][lo:hi], color=col, lw=0.9)
    ax[0].plot(t, D[f"unstable_{name}_x2"][lo:hi], color=col, lw=0.9, ls="--")
    flag = bool(D[f"unstable_{name}_flag"])
    handles.append(Line2D([0], [0], color=col, lw=1.2, label=f"{lab}: {'alarm' if flag else 'no alarm'}"))
handles += [Line2D([0], [0], color="k", lw=0.9, label="output $y$ (solid)"), Line2D([0], [0], color="k", lw=0.9, ls="--", label="hidden state $x_2$ (dashed)")]
ax[0].set_ylabel("$y$ and $x_2$")
ax[1].plot(t, D["unstable_matched_enc"][lo:hi], color="#8c8c8c", lw=0.4, alpha=0.5)
q = float(D["unstable_matched_q"]); ax[1].set_ylim(-q / 2, q / 2); ax[1].set_yticks([]); ax[1].set_ylabel("encrypted", color="#8c8c8c", fontsize=6)
pax = ax[1].twinx(); pax.set_zorder(ax[1].get_zorder() + 1); pax.patch.set_visible(False)
for name, col, lab in cases:
    pax.plot(t, D[f"unstable_{name}_rho"][lo:hi], color=col, lw=0.9)
pax.set_ylim(-2.6, 1.6); pax.set_ylabel("residual")
ax[1].set_xlabel("time since disturbance [steps]")
pax.legend(handles=handles, fontsize=5.5, loc="lower right", frameon=True, framealpha=0.9, edgecolor="none", borderpad=0.3, labelspacing=0.25)
for a in ax:
    a.grid(alpha=0.3, lw=0.4); a.axvline(0, color="k", ls=":", lw=0.6)
fig.tight_layout(pad=0.4); fig.savefig("stealth_fig_paper.pdf"); fig.savefig("stealth_fig_paper.png", dpi=220)
