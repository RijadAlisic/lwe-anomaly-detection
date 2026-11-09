"""Single-column figure for the paper (unstable zero): output and hidden state in one panel, residual below.
Style as in fig7/make_fig7.py (serif font, Okabe-Ito colors).  Reads det_results/stealth_example.npz.
usage: python stealth_fig_paper.py [width_in height_in out.pdf]"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

W = float(sys.argv[1]) if len(sys.argv) > 1 else 3.5
H = float(sys.argv[2]) if len(sys.argv) > 2 else 2.55
out = sys.argv[3] if len(sys.argv) > 3 else "stealth_fig_paper.pdf"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Tinos", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.5,
    "pdf.fonttype": 42, "axes.linewidth": 0.6,
})

D = np.load("det_results/stealth_example.npz")
K0 = 196                                                   # step at which the disturbance and the attack start
lo, hi = K0 - 4, K0 + 100
t = np.arange(lo, hi) - K0
C_ENC = "#8c8c8c"
cases = (("none", "#cc79a7", "Disturbance Only"), ("mismatched", "#0072b2", "Attack, State Guessed (0.5x)"),
         ("matched", "#d55e00", "Attack, State Known"))

fig, ax = plt.subplots(2, 1, figsize=(W, H), sharex=True, gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})
handles = []
for name, col, lab in cases:
    ax[0].plot(t, D[f"unstable_{name}_y"][lo:hi], color=col, lw=0.9)
    ax[0].plot(t, D[f"unstable_{name}_x2"][lo:hi], color=col, lw=0.9, ls="--")
    flag = bool(D[f"unstable_{name}_flag"])
    handles.append(Line2D([0], [0], color=col, lw=1.2, label=f"{lab}: {'Alarm' if flag else 'No Alarm'}"))
handles += [Line2D([0], [0], color="k", lw=0.9, label=r"Output $y$ (solid)"),
            Line2D([0], [0], color="k", lw=0.9, ls="--", label=r"Hidden State $x_2$ (dashed)")]
ax[0].set_ylabel(r"$y$, $x_2$")
ax[0].set_ylim(-3, 5.4)

ax[1].plot(t, D["unstable_matched_enc"][lo:hi], color=C_ENC, lw=0.5, alpha=0.4, drawstyle="steps-post")
q = float(D["unstable_matched_q"]); ax[1].set_ylim(-q / 2, q / 2); ax[1].set_yticks([])
ax[1].set_ylabel("Enc. Residual", color=C_ENC)
pax = ax[1].twinx(); pax.set_zorder(ax[1].get_zorder() + 1); pax.patch.set_visible(False)
for name, col, lab in cases:
    pax.plot(t, D[f"unstable_{name}_rho"][lo:hi], color=col, lw=0.9, drawstyle="steps-post")
pax.set_ylim(-3.1, 1.6); pax.set_yticks([-1, 0, 1]); pax.set_ylabel(r"Residual $\rho_k$")
ax[1].set_xlabel("Time Since Disturbance [Steps]", labelpad=1)
pax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.9, borderpad=0.3, handlelength=1.6,
           labelspacing=0.2, edgecolor="none")
for a in ax:
    a.grid(alpha=0.25, lw=0.4); a.axvline(0, color="k", ls=":", lw=0.6)
ax[0].set_xlim(-4, 100)
fig.subplots_adjust(left=0.13, right=0.87, top=0.98, bottom=0.145)
fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=220)
