"""Single-column figure for the paper (unstable zero): output, residual, hidden state; reads det_results/stealth_example.npz."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = np.load("det_results/stealth_example.npz")
T, K0 = 192, 196
lo, hi = K0 - 4, K0 + 100
t = np.arange(lo, hi) - K0
cases = (("none", "#7f7f7f", "disturbance only"), ("mismatched", "#0072b2", "attack, state guessed (0.5x)"),
         ("matched", "#d55e00", "attack, state known"))
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6})
fig, ax = plt.subplots(3, 1, figsize=(3.5, 3.9), sharex=True)
for name, col, lab in cases:
    g = lambda key: D[f"unstable_{name}_{key}"][lo:hi]
    flag = bool(D[f"unstable_{name}_flag"])
    ax[0].plot(t, g("y"), color=col, lw=0.9, label=f"{lab}: {'alarm' if flag else 'no alarm'}")
    ax[2].plot(t, g("x2"), color=col, lw=0.9)
ax[1].plot(t, D["unstable_matched_enc"][lo:hi], color="#8c8c8c", lw=0.4, alpha=0.5)
q = float(D["unstable_matched_q"]); ax[1].set_ylim(-q / 2, q / 2); ax[1].set_yticks([]); ax[1].set_ylabel("encrypted", color="#8c8c8c", fontsize=6)
pax = ax[1].twinx(); pax.set_zorder(ax[1].get_zorder() + 1); pax.patch.set_visible(False)
for name, col, lab in cases:
    pax.plot(t, D[f"unstable_{name}_rho"][lo:hi], color=col, lw=0.9)
pax.set_ylim(-1.6, 1.6); pax.set_ylabel("residual")
ax[0].set_ylabel("output $y$"); ax[2].set_ylabel("hidden state $x_2$"); ax[2].set_xlabel("time since disturbance [steps]")
ax[0].legend(fontsize=5.5, loc="upper right", frameon=False)
for a in ax:
    a.grid(alpha=0.3, lw=0.4); a.axvline(0, color="k", ls=":", lw=0.6)
fig.tight_layout(pad=0.4); fig.savefig("stealth_fig_paper.pdf"); fig.savefig("stealth_fig_paper.png", dpi=220)
