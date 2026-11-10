"""Type II error of the detector (beta) and of the attacker's test (beta_att) against the Type I error alpha, exact wrapped-normal model
(q = 16381, sigma = 26, ||d|| = 315/z).  Left: both curves for several z and the guessing line beta = 1 - alpha.  Right: beta - beta_att
(positive: the attacker's test has the smaller Type II error); the zero crossings are where the attacker overtakes."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

q, sig = 16381, 26
def cdf_abs(a, st, K=12):
    ks = np.arange(-K, K + 1); return sum(norm.cdf((a + k * q) / st) - norm.cdf((-a + k * q) / st) for k in ks)
def bdet(z, al):
    st = sig * 315 / z; a = brentq(lambda a: cdf_abs(a, st) - (1 - al), 1e-6, q / 2); return 2 * a / q
def batt(z, al): return 1 - cdf_abs(al * q / 2, sig * 315 / z)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Tinos", "STIXGeneral"], "mathtext.fontset": "stix", "font.size": 9})
al = np.linspace(0.01, 0.99, 99)
Z = [(1.575, "#8c8c8c"), (2.1, "#cc79a7"), (3.0, "#0072b2"), (8.0, "#d55e00")]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax[0].plot(al, 1 - al, color="k", lw=0.8, ls=":", label=r"guessing, $\beta=1-\alpha$")
for z, col in Z:
    ax[0].plot(al, [bdet(z, a) for a in al], color=col, lw=1.4, label=f"$z={z:g}$ detector")
    ax[0].plot(al, [batt(z, a) for a in al], color=col, lw=1.4, ls="--")
ax[0].set_xlabel(r"Type I error $\alpha$"); ax[0].set_ylabel(r"Type II error $\beta$"); ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)
ax[0].plot([], [], color="k", ls="--", lw=1.2, label="dashed: attacker")
ax[0].legend(fontsize=6.5, loc="lower left", framealpha=0.9, edgecolor="none")
ax[0].axvline(0.5, color="k", lw=0.4, alpha=0.4)
ax[1].axhline(0, color="k", lw=0.8)
for z, col in Z:
    d = np.array([bdet(z, a) - batt(z, a) for a in al]) * 100
    ax[1].plot(al, d, color=col, lw=1.4, label=f"$z={z:g}$")
    s = np.where(np.diff(np.sign(d)) != 0)[0]
    if len(s):
        i = s[0]; x0 = al[i] - d[i] * (al[i + 1] - al[i]) / (d[i + 1] - d[i])
        ax[1].plot([x0], [0], "o", color=col, ms=4); print(f"z={z}: crossover at alpha = {x0:.3f}")
ax[1].set_xlabel(r"Type I error $\alpha$"); ax[1].set_ylabel(r"$\beta-\beta_{\mathrm{att}}$ [percentage points]"); ax[1].set_xlim(0, 1)
ax[1].text(0.62, 12.2, "attacker's test better", fontsize=7, color="k"); ax[1].text(0.03, -18, "detector better", fontsize=7, color="k")
ax[1].legend(fontsize=7, loc="lower right", framealpha=0.9, edgecolor="none")
for a in ax: a.grid(alpha=0.25, lw=0.4)
fig.tight_layout(); fig.savefig("beta_vs_alpha.png", dpi=200); fig.savefig("beta_vs_alpha.pdf")
