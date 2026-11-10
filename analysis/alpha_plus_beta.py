"""alpha+beta against alpha for detector and attacker's test, exact wrapped-normal model (q=16381, sigma=26), several sigma~."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
q = 16381
def cdf_abs(a, st, K=14):
    ks = np.arange(-K, K + 1); return sum(norm.cdf((a + k*q)/st) - norm.cdf((-a + k*q)/st) for k in ks)
bdet = lambda st, al: 2*brentq(lambda a: cdf_abs(a, st) - (1-al), 1e-9, q/2)/q
batt = lambda st, al: 1 - cdf_abs(al*q/2, st)
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 9})
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
al = np.linspace(0.001, 0.999, 400)
for a_, st in zip(ax, (300, 1500, 6000)):
    sd = np.array([al_ + bdet(st, al_) for al_ in al]); sa = np.array([al_ + batt(st, al_) for al_ in al])
    a_.plot(al, sd, "#0072b2", lw=1.5, label="detector"); a_.plot(al, sa, "#d55e00", lw=1.5, ls="--", label="attacker")
    a_.axhline(1, color="gray", lw=0.8); a_.set_title(rf"$\tilde\sigma={st}$", fontsize=9); a_.set_xlabel(r"$\alpha$"); a_.grid(alpha=.25, lw=.4)
    print(f"sigma~={st}: min(alpha+beta) detector {sd.min():.4f} at alpha={al[sd.argmin()]:.3f} (beta={bdet(st, al[sd.argmin()]):.3f});"
          f"  attacker {sa.min():.4f} at alpha={al[sa.argmin()]:.3f} (beta={batt(st, al[sa.argmin()]):.3f})")
ax[0].set_ylabel(r"$\alpha+\beta$"); ax[0].legend(fontsize=7, loc="lower right", framealpha=.9, edgecolor="none"); ax[0].set_ylim(0, 1.05)
fig.tight_layout(); fig.savefig("alpha_plus_beta.png", dpi=200)
