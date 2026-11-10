"""Rate check: beta against sigma~ = sigma*||d||_2 at fixed alpha, exact wrapped-normal model (q = 16381, sigma = 26), log-log.
Detector (solid), attacker's test (dashed), Theorem 2's bound (dotted), guessing level 1 - alpha (horizontal line)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

q, sig, v = 16381, 26, 64
def cdf_abs(a, st, K=14):
    ks = np.arange(-K, K + 1); return sum(norm.cdf((a + k * q) / st) - norm.cdf((-a + k * q) / st) for k in ks)
def bdet(st, al):
    a = brentq(lambda a: cdf_abs(a, st) - (1 - al), 1e-9, q / 2); return 2 * a / q
def batt(st, al): return 1 - cdf_abs(al * q / 2, st)
def bound(st, al, Q=2 * ((q - 1) / 2) ** 2 * (v + 1) + 1):               # Theorem 2, a = 1/2, b = 2/pi, l = 1
    inner = 1 - (1 - al) ** 2 * (1 - np.exp(-(2 / np.pi) * Q ** 2 / (4 * st ** 2)))
    return np.sqrt(-4 * st ** 2 / (0.5 * q ** 2) * np.log(inner))

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Tinos", "STIXGeneral"], "mathtext.fontset": "stix", "font.size": 9})
st = np.logspace(1.2, 4.3, 160)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for a_, al in zip(ax, (0.1, 0.5)):
    a_.plot(st, [bdet(s, al) for s in st], color="#0072b2", lw=1.5, label="detector")
    a_.plot(st, [batt(s, al) for s in st], color="#d55e00", lw=1.5, ls="--", label="attacker")
    a_.plot(st, np.minimum(bound(st, al), 3), color="k", lw=0.9, ls=":", label="Theorem 2 bound")
    a_.axhline(1 - al, color="gray", lw=0.7); a_.text(st[0] * 1.1, (1 - al) * 1.08, r"$1-\alpha$", fontsize=7, color="gray")
    a_.set_xscale("log"); a_.set_yscale("log"); a_.set_ylim(1e-5, 2); a_.set_xlabel(r"$\tilde\sigma=\sigma\|d\|_2$ (short vectors $\leftarrow$)")
    a_.set_title(rf"$\alpha={al}$", fontsize=9); a_.grid(alpha=0.25, lw=0.4)
    a_.axvline(al * q / 2, color="#d55e00", lw=0.5, ls="-.")                                    # where the attacker's threshold alpha q/2 equals sigma~
ax[0].set_ylabel(r"Type II error $\beta$"); ax[0].legend(fontsize=7, loc="lower right", framealpha=0.9, edgecolor="none")
fig.tight_layout(); fig.savefig("beta_vs_sigma.png", dpi=200); fig.savefig("beta_vs_sigma.pdf")
# slopes: local log-log slope of beta against sigma~ at small sigma~
for al in (0.1, 0.5):
    s1, s2 = 60.0, 120.0
    sl = lambda f: (np.log(f(s2, al)) - np.log(f(s1, al))) / (np.log(s2) - np.log(s1))
    print(f"alpha={al}: log-log slope between sigma~=60 and 120:  detector {sl(bdet):.2f},  attacker {sl(batt):.1f}")
