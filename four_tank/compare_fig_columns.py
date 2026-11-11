"""Figure for the paper (linear x, 0..XMAX nonzero residual samples): detection statistic Z (top) and alarm fraction (bottom).
Top: at each integer number of nonzero samples n_rho, one column of dots per attack (orange: noise injection, blue / green: stable / unstable
zero-dynamics), side by side; lines: medians.  Bottom: alarm fraction (Z above the chi2_6 threshold 22.5 at alpha = 1e-3) and the plaintext test.
Writes comparison_fig.pdf / .png."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2

XMAX = 15
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Tinos", "STIXGeneral"], "mathtext.fontset": "stix",
                     "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 5.8,
                     "pdf.fonttype": 42, "axes.linewidth": 0.6})
thr = chi2.isf(1e-3, 6)

# noise injection, 100 keys per length: group the keys by their own number of nonzero samples
runs = []
D = json.load(open("det_results/detector_comparison_random_r2000_fine_stablezero.json")); runs += D["runs"]; ATT = D["ATT"]
noise = {}
for r in runs:
    e = r["epochs"][ATT]
    noise.setdefault(e["nz"], []).append(e["Z"])
# zero-dynamics, 50 keys per state guess g: one group per number of nonzero samples (g = 1 only for n = 0, the other g give the same ciphertexts)
Zr = json.load(open("det_results/zero_sweep_fine.json")) + json.load(open("det_results/zero_sweep_fine_more.json"))
zd = {}
for kind in ("stable", "unstable"):
    d = {}
    for r in Zr:
        if r["kind"] != kind or r["g"] == 0.0 or r["g"] in (0.99, 0.98):
            continue
        if r["nz1"] == 0 and r["g"] != 1.0:
            continue
        d.setdefault(r["nz1"], []).append(r["Z1"])
    zd[kind] = d

series = [("Noise injection", "#d55e00", -0.15, noise), ("Stable zero-dynamics", "#0072b2", 0.0, zd["stable"]),
          ("Unstable zero-dynamics", "#009e73", 0.15, zd["unstable"])]
fig, ax = plt.subplots(2, 1, figsize=(3.5, 2.15), sharex=True, gridspec_kw={"height_ratios": [1.35, 0.7], "hspace": 0.08})
rng = np.random.default_rng(0)
STEP = 0.15
def offset(n, idx):
    """Centered columns: the series present at this n_rho are spaced STEP apart around the integer n_rho."""
    present = [i for i, (_, _, _, d) in enumerate(series) if n in d]
    return (present.index(idx) - (len(present) - 1) / 2) * STEP
for idx, (name, col, _, d) in enumerate(series):
    ns = sorted(n for n in d if n <= XMAX)
    for n in ns:
        z = np.array(d[n], float)
        ax[0].plot(n + offset(n, idx) + rng.uniform(-0.05, 0.05, len(z)), z, ".", ms=3.2, color=col, alpha=0.45, mew=0, zorder=1)
    ax[0].plot([n + offset(n, idx) for n in ns], [np.median(d[n]) for n in ns], color=col, lw=1.0, label=name)
    ax[1].plot(ns, [np.mean(np.array(d[n]) > thr) for n in ns], color=col, marker={"#d55e00": "s", "#0072b2": "o", "#009e73": "^"}[col], ms=3, lw=1.0)
# harder encryption: sigma = 75 (r = 2000), noise injection, runs without decryption errors (the other 27 of 100 keys had 1-2 wrong decryptions)
D75 = json.load(open("det_results/detector_comparison_random_sigma75_r2000_fine_stablezero.json"))
n75 = {}
for r in D75["runs"]:
    if r["dec_errors"]:
        continue
    e = r["epochs"][D75["ATT"]]; n75.setdefault(e["nz"], []).append(e["Z"])
ns75 = sorted(n for n in n75 if n <= XMAX and len(n75[n]) >= 15)
ax[0].plot(ns75, [np.median(n75[n]) for n in ns75], color="#d55e00", ls="--", lw=1.2)
ax[1].plot(ns75, [np.mean(np.array(n75[n]) > thr) for n in ns75], color="#d55e00", ls="--", lw=1.2)
# harder encryption for the zero-dynamics attacks: sigma = 75, runs without decryption errors
Z75 = json.load(open("det_results/zero_sweep_fine_sigma75.json"))
for kind, col in (("stable", "#0072b2"), ("unstable", "#009e73")):
    d = {}
    for r in Z75:
        if r["kind"] != kind or r["dec"] or r["g"] is None or r["g"] == 0.0 or r["g"] in (0.99, 0.98):
            continue
        if r["nz1"] == 0 and r["g"] != 1.0:
            continue
        d.setdefault(r["nz1"], []).append(r["Z1"])
    ns = sorted(n for n in d if n <= XMAX)
    ax[0].plot(ns, [np.median(d[n]) for n in ns], color=col, ls="--", lw=1.2)
    ax[1].plot(ns, [np.mean(np.array(d[n]) > thr) for n in ns], color=col, ls="--", lw=1.2)
ax[0].axhline(thr, color="k", ls=":", lw=1.0, label=r"Threshold $22.5$")
ax[0].set_yscale("log"); ax[0].set_ylabel(r"Statistic $Z$")
ax[1].plot([0, 0.5], [0, 0], color="k", lw=0.9); ax[1].plot([0.5, 0.5], [0, 1], color="k", lw=0.9)
ax[1].plot([0.5, XMAX + 5], [1, 1], color="k", lw=0.9, label="Plaintext test")
ax[1].set_ylim(-0.08, 1.12); ax[1].set_yticks([0, 0.5, 1]); ax[1].set_ylabel("Alarm Fraction", fontsize=7, labelpad=1)
h0, l0 = ax[0].get_legend_handles_labels(); h1, l1 = ax[1].get_legend_handles_labels()
from matplotlib.lines import Line2D
extra = [Line2D([0], [0], color="#555555", lw=1.0, label=r"$\sigma=50$"), Line2D([0], [0], color="#555555", lw=1.2, ls="--", label=r"$\sigma=75$ (more noise)")]
ax[1].legend(h0 + h1 + extra, l0 + l1 + [e.get_label() for e in extra], loc="lower right", ncol=2, framealpha=0.9, borderpad=0.2, labelspacing=0.1, columnspacing=0.8,
             handlelength=1.3, edgecolor="none")
ax[1].set_xlim(-0.6, XMAX + 0.5); ax[1].set_xticks(np.arange(0, XMAX + 1, 5))
ax[1].set_xlabel("Number of Samples with $\\rho_k\\neq0$", labelpad=0.5)
for a in ax:
    a.grid(alpha=0.25, lw=0.4)
fig.subplots_adjust(left=0.13, right=0.97, top=0.985, bottom=0.18)
fig.savefig("comparison_fig.pdf"); fig.savefig("comparison_fig.png", dpi=220)
print("noise injection n_rho values:", sorted(noise)[:16])
