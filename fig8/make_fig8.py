"""Regenerates the numerical-results figure of 'Anomaly Detection with LWE Encrypted Control'
from the saved run, stored compactly in windowed_det_data.npz (plant output, encrypted and decrypted
residual, one-vector and six-vector test statistics per key epoch).
usage: python make_fig8.py [width_in height_in out.pdf]"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

W = float(sys.argv[1]) if len(sys.argv) > 1 else 7.16
H = float(sys.argv[2]) if len(sys.argv) > 2 else 3.3
out = sys.argv[3] if len(sys.argv) > 3 else "TAC_windowed_det_clean_short.pdf"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Tinos", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "pdf.fonttype": 42, "axes.linewidth": 0.6,
})

D = np.load("windowed_det_data.npz")
y, yhat, enc, rdec = D["y"], D["yhat_e"], D["enc_res"], D["rdec"]
tests, chi = D["det_tests"], D["det_chi"]
T, cs = int(D["T"]), float(D["cs"])
k = np.arange(len(y))
nw = len(chi)
edges = np.arange(nw + 1) * T

C_MEAS, C_PRED = "#0072b2", "#e69f00"          # Okabe-Ito palette, as in the original figure
C_ENC, C_DEC, C_1V, C_6V = "#8c8c8c", "#d55e00", "#cc79a7", "#0072b2"

fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(W, H), gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.30})

# ---- top: plant output and prediction
ax1.plot(k, y, color=C_MEAS, lw=0.6, alpha=0.9, label=r"Measured $y_k$")
ax1.plot(k, yhat / 10.0, color=C_PRED, lw=0.6, alpha=0.9, label=r"Predicted $\hat y_k$")
ax1.set_ylim(-1.6, 1.6); ax1.set_xlim(-250, 6400)
ax1.set_yticks([-1, 0, 1]); ax1.set_ylabel(r"$y$")
ax1.set_xticks(np.arange(0, 6001, 1000)); ax1.set_xticklabels([])
ax1.set_xlabel(r"Time Step $k$", labelpad=0)
ax1.grid(alpha=0.25, lw=0.4)
ax1.legend(loc="upper left", ncol=1, frameon=True, framealpha=0.9, borderpad=0.3, handlelength=1.6)
ax1b = ax1.twinx(); ax1b.set_ylim(-1.6, 1.6); ax1b.set_yticks([-1.6, -0.8, 0, 0.8, 1.6])
ax1b.set_ylabel("Scaled integer")

# ---- bottom: residuals and tests
LEFT = (-8700, 8700)
R0, RS = 7806.0, 966.7           # right axis value v sits at left coordinate RS*v - R0
rlim = ((LEFT[0] + R0) / RS, (LEFT[1] + R0) / RS)
ax3.step(k, enc, where="post", color=C_ENC, alpha=0.3, lw=0.6, zorder=1)
for w in range(nw):  # 1-vector test: |statistic| per window
    x0, x1 = edges[w], edges[w + 1]
    ax3.plot([x0, x1], [abs(tests[w])] * 2, color=C_1V, lw=1.4, ls="-", solid_capstyle="butt", zorder=5)
    ax3.plot([(x0 + x1) / 2], [abs(tests[w])], marker="o", ms=3.2, mfc="white", mec=C_1V, mew=0.9, ls="none", zorder=5)
ax3.set_ylim(*LEFT); ax3.set_xlim(-250, 6400)
ax3.set_xticks(np.arange(0, 6001, 1000)); ax3.set_yticks([-8000, -4000, 0, 4000, 8000])
ax3.set_ylabel("Residual / 1-Vec. Test"); ax3.grid(alpha=0.25, lw=0.4)
ax3b = ax3.twinx(); ax3b.set_ylim(*rlim); ax3b.set_yticks([0, 5, 10, 15]); ax3b.set_ylabel("Dec. Residual / 6-Vec. Test")
ax3b.step(k, np.abs(rdec), where="post", color=C_DEC, lw=0.6, zorder=2)
ax3b.step(edges, np.append(chi, chi[-1]), where="post", color=C_6V, lw=1.1, zorder=4)
thr = 8.558                       # chi^2_6 quantile for alpha = 0.2
C_THR, C_THR2 = "#003f6b", "#56b4e9"          # thresholds of the 6-vector test, same hue family as the test
ax3b.axhline(thr, color=C_THR, ls="--", lw=0.9, zorder=3)
ax3b.axhline(0.9 * thr, color=C_THR2, ls=":", lw=1.1, zorder=3)

handles = [Line2D([], [], color=C_ENC, alpha=0.4, lw=1.2, label="Enc. Residual"),
           Line2D([], [], color=C_1V, lw=1.4, marker="o", ms=3.2, mfc="white", mew=0.9, label="1-Vector Test"),
           Line2D([], [], color=C_DEC, lw=1, label=r"Dec. Residual $|\rho_k|$"),
           Line2D([], [], color=C_6V, lw=1.1, label="6-Vector Test"),
           Line2D([], [], color=C_THR, lw=0.9, ls="--", label=r"$\chi^2_6$ threshold ($\alpha=0.2$)"),
           Line2D([], [], color=C_THR2, lw=1.1, ls=":", label="10% lower threshold")]
ax3.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3, frameon=False, columnspacing=0.8, handlelength=1.6, handletextpad=0.4)
fig.subplots_adjust(left=0.175, right=0.845, top=0.985, bottom=0.205)
fig.savefig(out)
print("wrote", out, "size", W, "x", H, "in")
