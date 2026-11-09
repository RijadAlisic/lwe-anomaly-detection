"""Figures of the zero-dynamics sweep (reads det_results/zero_sweep.json): sweep_A.png, sweep_B.png."""
import json, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("det_results/zero_sweep.json"))
G = collections.defaultdict(list)
for r in R:
    G[(r["kind"], r["scen"], r["xi"], r["par"])].append(r)
COL = {0.7: "#0072b2", 0.35: "#d55e00"}

# ---- A: mismatch sweep
fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex="col")
gs = [0.0, 0.5, 0.8, 0.9, 0.95, 1.0]
for i, kind in enumerate(("stable", "unstable")):
    for xi in (0.7, 0.35):
        pts = [(g, G[(kind, "A", xi, g)]) for g in gs if (kind, "A", xi, g) in G]
        x = [p[0] for p in pts]
        a0 = [np.mean([r["wins"][0]["flag"] for r in rs]) for _, rs in pts]
        a12 = [np.mean([r["wins"][1]["flag"] or r["wins"][2]["flag"] for r in rs]) for _, rs in pts]
        x2 = [np.median([r["wins"][0]["max_x2"] for r in rs]) for _, rs in pts]
        err = [any(r["dec_errors"] for r in rs) for _, rs in pts]
        for ax, y in zip(axes[i], (a0, a12, x2)):
            ax.plot(x, y, "-o", color=COL[xi], label=f"xi = {xi}", lw=1.2, ms=5)
            ex = [xx for xx, e, yy in zip(x, err, y) if e]; ey = [yy for e, yy in zip(err, y) if e]
            ax.plot(ex, ey, "x", color="k", ms=9, mew=1.5, label="decryption errors" if xi == 0.7 else None)
    axes[i, 0].set_ylabel(f"{kind} zero")
for ax, t in zip(axes[0], ("alarm rate, attacked window", "alarm rate, windows after the attack", "hidden state max|x2| in the attack window")):
    ax.set_title(t, fontsize=9)
for ax in axes[:, :2].ravel():
    ax.set_ylim(-0.05, 1.05)
for ax in axes.ravel():
    ax.grid(alpha=0.3)
for ax in axes[1]:
    ax.set_xlabel("attacker's guess g (1 = matched)")
axes[0, 0].legend(fontsize=8)
fig.suptitle("A: zero-dynamics attack on the true state, attacker guesses g*xi (3 key seeds; x = decryption errors in those runs)", fontsize=10)
fig.tight_layout(); fig.savefig("sweep_A.png", dpi=130); plt.close(fig)

# ---- B: attack from rest
eps = [None, 0.05, 0.1, 0.2, 0.4]
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
for i, kind in enumerate(("stable", "unstable")):
    M = np.array([[np.mean([r["wins"][w]["flag"] for r in G[(kind, "B", 0.0, e)]]) for w in range(4)] for e in eps])
    ax = axes[i, 0]
    im = ax.imshow(M, vmin=0, vmax=1, cmap="Reds", aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(["0 calm", "1 attack", "2 after", "3 later"])
    ax.set_yticks(range(len(eps))); ax.set_yticklabels(["none" if e is None else f"{e}" for e in eps])
    for a in range(len(eps)):
        for w in range(4):
            ax.text(w, a, f"{M[a, w]:.0f}" if M[a, w] in (0, 1) else f"{M[a, w]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"{kind} zero: alarm rate per window", fontsize=9); ax.set_ylabel("amplitude eps")
    ax = axes[i, 1]
    y1 = [max(r["wins"][1]["max_x2"] for r in G[(kind, "B", 0.0, e)]) for e in eps]
    y2 = [max(r["wins"][2]["max_x2"] for r in G[(kind, "B", 0.0, e)]) for e in eps]
    xx = np.arange(len(eps)); ax.bar(xx - 0.2, y1, 0.4, label="window 1 (attack)", color="#e69f00"); ax.bar(xx + 0.2, y2, 0.4, label="window 2 (after the attack)", color="#d55e00")
    ax.set_xticks(xx); ax.set_xticklabels(["none" if e is None else f"{e}" for e in eps]); ax.set_xlabel("amplitude eps")
    ax.set_ylabel("max |x2| (hidden state)"); ax.set_title("hidden state: during and right after the attack", fontsize=9); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("B: zero-dynamics attack starting from rest at step 192, lasting one window (3 key seeds)", fontsize=10)
fig.tight_layout(); fig.savefig("sweep_B.png", dpi=130)
print("ok")
