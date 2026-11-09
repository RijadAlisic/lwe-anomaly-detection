"""Exact Type II errors of the detector's test and of the attacker's (opposite) test.

Setting (Section V of the paper): a short filtering vector d turns the encrypted residual into
X = d^T e mod q, with e Gaussian.  Under the null hypothesis X is a wrapped (discrete) Gaussian on
Z_q with standard deviation s = sigma*||d||_2 = q/(2z), where z = q/(2 sigma ||d||_2).

Detector   : H0 = wrapped Gaussian, H1 = uniform.  Reject H0 when |X| is LARGE (two-sided test).
Attacker   : H0 = uniform, H1 = wrapped Gaussian.  Reject H0 when |X| is SMALL (opposite test).

Both tests are computed exactly (no sampling).  Run:  python type2_table.py
"""
import numpy as np

Q = 16381
ZS = [1.145, 1.260, 1.400, 1.575, 1.800, 2.100]
ALPHAS = [0.10, 0.20, 0.30, 0.40]

# Table I of the paper (detector, attacker in parentheses), Type II error in percent.
PAPER = {
    1.145: [(89.5, 89.5), (79.1, 79.1), (68.8, 68.8), (58.6, 58.6)],
    1.260: [(89.0, 89.1), (78.2, 78.3), (67.6, 67.7), (57.2, 57.3)],
    1.400: [(88.1, 88.4), (76.5, 77.0), (65.4, 65.8), (54.9, 55.1)],
    1.575: [(86.4, 87.3), (73.6, 74.9), (61.9, 62.9), (51.3, 51.7)],
    1.800: [(83.0, 85.7), (68.4, 71.8), (56.4, 58.7), (46.2, 46.7)],
    2.100: [(76.2, 83.4), (60.6, 67.4), (49.2, 52.8), (40.0, 40.0)],
}


def wrapped_gaussian_pmf(s, q=Q, k_max=6):
    """Probability mass of N(0, s^2) wrapped onto the centered residues of Z_q."""
    x = np.arange(-(q // 2), q // 2 + 1) if q % 2 else np.arange(-(q // 2), q // 2)
    p = np.zeros(len(x))
    for k in range(-k_max, k_max + 1):
        p += np.exp(-((x + k * q) ** 2) / (2.0 * s * s))
    return x, p / p.sum()


def detector_beta(z, alpha, q=Q):
    """Reject H0 when |X| > t.  Choose the largest accepted interval with tail mass >= alpha."""
    x, p = wrapped_gaussian_pmf(q / (2.0 * z), q)
    ax = np.abs(x)
    # tail(t) = P(|X| > t) is decreasing in t; pick the largest t with tail(t) >= alpha.
    order = np.arange(0, q // 2 + 1)
    tail = np.array([p[ax > m].sum() for m in order])
    t = order[np.max(np.where(tail >= alpha - 1e-12)[0])]
    accepted = int((ax <= t).sum())
    return accepted / q          # Type II error under the uniform alternative


def attacker_beta(z, alpha, q=Q):
    """Reject H0 (uniform) when |X| <= s with uniform mass alpha.  Type II = P_LWE(|X| > s)."""
    x, p = wrapped_gaussian_pmf(q / (2.0 * z), q)
    ax = np.abs(x)
    s = int(np.floor((alpha * q - 1) / 2))
    return float(p[ax > s].sum())


if __name__ == "__main__":
    print(f"q = {Q}.  Type II error in %, detector (attacker's test).")
    print("   z    " + "".join(f"  alpha={int(a*100)}%      " for a in ALPHAS))
    worst = 0.0
    for z in ZS:
        row = []
        for a, ref in zip(ALPHAS, PAPER[z]):
            d, t = 100 * detector_beta(z, a), 100 * attacker_beta(z, a)
            worst = max(worst, abs(d - ref[0]), abs(t - ref[1]))
            row.append(f"{d:5.1f} ({t:5.1f})")
        print(f"{z:6.3f}  " + "   ".join(row))
    print(f"\nlargest deviation from the values printed in the paper: {worst:.2f} percentage points")
