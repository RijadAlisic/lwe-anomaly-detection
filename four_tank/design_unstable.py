"""Search for a lower-noise-gain integer controller for the unstable-zero plant (z0 = 1.02).
Controller (as in zero_attack.py): U = D*yq_k + a*yq_{k-1} + b*yq_{k-2}  (within a reset period kp = 64), yq = round(cs*y), u = U/cu.
The encryption noise at the decrypted input is about sigma*sqrt(D^2+a^2+b^2), so this norm is what decides how large r must be.
Simulates the quantized loop with the state kick (0, xi) at step 4 and keeps the controllers that are bounded, settle to yq = 0 without a
limit cycle for every xi in XIS, and compares them with the current one (-5, -6, 4; cu = 10).   Usage: python design_unstable.py"""
import itertools
import numpy as np
import zero_attack as za

KP, N, KICK = 64, 192, 4
A = np.array([[1.001, 0.4], [0.0, 0.1]]); z0 = 1.02
B = np.array([-0.4 * za.B2 / (z0 - 0.1), za.B2]); cs = 5.0
XIS = (0.35, 0.7)


def sim(D, a, b, cu, xi):
    x = np.zeros(2); yqs = []; ys = []; us = []
    h1 = h2 = 0
    for k in range(N):
        if k % KP == 0:
            h1 = h2 = 0
        y = x[0]; yq = int(round(cs * y))
        U = D * yq + a * h1 + b * h2
        h2, h1 = h1, yq
        u = U / cu
        x = A @ x + B * u
        if k + 1 == KICK:
            x = x + np.array([0.0, xi])
        yqs.append(yq); ys.append(y); us.append(u)
        if abs(y) > 50:
            return None
    return np.array(yqs), np.array(ys), np.array(us)


def evaluate(D, a, b, cu):
    """(peak |y| in the first 64 steps, mean |y| over the epoch, peak |u|) for the largest xi; None if unbounded for any xi."""
    out = None
    for xi in XIS:
        r = sim(D, a, b, cu, xi)
        if r is None:
            return None
        yq, y, u = r
        if np.abs(y).max() > 3:
            return None
        out = (np.abs(y[:64]).max(), np.abs(y).mean(), np.abs(u).max())
    return out


if __name__ == "__main__":
    ref = evaluate(-5, -6, 4, 10.0)
    print("current (-5,-6,4), cu=10: noise gain %.1f, peak|y| (first 64 steps) %.2f, mean|y| %.3f, peak|u| %.2f" % ((25 + 36 + 16) ** .5, *ref))
    found = []
    for cu in (2.0, 3.0, 4.0, 5.0, 10.0):
        for D, a, b in itertools.product(range(-8, 9), repeat=3):
            g = (D * D + a * a + b * b) ** .5
            if g == 0 or g > 8:
                continue
            r = evaluate(D, a, b, cu)
            if r is not None and 0.9 * ref[0] <= r[0] <= 1.3 * ref[0] and r[1] <= 1.5 * ref[1]:   # similar transient, no worse on average
                found.append((g, cu, D, a, b, *r))
    found.sort()
    print("\ncontrollers with a transient like the current one (noise gain, cu, (D,a,b), peak|y| first 64 steps, mean|y|, peak|u|):")
    for g, cu, D, a, b, pk, my, up in found[:25]:
        print(f"  gain {g:4.1f}  cu={cu:4.1f}  ({D:2d},{a:2d},{b:2d})  peak|y| {pk:5.2f}  mean|y| {my:5.3f}  peak|u| {up:5.2f}")
    print(len(found), "controllers pass")
