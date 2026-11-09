# Zero-dynamics (stealthy) attack study

Files: `zero_attack.py` (plant, controller, attack), `zero_attack_study.py` (encrypted runs + detection, writes
`det_results/zero_attack.npz`), `zero_attack_plots.py` (figures `zero_attack_stable.png`, `zero_attack_unstable.png`),
`zero_attack_sweep.py` (mismatch sweep), `sigma_sweep.py` (noise sweep).

## Nominal plant (discrete time, Ts = 1 s)

The linear plant of the paper, with one change in the input matrix so that it has a transmission zero:

    x+ = A x + B u,   y = C x,    A = [[1.001, 0.4], [0, 0.1]],   C = [1 0],   B = [b1, 0.4722]^T

The paper's B has b1 = 0 (relative degree 2, no zero).  With b1 != 0 the zero is z0 = 0.1 - 0.4*0.4722/b1:

| case | z0 | b1 | open-loop poles |
|---|---|---|---|
| stable zero | 0.7 | -0.3148 | 1.001 (unstable), 0.1 |
| unstable zero | 1.02 | -0.2053 | 1.001 (unstable), 0.1 |

Initial state (the zero-dynamics direction): x(0) = (0, xi) with xi = 3.0 (stable) and xi = 0.7 (unstable).

## Controller (integer, same structure as the paper's eq. (23), c = 1, reset every kp = 64 steps)

    X+ = [[0,0],[1,0]] X + [1,0]^T yq,   U = [a b] X + D yq,   Yhat = [1 0] X,   rho = yq - Yhat,   u = U / cu,   yq = round(cs y)

so X = (yq[k-1], yq[k-2]) and U = D yq[k] + a yq[k-1] + b yq[k-2], a two-tap output feedback.  X is reset to 0 at
k = 0, 64, 128, ... .  Gains found by a brute-force search for a stable closed loop (integer coefficients):

| case | (D, a, b) | cs | cu | real gain cs/cu*(D,a,b) | closed-loop pole radii |
|---|---|---|---|---|---|
| stable zero | (5, 3, 1) | 5 | 5 | (5, 3, 1) | 0.75, 0.70, 0.70, 0.61 |
| unstable zero | (-5, -1, 1) | 5 | 10 | (-2.5, -0.5, 0.5) | 0.97, 0.53, 0.53, 0.38 |

The unstable-zero loop has a slow pole at 0.974, which is why the plateau after the initial transient decays slowly.
The residual is the paper's: yq[k] - yq[k-1] (the controller's "estimate" is the previous quantized output), so it is
zero whenever the quantized output does not change.  The output quantization step is 1/cs = 0.2, so the deadband is
|y| < 0.1.

## Attack

Zero-dynamics attack on the actuator: from x(0) = (0, xi) the input

    u_a(k) = -(0.4/b1) * xi * z0^k      (integer injection U_a = round(cu * u_a), added to the ciphertext of U)

keeps y(k) = 0 exactly (up to the rounding of u_a), so neither the controller nor the residual reacts.  It runs
for the whole key epoch of T = 192 steps.
* matched: the attacker's xi equals the true one.
* mismatched: the attacker assumes 0.5*xi.
* none: no attack (the initial state alone produces a transient, y(1) = 0.4 xi).
The stable-zero attack decays (z0^k) and is harmless: the hidden state x2 returns to 0.  The unstable-zero attack grows
as 1.02^k: after 192 steps x2 has grown from 0.7 to about 31 and u_a to about 60, while y stays at 0.
z0 = 1.02 and xi = 0.7 are limited by the ciphertext range: r*U_a must stay below q/2.

## Encryption and detection

LWE with the AERO2 parameters (v = 64, q = 2^23 + 9, sigma = 50, r = 4096), key refresh every T = 192 steps,
integer controller evaluated on ciphertexts, no decryption errors.  Detector: kernel lattice of the 192 residual
ciphertexts, LLL (blaster), 6 shortest vectors, chi-square test at alpha = 0.001.

## Results (one key epoch each)

| zero | attack | nonzero residuals | max x2 | Z | alarm |
|---|---|---|---|---|---|
| stable | none | 9 | 3.0 | 31.7 | yes (transient) |
| stable | matched | 0 | 3.0 | 10.8 | no |
| stable | mismatched | 17 | 3.0 | 43.5 | yes |
| unstable | none | 7 | 0.84 | 66.7 | yes (transient) |
| unstable | matched | 0 | 30.7 | 10.8 | no |
| unstable | mismatched | 3 | 15.4 | 112.2 | yes |

Detectability of a mismatch depends on the amplitude xi, not on the stability of the zero: a state error of
(1-g)*xi shows up as about 0.4*(1-g)*xi in the output, and only deviations above the quantization deadband (0.1)
reach the residual.  At equal xi the two cases behave alike (see `zero_attack_sweep.py`).
