import time
import math
import socket
import json
import itertools
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
from charm_compat import integer   # charm-crypto if available, else a pure-Python int fallback


def elapsed_time(t0):
    """Elapsed time (s) since a perf_counter() timestamp t0."""
    return time.perf_counter() - t0


def build_physics_model(A_c, B_c, C_phys, D_phys, T_samp):
    """ZOH-discretize a continuous-time physics model at T_samp -> (A_d, B_d)."""
    A_d, B_d, _, _, _ = cont2discrete((A_c, B_c, C_phys, D_phys), T_samp, method='zoh')
    return A_d, B_d


def build_observer_gain(A, C, Q_l, R_l):
    """Discrete LQE gain L (4x2) via the LQR/LQE dual Riccati solve."""
    A_dual = A.T
    B_dual = C.T
    P_l = solve_discrete_are(A_dual, B_dual, Q_l, R_l)
    K_dual = np.linalg.inv(R_l + B_dual.T @ P_l @ B_dual) @ (B_dual.T @ P_l @ A_dual)   # 2x4
    return K_dual.T   # L, 4x2


def design_lqr(u_arr, y_arr, yr_arr, pitch_arr, pitchrate_arr, target_rad,
               T_samp, Q_lqr, Q_i, R_lqr, sysid_time,
               A_c, B_c, C_phys, D_phys, label=""):

    tag = f" ({label})" if label else ""
    result = dict(ready=False, K=None, Ki=None, x_ss=None, u_ss=None, A=None, B=None, c=None)

    def fit_4state():
        # x[k+1] = A x[k] + B u[k]   for the valid overlapping range
        n = len(u_arr)

        yaw_now   = y_arr[:n-1]
        yr_now    = yr_arr[:n-1]
        pitch_now = pitch_arr[:n-1]
        pr_now    = pitchrate_arr[:n-1]

        # yaw_next   = y_arr[1:n]
        yr_next    = yr_arr[1:n]
        # pitch_next = pitch_arr[1:n]
        pr_next    = pitchrate_arr[1:n]

        U = u_arr[:n-1]

        print("State/input standard deviations:")
        print(f"  yaw   : {np.std(yaw_now):.6e}")
        print(f"  yr    : {np.std(yr_now):.6e}")
        print(f"  pitch : {np.std(pitch_now):.6e}")
        print(f"  pr    : {np.std(pr_now):.6e}")
        print(f"  u     : {np.std(U):.6e}")

        # ---------------------------------------------------------
        # Yaw-rate subsystem:
        #
        # yr[k+1] = a_yr * yr[k] + b_yr * u[k]
        # ---------------------------------------------------------

        Phi_yr = np.column_stack([
            yr_now,
            U,
        ])

        theta_yr, _, _, _ = np.linalg.lstsq(
            Phi_yr,
            yr_next,
            rcond=None
        )

        # ---------------------------------------------------------
        # Pitch-rate subsystem:
        #
        # pr[k+1] = a_pitch * pitch[k]
        #         + a_pr * pr[k]
        #         + b_pitch * u[k]
        # ---------------------------------------------------------

        Phi_pr = np.column_stack([
            pitch_now,
            pr_now,
            U,
        ])

        theta_pr, _, _, _ = np.linalg.lstsq(
            Phi_pr,
            pr_next,
            rcond=None
        )

        # ---------------------------------------------------------
        # Build A, B
        # ---------------------------------------------------------

        A = np.zeros((4, 4))
        B = np.zeros((4, 1))

        # Known yaw kinematics
        A[0, 0] = 1.0
        A[0, 1] = T_samp

        # Yaw-rate dynamics
        A[1, 1] = theta_yr[0]
        B[1, 0] = theta_yr[1]

        # Known pitch kinematics
        A[2, 2] = 1.0
        A[2, 3] = T_samp

        # Pitch-rate dynamics
        A[3, 2] = theta_pr[0]
        A[3, 3] = theta_pr[1]
        B[3, 0] = theta_pr[2]

        return A, B

    if sysid_time == 0:
        # Default path: physics-derived model (QA2C parameters), ZOH-discretized.
        # 2 inputs [V_pitch, V_yaw] -> B is 4x2.
        A, B = build_physics_model(A_c, B_c, C_phys, D_phys, T_samp)

    else:
        # Legacy path: empirical fit from the Phase-0 PRBS excitation, kept for
        # a possible future demo. NOTE — this assumes a single *differential*
        # input (v_half, -v_half) and the old state order [yaw, yaw_rate,
        # pitch, pitch_rate], so B comes back 4x1. Q_lqr/R_lqr above are now
        # sized/ordered for the 2-input physics model; the augmentation code
        # below picks the right shapes automatically from B.shape[1], but the
        # *ordering* mismatch (yaw-first here vs pitch-first in Q_lqr) means
        # this path needs its own reordered Q before it'll give sane gains.
        A, B = fit_4state()

    print('Matrix A')
    print(A)
    print('Matrix B')
    print(B)


    eig_ol = np.linalg.eigvals(A)

    n = A.shape[0]      # number of plant states (4)
    m = B.shape[1]      # number of inputs: 2 for the physics model, 1 legacy

    # Q_i may be a scalar (same integral weight on every integrator, old
    # behavior) or a per-integrator vector -- [Q_i_pitch, Q_i_yaw] for the
    # MIMO (m==2) path. A scalar is broadcast to all n_z integrators.
    Q_i_arr = np.atleast_1d(np.asarray(Q_i, dtype=float))

    if m == 2:
        # MIMO LQI: 2 integrators, one per tracked output.
        #   z_pitch[k+1] = z_pitch[k] + T*(pitch[k] - 0)
        #   z_yaw[k+1]   = z_yaw[k]   + T*(yaw[k]   - target_rad)
        n_z = 2
        A_aug = np.zeros((n + n_z, n + n_z))
        A_aug[:n, :n] = A
        A_aug[n:, :2] = T_samp * np.eye(2)   # integrate pitch (state0), yaw (state1)
        A_aug[n:, n:] = np.eye(n_z)
        B_aug = np.vstack([B, np.zeros((n_z, m))])

        if Q_i_arr.size == 1:
            Q_i_use = np.full(n_z, Q_i_arr[0])
        else:
            assert Q_i_arr.size == n_z, f"Q_i must be a scalar or length-{n_z} vector [Q_i_pitch, Q_i_yaw], got size {Q_i_arr.size}"
            Q_i_use = Q_i_arr

        Q_aug = np.zeros((n + n_z, n + n_z))
        Q_aug[:n, :n] = Q_lqr
        Q_aug[n:, n:] = np.diag(Q_i_use)

        x_ss = np.array([0.0, target_rad, 0.0, 0.0])   # pitch_ss=0, yaw_ss=target
        u_ss = np.zeros(m)
    else:
        # Legacy single-input path: 1 integrator on yaw position error (state 0).
        n_z = 1
        Q_i_use = np.full(n_z, Q_i_arr[0])   # only one integrator here -- use the first weight given

        A_aug = np.zeros((n + 1, n + 1))
        A_aug[:n, :n] = A
        A_aug[n, 0] = T_samp
        A_aug[n, n] = 1.0
        B_aug = np.vstack([B, [[0.0]]])

        Q_aug = np.zeros((n + 1, n + 1))
        Q_aug[:n, :n] = Q_lqr
        Q_aug[n, n] = Q_i_use[0]

        x_ss = np.array([target_rad, 0.0, 0.0, 0.0])
        u_ss = 0.0

    R_use = R_lqr[:m, :m]   # match R to however many inputs this model has

    try:
            P_aug = solve_discrete_are(A_aug, B_aug, Q_aug, R_use)
            K_aug = np.linalg.inv(R_use + B_aug.T @ P_aug @ B_aug) @ (B_aug.T @ P_aug @ A_aug)   # m x (n+n_z)
            K  = K_aug[:, :n]     # m x n   state-feedback block
            Ki = K_aug[:, n:]     # m x n_z integral-gain block
            eig_cl = np.linalg.eigvals(A_aug - B_aug @ K_aug)

            print(f"\n  LQR design{tag}: Q=diag({Q_lqr.diagonal().tolist()} + Q_i={Q_i_use.tolist()})  R=diag({np.diag(R_use).tolist()})")
            print(f"    closed-loop poles (z) = {eig_cl}")
            for p in eig_cl:
                if 0 < abs(p) < 1:
                    tau_cl = -T_samp / np.log(abs(p))
                    print(f"      |z|={abs(p):.4f}  ->  tau ~= {tau_cl:.3f} s")

            if m == 2:
                state_labels = ["pitch", "yaw", "pitch_rate", "yaw_rate"]
                int_labels   = ["z_pitch", "z_yaw"]
                row_labels   = ["V_pitch", "V_yaw"]
                print(f"\n  ── MIMO LQI gains  (u[k] = u[k-1] - K.dx - Ki.dz) ──")
                for i, rl in enumerate(row_labels):
                    state_terms = "  ".join(f"{lbl}={K[i, j]:.5f}" for j, lbl in enumerate(state_labels))
                    int_terms   = "  ".join(f"{lbl}={Ki[i, j]:.5f}" for j, lbl in enumerate(int_labels))
                    print(f"    {rl}: {state_terms}   {int_terms}")
            else:
                print(f"\n  ── State feedback gains K  (u =  -K @ x -K_i*z) ──")
                print(f"    K_y     = {K[0,0]:.5f}   (gain on yaw position error, y - y_ss)")
                print(f"    K_yr    = {K[0,1]:.5f}   (gain on yaw rate)")
                print(f"    K_pitch = {K[0,2]:.5f}   (gain on pitch angle, pitch - pitch_ss)")
                print(f"    K_pr    = {K[0,3]:.5f}   (gain on pitch rate)")
                print(f"    K_i     = {Ki[0,0]:.5f}   (gain on integral of yaw position error)")

            result.update(ready=True, K=K, Ki=Ki, x_ss=x_ss, u_ss=u_ss, A=A, B=B)

    except Exception as _e:
        print(f"\n  LQR design failed ({_e}) — likely an unstabilizable/poorly-conditioned "
              f"(A,B) from a noisy or too-short fit. Try a longer/bigger kick.")


    return result


def lattice_weights(vec, scale):
    """
    Represent real vector `vec` (length n) EXACTLY as a convex
    combination of the 2^n integer lattice points (at fixed-point scale
    `scale`) surrounding it. Returns (corners: int64 array (m,n),
    weights: float array (m,)) with weights>=0, sum(weights)=1, and
    weights @ corners == vec*scale to float precision — an identity of
    multilinear interpolation, not an approximation.
    Zero-weight corners are dropped.
    """
    n = len(vec)
    scaled = np.asarray(vec, dtype=float) * scale
    floor_v = np.floor(scaled)
    frac = scaled - floor_v
    corners, weights = [], []
    for bits in itertools.product((0, 1), repeat=n):
        w = 1.0
        for j, b in enumerate(bits):
            w *= frac[j] if b else (1.0 - frac[j])
        if w > 1e-12:
            corners.append(floor_v + np.array(bits, dtype=float))
            weights.append(w)
    return np.array(corners, dtype=np.int64).reshape(-1, n), np.array(weights, dtype=float)


def dither_round(vec, rng):
    """
    Stochastic (dithered) rounding of each independent component of `vec`:
    draws floor or ceil per-component with P(ceil)=frac, so
    E[result] == vec exactly (unbiased) -- the 1-D/per-component analogue
    of lattice_weights' corner dithering, used for quantizing a *signal*
    (e.g. a live measurement) rather than a fixed gain vector.
    Plain np.round is a deterministic function of vec and has no such
    unbiasedness guarantee across a time-varying signal.
    """
    floor_v = np.floor(vec)
    frac = vec - floor_v
    bits = (rng.random(len(vec)) < frac).astype(np.int64)
    return floor_v.astype(np.int64) + bits


# ── LWE primitives -- copied from aero2_unified_detect.py so ciphertexts
# produced here (centered mod q, symmetric LWE, additively homomorphic
# ct_add/ct_sub/ct_smul/ct_lincomb) are structurally identical to that
# script's. ──

def centered_mod_int(z_int, q_int):
    """Centered rep for Python ints in [-q/2, q/2)."""
    z_int %= q_int
    half = (q_int - 1) // 2
    if z_int > half:
        z_int -= q_int
    return z_int


def centered_mod(z, q):
    """Return centered representative in [-q/2, q/2) (for odd q)."""
    q_int_ = int(q)
    z_int = int(z % q_int_)
    half = (q_int_ - 1) // 2
    if z_int > half:
        z_int -= q_int_
    return integer(z_int)


class DiscreteGaussianCentered:
    def __init__(self, sigma, q_int=None, tau=10.0):
        self.sigma = float(sigma)
        self.q_int = int(q_int) if q_int is not None else None

    def sample(self, size=None):
        scalar = size is None
        n_ = 1 if scalar else size
        samples = np.random.normal(0.0, self.sigma, size=n_).round().astype(np.int64)
        if self.q_int is not None:
            half = self.q_int // 2
            samples = ((samples + half) % self.q_int) - half
        return int(samples[0]) if scalar else samples


def lwe_keygen(n, q_int, secret_dist="small"):
    """Symmetric LWE key: secret s in Z_q^n."""
    return np.random.randint(0, q_int, size=n, dtype=np.int64)


def lwe_encrypt_scalar(mu_elem, s, q_int, DG):
    """
    Encrypt scalar mu -> (a,b) in Z_q, with discrete-Gaussian noise e~DG.
    Ciphertext is STORED CENTERED: a is np.ndarray(dtype=object) of charm
    integers in [-q/2,q/2), b is a charm integer in [-q/2,q/2).
    """
    n_ = len(s)
    mu = centered_mod_int(int(mu_elem), q_int)
    a_u = np.random.randint(0, q_int, size=n_, dtype=np.int64)
    a_c = np.array([centered_mod_int(int(x), q_int) for x in a_u], dtype=np.int64)
    e = int(DG.sample())
    dot_as = 0
    for ai, si in zip(a_c, s):
        dot_as += int(ai) * int(si)
    b = centered_mod_int(dot_as + e + mu, q_int)
    a = np.array([integer(int(x)) for x in a_c], dtype=object)
    b = integer(int(b))
    return (a, b)


def lwe_decrypt_scalar(ct, s, q_int):
    """Decrypt (a,b) -> charm integer.Element in centered rep."""
    a, b = ct
    mu_hat = (int(b) - int(np.dot(a, s))) % q_int
    mu_hat = centered_mod_int(mu_hat, q_int)
    return integer(mu_hat)


def ct_add(ct1, ct2, q):
    """(a,b) + (a',b') mod q, stored centered."""
    a1, b1 = ct1
    a2, b2 = ct2
    a = [centered_mod(x + y, q) for x, y in zip(a1, a2)]
    b = centered_mod(b1 + b2, q)
    return (a, b)


def ct_neg(ct, q):
    """-(a,b) mod q, stored centered."""
    a, b = ct
    return ([centered_mod(integer(-1) * x, q) for x in a], centered_mod(integer(-1) * b, q))


def ct_sub(ct1, ct2, q):
    """ct1 - ct2 mod q, stored centered."""
    return ct_add(ct1, ct_neg(ct2, q), q)


def ct_smul(alpha, ct, q):
    """Plaintext scalar alpha times ciphertext ct, stored centered. Supports negative alpha."""
    alpha = integer(alpha)
    a, b = ct
    a2 = [centered_mod(alpha * x, q) for x in a]
    b2 = centered_mod(alpha * b, q)
    return (a2, b2)


def ct_zero(n_):
    """Encryption of 0 in ciphertext space (centered)."""
    return ([integer(0) for _ in range(n_)], integer(0))


def ct_lincomb(alphas, cts, q):
    """Compute Sum alpha_i*ct_i mod q, stored centered."""
    n_ = len(cts[0][0])
    acc = ct_zero(n_)
    for alpha, ct in zip(alphas, cts):
        acc = ct_add(acc, ct_smul(alpha, ct, q), q)
    return acc


# ── Nullspace-mod-q helpers -- copied from functions.py (that module also
# pulls in matplotlib/g6k/fpylll/BKZ machinery this project doesn't need;
# these three functions are the only pieces the live LLL anomaly detector
# actually uses, so they're duplicated here to keep everything the
# controller/detector needs inside this one file). ──

def nullspace_mod_q(B, q):
    """
    Computes a basis for { x : B x = 0 (mod q) } using vectorized NumPy RREF mod q.

    Parameters
    ----------
    B : list of rows, 2D list, or 2D ndarray
    q : int (modulus)

    Returns
    -------
    basis : list of list of ints
    """
    q_int = int(q)

    # 1. Convert input to standard integer NumPy matrix
    # Use np.int64 if q < 2^31 (q^2 < 2^62 fits in int64); otherwise use object
    dtype = np.int64 if q_int < 2147483648 else object

    if isinstance(B, np.ndarray):
        M = np.mod(B.astype(dtype), q_int)
    else:
        M = np.array([[int(x) for x in row] for row in B], dtype=dtype) % q_int

    nrows, ncols = M.shape
    if nrows == 0 or ncols == 0:
        return []

    pivot_cols = []
    r = 0  # current pivot row index

    for c in range(ncols):
        # 2. Find an invertible pivot in column c (gcd(val, q) == 1)
        pivot = None
        for i in range(r, nrows):
            val = int(M[i, c]) % q_int
            if val != 0 and math.gcd(val, q_int) == 1:
                pivot = i
                break

        if pivot is None:
            continue  # Column cannot be pivoted (either all zero or non-invertible mod q)

        # 3. Swap pivot row into position r
        if pivot != r:
            M[[r, pivot]] = M[[pivot, r]]

        # 4. Normalize pivot row so M[r, c] == 1 (mod q)
        inv = pow(int(M[r, c]), -1, q_int)
        M[r, :] = (M[r, :] * inv) % q_int

        # 5. Vectorized Elimination of column c across ALL other rows
        # Creates a 2D column vector of factors: shape (nrows, 1)
        col_factors = M[:, c].copy()
        col_factors[r] = 0  # Do not eliminate the pivot row itself

        # Broadcast subtraction: M[i, j] = (M[i, j] - col_factors[i] * M[r, j]) % q
        M = (M - col_factors[:, None] * M[r, :]) % q_int

        pivot_cols.append(c)
        r += 1
        if r == nrows:
            break

    # 6. Extract free columns and construct nullspace basis
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(ncols) if c not in pivot_set]

    basis = []
    for f in free_cols:
        x = np.zeros(ncols, dtype=np.int64)
        x[f] = 1
        # In RREF: x[pcol] = -M[row_idx, f] (mod q)
        for row_idx, pcol in enumerate(pivot_cols):
            x[pcol] = (-int(M[row_idx, f])) % q_int
        basis.append(x.tolist())

    return basis


def left_nullspace_mod_q(A, q):
    """
    Vectorized left nullspace: returns d such that d^T A = 0 (mod q).
    """
    q_int = int(q)
    dtype = np.int64 if q_int < 2147483648 else object

    if isinstance(A, np.ndarray):
        A_np = A.astype(dtype)
    else:
        A_np = np.array([[int(x) for x in row] for row in A], dtype=dtype)

    # B = A^T
    B = A_np.T % q_int
    return nullspace_mod_q(B, q_int)


def recenter_mod_q(B_np, q_int):
    """Vectorized C-level mod-q centering: maps integers back to [-q/2, q/2]."""
    q2 = q_int // 2
    return np.mod(B_np + q2, q_int) - q2


# ── Remote plant proxy -- drop-in stand-in for pal.products.aero2.Aero2 ────
# Real hardware (Aero2(hardware=1)) needs the Quanser HIL/USB driver stack,
# which is native-Windows; charm-crypto/blaster (this project's crypto/
# lattice deps) are Linux-only. The two don't reliably coexist in one
# process/OS, so the real-hardware deployment splits into two processes:
#   - aero2_plant_bridge.py: runs natively on the machine physically wired
#     to the Aero2 (zero crypto deps -- just pal/numpy/stdlib sockets),
#     owns the Aero2 object, and exposes a tiny newline-delimited-JSON TCP
#     protocol (see that file's docstring for the exact message formats).
#   - this script (aero2_lwe_run.py + aero2_tools.py, wherever charm/
#     blaster actually run -- e.g. WSL on the same box): connects to the
#     bridge and uses RemotePlantProxy in place of a direct Aero2 object.
# RemotePlantProxy implements exactly the subset of the Aero2 API this
# project touches, so swapping backends is a single assignment at hardware
# setup time -- nothing downstream needs to know which one it's talking to.
class RemotePlantProxy:
    """Drop-in stand-in for pal.products.aero2.Aero2, backed by a TCP
    connection to aero2_plant_bridge.py. Implements read_analog_encoder_
    other_channels(), .pitchAngle/.yawAngle/.pitchRate/.yawRate,
    write_voltage(), write_led(), and zero_encoders() (in place of the
    direct-Aero2 backend's myAero2.card.set_encoder_counts(...) call).
    One "read" or "write" call here = one blocking request/reply round
    trip on the wire -- at FREQUENCY_HZ=10 (100ms/step budget) this is
    negligible overhead on localhost or LAN, but it IS a new source of
    per-step jitter that didn't exist with a direct Aero2 object; worth
    re-checking the LOOP TIMING report (noise_cal mode) once running over
    the bridge, same as any other timing-affecting change.
    """

    def __init__(self, host, port, connect_timeout=10.0):
        self._sock = socket.create_connection((host, port), timeout=connect_timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._buf = bytearray()
        self.pitchAngle = self.yawAngle = self.pitchRate = self.yawRate = 0.0

    def _send(self, obj):
        self._sock.sendall((json.dumps(obj) + "\n").encode())

    def _recv(self):
        while b"\n" not in self._buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("aero2_plant_bridge connection closed unexpectedly")
            self._buf += chunk
        line, _, rest = self._buf.partition(b"\n")
        self._buf = bytearray(rest)
        return json.loads(line.decode())

    def zero_encoders(self):
        """Bridge-side equivalent of the direct-Aero2 backend's
        myAero2.card.set_encoder_counts(...) startup call."""
        self._send({"cmd": "zero"})
        self._recv()

    def read_analog_encoder_other_channels(self):
        self._send({"cmd": "read"})
        reply = self._recv()
        self.pitchAngle = reply["pitch"]
        self.yawAngle   = reply["yaw"]
        self.pitchRate  = reply["pitch_rate"]
        self.yawRate    = reply["yaw_rate"]

    def write_voltage(self, v_pitch, v_yaw):
        self._send({"cmd": "write", "v": [float(v_pitch), float(v_yaw)]})
        self._recv()   # wait for ack -- keeps the protocol strictly synchronous, no reordering risk

    def write_led(self, color):
        # Best-effort/cosmetic only (run/done status LED on the physical
        # unit) -- never worth failing or stalling the control loop over.
        try:
            self._send({"cmd": "led", "rgb": [float(c) for c in color]})
            self._recv()
        except Exception:
            pass

    def close(self):
        try:
            self._send({"cmd": "stop"})
            self._recv()
        except Exception:
            pass
        self._sock.close()
