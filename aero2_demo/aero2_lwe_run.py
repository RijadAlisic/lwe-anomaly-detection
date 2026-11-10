"""
aero2_lwe_run.py — clean, two-mode entry point for the LWE-encrypted,
lattice-quantized Aero2 pitch/yaw controller.

Everything that used to run before "real" LWE control in
aero2_motor_mapping_test.py has been dropped:
  - Phase 0 (open-loop PRBS excitation) was already dead code there
    (SYSID_TIME=0 makes design_lqr() use the nominal physics model
    directly and never touches the PRBS logs).
  - Phase 2 (plain-float LQI) and Phase 3 (unencrypted lattice-quantized
    LQI) were both muted (while-condition * 0) and unused downstream.
  - The Luenberger/LQE shadow observer (Q_l/R_l/desired_poles/L_obs) was
    only ever driven by Phase 2, so it goes with it. The DEADBEAT/
    autonomous shadow observer (static FIR over the residual, used by
    Phase 3/4) is kept, since control_detect mode still uses it.

Only two project files are needed to run this: this script and
aero2_tools.py (every LWE/lattice/nullspace helper now lives there — the
old functions.py dependency, which also pulled in matplotlib/g6k/fpylll/BKZ
machinery this script never touches, has been dropped).

Three modes, selected by the MODE constant in the USER CONFIG block below:
  "noise_cal"      -- Phase 4a: A.s=0 noise-calibration run. Drives the real
                       plant with the real dithered lattice controller, but
                       decrypts under a zeroed secret so mu_hat = e+mu
                       directly -- measures std(r-y), the real residual noise
                       floor, before ever touching a genuine secret.
  "control_detect" -- Phase 4 + the live BLASTER/LLL anomaly detector: real
                       secret, real encryption, real-time detection on a
                       background thread, with periodic LWE key rotation.
  "both"           -- runs noise_cal immediately followed by control_detect,
                       back to back, in a single invocation (one hardware
                       session, one scope window) -- convenient for a
                       calibrate-then-attack-test workflow without having to
                       relaunch the script and re-home the plant in between.

ALL tunable knobs (mode, debug switches, LQR weights, physics parameters,
deadbeat window length, LWE parameters, signal-scaling parameters, detector
parameters, key-rotation policy, run lengths, RNG seeds) live in the single
USER CONFIG block right below the imports. Nothing past that block should
need editing for a normal parameter sweep.
"""

import time
import sys
import os
import math
import threading
import queue
from collections import deque
import numpy as np
from pal.utilities.scope import MultiScope
import pyqtgraph as pg   # scope.py is built on this -- used directly here to draw
                          # detection-window markers (shaded regions) on top of the
                          # existing live plots, without touching scope.py itself
from scipy.signal import place_poles
from scipy.stats import chi2 as _chi2
from blaster import reduce as _blaster_reduce
import blaster.blaster as _blaster_internals   # gives us access to (and lets us monkeypatch)
                                                # is_weakly_lll_reduced(), the convergence check
                                                # BLASTER's own internal lll_reduce() while-loop
                                                # calls on every iteration -- we piggyback our
                                                # det_pause wait on that call so a pause can take
                                                # effect *inside* a single _blaster_reduce() call,
                                                # not just between our own outer passes (see the
                                                # det_pause install block further down for details).
from charm_compat import integer   # charm-crypto if available, else a pure-Python int fallback
# NOTE: pal.products.aero2.Aero2 is imported lazily, only inside the
# PLANT_BACKEND == "local" branch of the HARDWARE section below -- it binds
# the Quanser HIL/USB driver stack at import time on some builds, which
# isn't installed (and isn't needed) in a tcp_bridge-backend environment
# (e.g. WSL, talking to aero2_plant_bridge.py instead).

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from aero2_tools import (
    elapsed_time, build_physics_model, design_lqr,
    lattice_weights, dither_round,
    centered_mod_int, centered_mod, DiscreteGaussianCentered,
    lwe_keygen, lwe_encrypt_scalar, lwe_decrypt_scalar,
    ct_add, ct_neg, ct_sub, ct_smul, ct_zero, ct_lincomb,
    left_nullspace_mod_q, recenter_mod_q,
    RemotePlantProxy,
)


# ══════════════════════════════════════════════════════════════════════════
# ══════════════════════════════  USER CONFIG  ═══════════════════════════════
# Every tunable knob in this script. Nothing below this block should need
# editing for a normal experiment/parameter sweep.
# ══════════════════════════════════════════════════════════════════════════

# ── Mode ─────────────────────────────────────────────────────────────────
#   "noise_cal"      -- Phase 4a noise-calibration run only (A.s=0)
#   "control_detect" -- real Phase 4 control + live LLL anomaly detector
#   "both"           -- noise_cal immediately followed by control_detect,
#                        back to back, in one script invocation
MODE = "control_detect"

# Forces e=0 on every LWE encryption (sigma=0 -> DG always samples 0), so
# the ciphertext/decrypt arithmetic can be sanity-checked in isolation from
# the actual LWE noise. Leave False for real runs.
DEBUG_ZERO_NOISE = False

# Forces A.s=0 for every ciphertext in control_detect mode, the same trick
# noise_cal mode always uses internally (there via its own fixed s4_zero
# secret) -- but applied here to the REAL secret s4 (both at startup and at
# every key-rotation event), so the actual control_detect code path runs
# unmodified (real encrypt/decrypt calls, real detector, real key rotation
# bookkeeping) while b = A.s + e + mu collapses to b = e + mu, i.e.
# decryption trivially recovers e+mu with no key-dependent term to get
# wrong. Useful for isolating "is the arithmetic/plumbing correct" from
# "is the secret/decryption right" without switching MODE or duplicating
# the control loop. Leave False for real runs.
DEBUG_ZERO_KEYS = False

# ── Sensor (false-data-injection) attack, control_detect mode only ─────────
# Simulates a compromised/spoofed yaw encoder: for ATTACK_DURATION seconds
# starting at ATTACK_TIME_START (on the shared scope timeline, _gt() --
# the same clock as the live plot's x-axis), a constant bias
# ATTACK_BIAS_DEG is added directly to the raw yawAngle reading BEFORE it
# is quantized/encrypted -- i.e. the
# attack happens at the sensor, not in the ciphertext math, so it also fools
# the controller (real physical mis-tracking) exactly like a real FDI
# attacker would, while the encrypted r-y stream fed to the live LLL
# detector carries the same corrupted measurement. This is the thing the
# detector is supposed to catch.
#
# DEBUG_ZERO_ATTACK follows the same "force-a-term-to-zero" naming as
# DEBUG_ZERO_NOISE/DEBUG_ZERO_KEYS above: True forces the injected bias to
# 0, i.e. disables the attack entirely (safe default for normal runs). Set
# False to actually inject it.
ATTACK_TIME_START = 90.0  # s on the scope timeline (_gt(), same clock as the live plot's x-axis)
                            # when the attack begins
ATTACK_DURATION    = 10.0  # s -- attack stays active this long, then reverts to the true reading
ATTACK_BIAS_DEG    = 45.0   # deg -- constant additive spoof applied to yawAngle while active
DEBUG_ZERO_ATTACK  = False  # True = attack disabled (bias forced to 0). Set False to enable the attack.

# ── Timing / hardware ───────────────────────────────────────────────────
FREQUENCY_HZ  = 10.0    # control-loop rate. Real achieved rate is bounded
                         # above by per-step crypto compute cost -- see the
                         # LOOP TIMING report noise_cal mode prints.
V_MAX         = 10.0    # per-channel voltage limit (V_pitch, V_yaw)
TARGET_REF_DEG = 20.0   # commanded yaw setpoint (pitch setpoint is always 0)

# hardware gave me mean +300 and std 212 <- TWICE

# "local"      -- direct pal.products.aero2.Aero2 object in THIS process
#                 (virtual QLab sim, or real hardware=1 IF this process's
#                 environment has both the Quanser HIL driver stack AND
#                 charm-crypto/blaster available together -- typically only
#                 true for the virtual sim).
# "tcp_bridge" -- talk to a separate aero2_plant_bridge.py process (see that
#                 file) over TCP instead of touching Aero2 directly. Use
#                 this for real hardware if the HIL driver (native Windows)
#                 and charm/blaster (Linux-only) can't coexist in one
#                 process on the lab PC -- run aero2_plant_bridge.py
#                 natively there, and this script wherever charm/blaster do
#                 work (e.g. WSL on the same box), pointed at BRIDGE_HOST.
PLANT_BACKEND = "local"
BRIDGE_HOST   = "127.0.0.1"   # aero2_plant_bridge.py's host. If this script runs inside WSL on
                               # the SAME Windows machine as the bridge: WSL2's "localhost" does
                               # NOT reliably reach the Windows host on older/NAT-mode WSL setups
                               # (only newer "mirrored" networking mode guarantees this) -- if
                               # 127.0.0.1 fails to connect, get the Windows host's IP as seen
                               # from inside WSL via `cat /etc/resolv.conf` (nameserver line) or
                               # `ip route show | grep default`, and use that instead.
BRIDGE_PORT   = 8765

# ── LQR cost matrices  (x = [pitch, yaw, pitch_rate, yaw_rate], u = [V_pitch, V_yaw]) ──
Q_LQR_DIAG = [1500.0, 1500.0, 10.0, 10.0]   # state cost
Q_I_DIAG   = [0.0, 500.0]                 # integral-state cost [Q_i_pitch, Q_i_yaw] -- separate
                                             # weights for the two LQI integrators; raise Q_i_pitch/
                                             # Q_i_yaw independently to trim out steady-state bias
                                             # faster on that axis specifically (higher = more
                                             # aggressive integral action / faster bias correction,
                                             # at the cost of more overshoot risk on that axis).
R_LQR_DIAG = [0.1, 0.1]                     # input cost [V_pitch, V_yaw]

# ── Physics model (QA2C hardware parameters), continuous-time ─────────────
JP, JY   = 0.0211, 0.0221     # pitch/yaw inertia (kg*m^2)
DP, DY   = 0.0053, 0.0062     # pitch/yaw damping
MG       = 0.0153             # gravity coupling (pitch restoring torque)
KPP, KYY = 0.0011, 0.0047     # pitch/yaw motor gains
KPY, KYP = 0.0021, -0.0027    # cross-axis input coupling

# ── Deadbeat / autonomous shadow observer ─────────────────────────────────
# x_hat[k] = Sum_{i=1..DEADBEAT_LEN} Wi.dy[k-i] + c_bar_obs, Wi = M_obs^(i-1).L.
# The exact-reconstruction minimum is 4 (== state dim n): with DEADBEAT_POLES
# placed at ~0, M_obs is nilpotent with M_obs^4~=0, so 4 taps recover x_hat
# exactly (in the noiseless case). Raising DEADBEAT_LEN adds redundant taps
# (more independent ciphertexts averaged into the estimate, which can help
# average down LWE noise) at the cost of needing an older window of y (more
# lag before x_hat reflects new plant motion); lowering it below 4 trades
# reconstruction accuracy for less lag. DEADBEAT_POLES must stay length-4
# (place_poles needs exactly n=4 poles for this 4-state system) regardless
# of DEADBEAT_LEN.
DEADBEAT_LEN   = 4
DEADBEAT_POLES = np.array([1e-5, -1e-5, 2e-5, -2e-5])

# ── LWE encryption ─────────────────────────────────────────────────────
LWE_N      = 64              # dimension of the LWE secret vector
LWE_Q_BASE = 2**23 + 9       # prime modulus (~8.4M)
LWE_SIGMA  = 50               # discrete-Gaussian std for encryption noise

# ── Signal / gain-lattice scaling ─────────────────────────────────────────
# cs_enc (state-quantization scale) is picked at runtime as the largest
# value such that the single worst (row,state) term of K_LQR.dx still
# leaves CS_ENC_MARGIN of q/2 headroom, assuming |dx| never exceeds
# DX_BOUND_DEG; CS_ENC_FRACTION then spends only that fraction of the
# resulting max, for extra safety margin.
C_RAW_ENC       = 1     # gain-lattice fixed-point scale (K_LQR, u_ss, observer FIR rows)
DX_BOUND_DEG    = np.array([20.0, 60.0, 200.0, 200.0])   # [pitch, yaw, pitch_rate, yaw_rate] worst-case bound
CS_ENC_MARGIN   = 0.5   # fraction of q/2 headroom to spend, worst-case single term
CS_ENC_FRACTION = 0.7   # fraction of the resulting max cs_enc actually used

# Z_BOUND_DEG_S bounds the LQI integral state z = integral(dt * [pitch-pitch_ss,
# yaw-yaw_ss]) -- units deg*s. Sized the same way cs_enc is sized from
# DX_BOUND_DEG: the largest cs_z such that the worst (row,z) term of Ki.z still
# leaves CS_ENC_MARGIN of q/2 headroom, assuming |z| never exceeds this bound.
# Also doubles as a simple clamp-style anti-windup limit -- z is clipped to
# +/-Z_BOUND_DEG_S every step, so a saturated/slammed plant can't wind the
# integrator up past what the lattice/cs_z headroom was sized for.
Z_BOUND_DEG_S   = np.array([5.0, 300.0])   # [z_pitch, z_yaw] anti-windup clamp, deg*s -- yaw bound
                                            # opened way up (was 15.0) so the integrator can no
                                            # longer saturate/cap out under normal operation; note
                                            # this also shrinks cs_z proportionally (see _cs_z_max
                                            # below), i.e. coarser z-quantization / more integral-path
                                            # decryption noise in exchange for headroom -- watch the
                                            # printed "Worst-case summed integral-path noise: V_yaw~=..."
                                            # line at startup if voltage noise looks worse than before.

# ── Live LLL anomaly detector (control_detect mode only) ──────────────────
DET_DELTA_LLL       = 1.012031   # assumed attacker's root Hermite factor -> sizes DET_WINDOW
DET_NUM_DOF         = 10         # shortest lattice vectors used for the test statistic
DET_Z_THRESH        = 3.0        # norm_thrs = q / (2*sigma_eff*z_thresh)
DET_MAX_CYCLES      = 2          # give up (use best-effort reduction) after this many full depth-restart cycles
DET_INCREMENT       = 10         # depth step on stall
DET_IMPROVE_THRESH  = 0.80       # require >=20% shortening to stay at current depth
DET_VERBOSE         = True       # per-LLL-pass diagnostics -- set False to quiet the console down

# The live LLL detector thread's big-int mod arithmetic doesn't release the
# GIL the way numpy/BLAS calls do, so a live LLL pass can visibly steal
# scheduler time from the real-time control loop -- the main loop prints a
# "[timing ...] SLOW loop" warning when it notices this, and (if enabled
# below) pauses the detector thread until the control loop has been back on
# schedule for a while, so it can catch up without fighting the detector for
# the GIL.
DET_PAUSE_ON_SLOW_LOOP = True    # pause the LLL detector thread when the control loop falls behind
DET_PAUSE_SLOW_FACTOR  = 5     # an iteration counts as "slow" once elapsed > this * T_samp
DET_PAUSE_RECOVER_S    = 7.0     # keep the detector paused until the control loop has been
                                  # back on schedule for this many seconds straight
DET_PAUSE_POLL_S       = 0.02    # how often the paused detector thread re-checks det_pause

# Cheap, safe, global latency reducer: CPython's default GIL "switch interval"
# is 5ms -- when a pure-Python bytecode loop is running, that's how long
# another waiting thread may have to wait for a handoff. Shortening it doesn't
# help against a single non-yielding native call that never returns to the
# bytecode eval loop (nothing at the Python level can preempt that), but it
# does reduce the worst-case handoff latency everywhere else (between
# statements, at the QR / native-LLL / reduce / matmul sub-step boundaries
# inside a single _blaster_reduce() call, etc.), so it's left on.
sys.setswitchinterval(0.001)   # 1ms, down from the CPython default of 5ms

# Live scope markers -- shade the [t_start, t_end] span each detector window
# actually covered directly on the scope plots (translucent band + a solid
# boundary line at each edge), colored by whether that window got flagged.
# Purely cosmetic/diagnostic -- doesn't touch detection logic at all.
DET_MARK_ENABLE   = True

# After the control run ends, the detector finishes its in-flight window, then
# backfills time spans no live window ever covered (the live worker always grabs
# the LATEST window, so a slow LLL pass or a pause leaves holes). Only holes of
# at least DET_FILL_MIN_GAP_S seconds are filled, tiled with DET_WINDOW-sample
# windows from the logged ciphertexts of the same key epoch. Backfilled spans
# are shaded like live ones. DET_EXIT_WHEN_DONE=True exits right after the
# detector is finished instead of keeping the scope window open.
DET_FILL_GAPS       = True
DET_FILL_MIN_GAP_S  = 20.0
DET_EXIT_WHEN_DONE  = False
DET_MARK_AXES     = (0,)                       # which scope.axes[] rows to shade (0=Yaw -- only axis kept in scope)
DET_MARK_COLOR_OK = (80, 160, 80, 40)          # nominal window: translucent green, (r,g,b,alpha)
DET_MARK_COLOR_FLAG = (200, 60, 60, 70)        # flagged window: translucent red, more opaque

# sigma_eff_det is the assumed per-sample std(r-y) the chi^2 test's z-scores
# and the LLL norm threshold are calibrated against -- this is a REAL
# noise-floor number, not something to leave to a theoretical formula by
# default. Same units/scale as noise_cal mode's "std (counts)" printout
# (cs_enc-scaled integer counts, i.e. the raw decrypted-residual domain).
# Three ways to set it, checked in this order:
#   1) DET_SIGMA_EFF_OVERRIDE set to a number (e.g. from a PAST noise_cal
#      run's printed "std (counts)" value) -> always used verbatim.
#   2) DET_SIGMA_EFF_OVERRIDE left None AND MODE="both" -> measured live,
#      this run, from the noise_cal phase that just completed
#      (ry_arr.std()) -- the actual empirical noise floor, dither and
#      tracking-error contributions included.
#   3) DET_SIGMA_EFF_OVERRIDE left None AND MODE="control_detect" (no
#      noise_cal phase ran, so no measurement exists) -> falls back to the
#      theoretical sqrt(2)*LWE_SIGMA (LWE noise only, ignores dither/tracking
#      contributions -- run "both" or set the override for a real number).
DET_SIGMA_EFF_OVERRIDE = 200

# ── LWE key rotation (control_detect mode only) ────────────────────────────
# Every detection window (DET_WINDOW ciphertexts, all under the same secret)
# gives an attacker/observer more samples to feed a lattice-reduction attack
# against that key. KEY_ROTATION_WINDOWS caps how many multiples of
# DET_WINDOW may elapse before the LWE secret is rotated (regenerated) --
# smaller = less ciphertext exposure per key (more secure) but more keygen
# overhead and a brief DEADBEAT_LEN-step observer warm-up each rotation;
# larger = smoother operation but more samples accumulate under one key.
# Set to None or <=0 to disable rotation entirely.
KEY_ROTATION_WINDOWS = 20

# ── Mode run-lengths ───────────────────────────────────────────────────
T_SETTLE_NOISE_S = 20.0    # noise_cal: settle time before logging starts
T_NOISE_TEST_S   = 120.0   # noise_cal: measurement window length after settling
T_CONTROL_RUN_S  = 300.0   # control_detect: total run length

# ── RNG seeds (reproducible dithering) ─────────────────────────────────
RNG_SEED_NOISE_CAL = 13
RNG_SEED_CONTROL   = 11

# ── Diagnostics ─────────────────────────────────────────────────────────
PRINT_INTERVAL_S = 2.0   # throttle interval for per-step diagnostic prints

assert MODE in ("noise_cal", "control_detect", "both"), f"unknown MODE={MODE!r}"
assert len(DEADBEAT_POLES) == 4, "DEADBEAT_POLES must have exactly 4 entries (state dim n=4)"
assert PLANT_BACKEND in ("local", "tcp_bridge"), f"unknown PLANT_BACKEND={PLANT_BACKEND!r}"


# ══════════════════════════════════════════════════════════════════════════
# DERIVED SETUP -- everything below reads from USER CONFIG above; no magic
# numbers past this point.
# ══════════════════════════════════════════════════════════════════════════
T_samp       = 1.0 / FREQUENCY_HZ
V_max        = V_MAX
output_limit = 2 * V_max
frequency    = FREQUENCY_HZ   # kept for the print statements below

USE_OBSERVER_FEEDBACK = False   # unused (no controller branch currently reads it;
                                 # kept as a placeholder toggle for future work)


# ════════════════════════════════════════════════════════════════════════════
# HARDWARE -- see PLANT_BACKEND in USER CONFIG. Either way, everything below
# this block only ever touches myAero2 through the same small API (read_
# analog_encoder_other_channels / .pitchAngle etc. / write_voltage /
# write_led), so nothing downstream needs to know which backend is live.
# ════════════════════════════════════════════════════════════════════════════
if PLANT_BACKEND == "local":
    from pal.products.aero2 import Aero2
    myAero2 = Aero2(id=0, hardware=0, readMode=0, frequency=500)
    myAero2.card.set_encoder_counts(
        myAero2.READ_ENCODER_CHANNELS, len(myAero2.READ_ENCODER_CHANNELS),
        np.array([0, 0, 0, 0], dtype=np.float64)
    )   # zero encoders in software — same as the physical reset button
else:   # "tcp_bridge" -- see aero2_plant_bridge.py, run natively on the machine wired to the real Aero2
    print(f"\n  Connecting to aero2_plant_bridge.py at {BRIDGE_HOST}:{BRIDGE_PORT}...")
    myAero2 = RemotePlantProxy(BRIDGE_HOST, BRIDGE_PORT)
    myAero2.zero_encoders()
    print("  Connected, encoders zeroed.")

color_run  = np.array([0, 1, 0], dtype=np.float64)
color_done = np.array([1, 0, 0], dtype=np.float64)


def _gt():
    """Elapsed time (s) on the shared scope timeline."""
    return elapsed_time(_scope_t0)


# ────────────────────────────────────────────────────────────────────────────
# Physics-based continuous-time model (QA2C parameters), ZOH-discretized.
#   x = [pitch, yaw, pitch_rate, yaw_rate]ᵀ   (rad, rad, rad/s, rad/s)
#   u = [V_pitch, V_yaw]ᵀ  ->  rotor0, rotor1 voltages, written directly
#       via myAero2.write_voltage(V_pitch, V_yaw) — no differential mixing.
# ────────────────────────────────────────────────────────────────────────────
Q_lqr = np.diag(Q_LQR_DIAG)
Q_i   = np.array(Q_I_DIAG)   # [Q_i_pitch, Q_i_yaw] -- design_lqr() broadcasts a scalar or takes this as-is
R_lqr = np.diag(R_LQR_DIAG)

A_c = np.array([
    [0.0,     0.0, 1.0,     0.0    ],
    [0.0,     0.0, 0.0,     1.0    ],
    [-MG/JP,  0.0, -DP/JP,  0.0    ],
    [0.0,     0.0, 0.0,     -DY/JY ],
])
B_c = np.array([
    [0.0,      0.0     ],
    [0.0,      0.0     ],
    [KPP/JP,   KPY/JP  ],
    [KYP/JY,   KYY/JY  ],
])
C_phys = np.array([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0]])
D_phys = np.zeros((2, 2))

SYSID_TIME = 0   # design_lqr uses the nominal physics model directly when 0
                  # (no empirical PRBS system-ID pass in this file)


scope = MultiScope(rows=1, cols=1, title="Aero2 LWE Control", fps=30)
scope.addAxis(row=0, col=0, timeWindow=SYSID_TIME, xLabel="t (s)", yLabel="Yaw (deg)", yLim=(-30, 30))
scope.axes[0].attachSignal(name="yaw_ref", width=3)
scope.axes[0].attachSignal(name="yaw_meas", width=3)
_scope_t0 = time.perf_counter()   # shared timeline origin

_det_mark_items = []   # keeps refs to LinearRegionItem/InfiniteLine objects alive
                        # (pyqtgraph doesn't hold a strong ref once added to a plot,
                        # and PlotItem.addItem() alone is not enough to prevent gc
                        # in some pyqtgraph versions)


_det_mark_watermark = float("-inf")   # rightmost t_end already shaded by a PREVIOUS window's
                                        # marker -- windows highly overlap in real time (DET_WINDOW
                                        # spans much longer than one LLL pass takes), so without this
                                        # every new window's shaded band would re-draw on top of the
                                        # tail of the previous one(s); we only ever draw the NEW,
                                        # not-yet-shaded slice [max(t_start, watermark), t_end]


def _mark_detection_window(t_start, t_end, flagged, tag="", clip=None):
    """Shade the NOT-YET-SHADED part of [t_start, t_end] on the configured scope
    axes -- called from the MAIN loop only (never from the detector thread) so
    all Qt/pyqtgraph object creation happens on the GUI thread, same as the
    rest of the scope calls. Consecutive windows heavily overlap in real time,
    so only the portion past the running watermark (the furthest t_end already
    drawn by an earlier call) is actually shaded -- previously-drawn territory
    is left alone rather than being re-shaded/overlaid."""
    if not DET_MARK_ENABLE:
        return
    global _det_mark_watermark
    if clip is not None:
        # backfill: shade only the part inside the gap being filled, ignore the watermark
        draw_start, t_end = max(t_start, clip[0]), min(t_end, clip[1])
    else:
        draw_start = max(t_start, _det_mark_watermark)
        _det_mark_watermark = max(_det_mark_watermark, t_end)
    color = DET_MARK_COLOR_FLAG if flagged else DET_MARK_COLOR_OK
    edge_color = (200, 30, 30) if flagged else (60, 120, 60)
    for _axi in DET_MARK_AXES:
        if _axi >= len(scope.axes):
            continue
        _plot = scope.axes[_axi].plot
        if draw_start < t_end:
            region = pg.LinearRegionItem(
                values=(draw_start, t_end),
                movable=False,
                brush=pg.mkBrush(*color),
                pen=pg.mkPen(color=edge_color, width=1),
            )
            region.setZValue(-10)   # keep it behind the signal traces
            _plot.addItem(region)
            _det_mark_items.append(region)
        if flagged:
            # extra emphasis for a flagged window: a labelled boundary line at t_end
            # (solid pen -- avoid relying on a specific Qt PenStyle enum path,
            # which differs between the PyQt5/PyQt6 bindings pyqtgraph may be on)
            line = pg.InfiniteLine(pos=t_end, angle=90,
                                    pen=pg.mkPen(color=edge_color, width=2),
                                    label=f"anomaly{tag}", labelOpts={'position': 0.95, 'color': edge_color})
            _plot.addItem(line)
            _det_mark_items.append(line)


# ────────────────────────────────────────────────────────────────────────────
# LQR design -- SYSID_TIME=0 means design_lqr() uses the nominal physics
# model (A_c/B_c/C_phys/D_phys) directly; the u/y/yr/pitch/pitchrate
# arguments below are only consumed by the empirical-fit branch (sysid_time
# != 0), so empty arrays are fine here.
# ────────────────────────────────────────────────────────────────────────────
_empty = np.array([], dtype=float)
_res0 = design_lqr(_empty, _empty, _empty, _empty, _empty,
                    target_rad=np.deg2rad(TARGET_REF_DEG),
                    T_samp=T_samp, Q_lqr=Q_lqr, Q_i=Q_i, R_lqr=R_lqr, sysid_time=SYSID_TIME,
                    A_c=A_c, B_c=B_c, C_phys=C_phys, D_phys=D_phys, label="LQR design")
lqr_K, lqr_Ki, lqr_x_ss, lqr_u_ss = _res0["K"], _res0["Ki"], _res0["x_ss"], _res0["u_ss"]
print(f"  lqr_K (rows=[pitch,yaw], cols=[pitch,yaw,pitch_rate,yaw_rate]):\n{lqr_K}")
print(f"  lqr_Ki (rows=[pitch,yaw], cols=[z_pitch,z_yaw]):\n{lqr_Ki}")
print(f"  lqr_x_ss={lqr_x_ss}  lqr_u_ss={lqr_u_ss}")

A_phys, B_phys = build_physics_model(A_c, B_c, C_phys, D_phys, T_samp)

# ────────────────────────────────────────────────────────────────────────────
# DEADBEAT / AUTONOMOUS shadow observer -- static FIR, ciphertext-space
# analogue of a Luenberger observer, unrolled over a bounded DEADBEAT_LEN-
# sample residual window instead of a persistent recursive estimate
# (encryption tolerates a fixed finite-window FIR -- bounded noise growth --
# but not an ever-recursing state, whose noise would grow without bound step
# after step). Only USED by control_detect mode below, but cheap enough to
# always build here alongside the rest of the shared LQR/physics setup.
#
#   x_hat[k] = W1.dy[k-1] + ... + W_{DEADBEAT_LEN}.dy[k-DEADBEAT_LEN] + c_bar_obs
#   Wi = M_obs^(i-1) . L_obs_db,  dy[k-i] = y[k-i] - y_ss
# ────────────────────────────────────────────────────────────────────────────
A_cl0 = A_phys - B_phys @ lqr_K                     # expected closed loop under u=u_ss-K.(x-x_ss)
_obs_pp  = place_poles(A_cl0.T, C_phys.T, DEADBEAT_POLES)   # deadbeat-ish, distinct poles (place_poles
                                                              # requires multiplicity <= rank(C.T)=2)
L_obs_db = _obs_pp.gain_matrix.T                     # 4x2 deadbeat observer gain
M_obs    = A_cl0 - L_obs_db @ C_phys                 # nilpotent (M_obs^4 ~= 0) error-dynamics matrix
const_obs = B_phys @ (lqr_u_ss + lqr_K @ lqr_x_ss)   # fixed vector -- B times the assumed-u's constant part

# Wi = M_obs^(i-1).L_obs_db for i=1..DEADBEAT_LEN, and c_bar_obs =
# Sum_{k=0}^{DEADBEAT_LEN-1} M_obs^k . const_obs (generalizes the plain
# geometric-sum-of-a-nilpotent-matrix identity used for the exact DEADBEAT_LEN=4 case).
_W_obs_pows = [np.linalg.matrix_power(M_obs, k) @ L_obs_db for k in range(DEADBEAT_LEN)]
c_bar_obs = sum(np.linalg.matrix_power(M_obs, k) for k in range(DEADBEAT_LEN)) @ const_obs
Wobs_rows = [np.concatenate([_W_obs_pows[k][i] for k in range(DEADBEAT_LEN)]) for i in range(4)]   # 4 rows, (2*DEADBEAT_LEN)-vec each

# FIR stated in terms of the RESIDUAL dy=y-y_ss (not raw y, which sits at a
# nonzero operating point basically always -- e.g. yaw~30deg -- where a
# coarse c_raw_enc=1 gain lattice would make single-corner dithering swings
# show up as huge jumps in x_hat every step). Constant y_ss contribution
# folded into c_bar_obs.
y_ss_obs  = lqr_x_ss[:2]                                                       # [pitch_ss, yaw_ss]
c_bar_obs = c_bar_obs + sum(_W_obs_pows) @ y_ss_obs

print(f"\n  Deadbeat/autonomous observer (DEADBEAT_LEN={DEADBEAT_LEN}): eig(M_obs)={np.linalg.eigvals(M_obs)}")
print(f"  ||M_obs^{DEADBEAT_LEN}||={np.linalg.norm(np.linalg.matrix_power(M_obs, DEADBEAT_LEN)):.2e}  "
      f"(should be ~0 at DEADBEAT_LEN=4 -- confirms deadbeat/nilpotent)")
print(f"  c_bar_obs = {c_bar_obs}")


# ════════════════════════════════════════════════════════════════════════════
# LWE parameters -- shared by both modes. Same convention (n, prime modulus
# q, discrete-Gaussian sigma) as aero2_sysid.py / aero2_unified_detect.py.
# ════════════════════════════════════════════════════════════════════════════
lwe_n     = LWE_N               # dimension of LWE secret vector
lwe_q     = integer(LWE_Q_BASE)   # prime modulus (~8.4M)
lwe_q_int = int(lwe_q)
lwe_sigma = LWE_SIGMA            # discrete Gaussian std for encryption noise

# Debug switch: forces e=0 on every encryption (sigma=0 -> DG always samples
# 0), so the ciphertext/decrypt arithmetic can be sanity-checked in isolation
# from the actual LWE noise -- with this True, decrypted(r-y) in noise_cal
# mode will exactly equal the true quantized (r-y), sample by sample. Leave
# False for real runs (needed to get a meaningful std(r-y) measurement).
DG4 = DiscreteGaussianCentered(sigma=(0.0 if DEBUG_ZERO_NOISE else lwe_sigma),
                                q_int=lwe_q_int, tau=10.0)

# cs_enc -- state-quantisation scale for the encrypted path. Pick the largest
# cs_enc such that the SINGLE worst (row,state) term still leaves CS_ENC_MARGIN
# of q/2 headroom, using the loosest gain-lattice scale (c_raw_enc=1, plain
# floor/ceil of K_LQR).
c_raw_enc  = C_RAW_ENC
_dx_bound4 = np.deg2rad(DX_BOUND_DEG)
_margin4   = CS_ENC_MARGIN   # fraction of q/2 headroom to spend, worst-case single term
_worst_term4 = float(np.max(np.abs(lqr_K) * _dx_bound4[None, :]) * c_raw_enc)
_cs_enc_max  = _margin4 * (lwe_q_int / 2) / _worst_term4
cs_enc = int(CS_ENC_FRACTION * _cs_enc_max)   # -> CS_ENC_FRACTION * _margin4 fraction of q/2 actually used

K_lat_corners_p4, K_lat_weights_p4 = lattice_weights(lqr_K[0, :], c_raw_enc)   # V_pitch row
K_lat_corners_y4, K_lat_weights_y4 = lattice_weights(lqr_K[1, :], c_raw_enc)   # V_yaw row
uss_lat_corners4, uss_lat_weights4 = lattice_weights(lqr_u_ss, c_raw_enc)      # [u_ss_pitch, u_ss_yaw]

_noise_row_p4 = lwe_sigma * np.abs(lqr_K[0, :]).sum() / cs_enc
_noise_row_y4 = lwe_sigma * np.abs(lqr_K[1, :]).sum() / cs_enc
print(f"\n  LWE: n={lwe_n}  q={lwe_q_int}  sigma={lwe_sigma}  cs_enc={cs_enc}  c_raw_enc={c_raw_enc}")
print(f"  Worst-case single-term headroom used ~= {100*cs_enc/_cs_enc_max*_margin4:.1f}% of q/2")
print(f"  Worst-case summed noise (4 terms, linear bound): V_pitch~={_noise_row_p4:.3f} V   "
      f"V_yaw~={_noise_row_y4:.3f} V   (V_max={V_max} V)")

# Shadow observer's static FIR rows, lattice-decomposed under the SAME
# cs_enc/c_raw_enc convention as the controller above -- lets the observer's
# ct_lincomb reuse the controller's own RESIDUAL ciphertexts (dy=y-y_ss, not
# raw y) directly, no extra encryption calls.
Wobs_lat_corners4, Wobs_lat_weights4 = [], []
for _i in range(4):
    _c4, _w4 = lattice_weights(Wobs_rows[_i], c_raw_enc)
    Wobs_lat_corners4.append(_c4)
    Wobs_lat_weights4.append(_w4)

_worst_term_obs4 = float(max(np.max(np.abs(_r) * np.tile(_dx_bound4[:2], DEADBEAT_LEN)) for _r in Wobs_rows) * c_raw_enc)
_noise_obs4 = [lwe_sigma * np.abs(_r).sum() / cs_enc for _r in Wobs_rows]
print(f"  Observer FIR headroom used ~= {100*cs_enc*_worst_term_obs4/(lwe_q_int/2):.1f}% of q/2 "
      f"(reusing controller's cs_enc)")
print(f"  Observer per-state noise (rad or rad/s): "
      f"pitch~={_noise_obs4[0]:.4f}  yaw~={_noise_obs4[1]:.4f}  "
      f"pitch_rate~={_noise_obs4[2]:.4f}  yaw_rate~={_noise_obs4[3]:.4f}")

# ── Integral-state (Ki) lattice decomposition, same convention as K ────────
# z MUST track the integral of the RESIDUAL dx[:2] = [pitch-pitch_ss,
# yaw-yaw_ss], not raw [pitch,yaw] -- design_lqr's A_aug integrates whatever
# signal the caller feeds it (z[k+1]=z[k]+T*(.)), and never sees x_ss itself,
# so Ki is only meaningful (won't fight a nonzero yaw target, e.g. 30deg) if
# the runtime accumulates z from the SAME dx residual the K term already
# uses -- see ct_dx4[0:2] / (xq4-x_ss_q4)[0:2] below. Same "decrypted only
# once, right before use" pattern as the K term: z is a local plaintext
# accumulator (no different from p4/y4 already being local plaintext sensor
# reads), freshly re-encrypted every step, combined against Ki via
# ct_lincomb, and only THAT scalar decrypted -- no unbounded ciphertext-noise
# growth from repeated homomorphic addition.
_z_bound4 = np.deg2rad(Z_BOUND_DEG_S)
_worst_term_z4 = float(np.max(np.abs(lqr_Ki) * _z_bound4[None, :]) * c_raw_enc)
_cs_z_max = _margin4 * (lwe_q_int / 2) / _worst_term_z4
cs_z = int(CS_ENC_FRACTION * _cs_z_max)

Ki_lat_corners_p4, Ki_lat_weights_p4 = lattice_weights(lqr_Ki[0, :], c_raw_enc)   # V_pitch row, [Ki_pitch, Ki_yaw]
Ki_lat_corners_y4, Ki_lat_weights_y4 = lattice_weights(lqr_Ki[1, :], c_raw_enc)   # V_yaw row

_noise_rowKi_p4 = lwe_sigma * np.abs(lqr_Ki[0, :]).sum() / cs_z
_noise_rowKi_y4 = lwe_sigma * np.abs(lqr_Ki[1, :]).sum() / cs_z
print(f"\n  Ki (integral): cs_z={cs_z}  z_bound(deg*s)={Z_BOUND_DEG_S.tolist()}  Q_I_DIAG={Q_I_DIAG}")
print(f"  Worst-case summed integral-path noise: V_pitch~={_noise_rowKi_p4:.3f} V   "
      f"V_yaw~={_noise_rowKi_y4:.3f} V   (V_max={V_max} V)")


if MODE in ("noise_cal", "both"):
    # ════════════════════════════════════════════════════════════════════════
    # NOISE CALIBRATION RUN (A.s=0): drives the plant with the EXACT same
    # dithered lattice controller as control_detect mode, but with the LWE
    # secret forced to the zero vector. ct_add/ct_sub/ct_smul/ct_lincomb
    # never special-case s, so the ciphertext arithmetic (and its noise
    # growth) is identical to a real secret; the only effect of s=0 is that
    # a.s=0 always, so decryption reduces to mu_hat = b (mod q, centered) =
    # e + mu directly. This isolates and measures the REAL discrete-Gaussian
    # noise landing in the r-y (yaw) residual under actual operating
    # conditions (real dithering, real plant motion).
    #
    # Runs for t_settle_noise seconds first (controller active, plotted, but
    # NOT logged) to let r-y settle to steady state, then logs noise for
    # t_noise_test more seconds.
    # ════════════════════════════════════════════════════════════════════════
    s4_zero = np.zeros(lwe_n, dtype=np.int64)   # A.s=0 for every ciphertext under this secret

    # r is re-encrypted FRESH every step (see loop below), same as y -- so
    # each ct_dx0 gets two independent noise draws, exactly like the real
    # signal path.
    x_ss_q4_0 = np.round(cs_enc * lqr_x_ss).astype(np.int64)

    z_state0 = np.zeros(2)   # LQI integral state [z_pitch, z_yaw], rad*s, plaintext-local (see Ki note above)

    t_settle_noise = T_SETTLE_NOISE_S   # s — run (and plot) the controller this long before logging any
                                         #    noise samples, so r-y has settled to steady state first
    t_noise_test   = T_NOISE_TEST_S     # s — measurement window length, AFTER the settle period
    t_noise_total  = t_settle_noise + t_noise_test

    for _ax in scope.axes:
        _ax.timeWindow += t_noise_total

    ry_log = []   # decrypted (r-y), yaw only, per step (integer, cs_enc counts) -- the raw residual
                  # signal itself (tracking error + dither + LWE noise, all of it), not a diff
                  # against anything. Only appended to once r-y has reached steady state.
    _noise_logging_started = False   # flips True + prints once, when settle period elapses
    _rng_noise = np.random.default_rng(RNG_SEED_NOISE_CAL)
    _next_print_noise = 0.0   # throttled diagnostic prints of raw r, y, and r-y -- see loop below

    compute_dt_log = []   # actual per-step COMPUTE time (before sleep), seconds -- crypto/CPU cost only
    period_dt_log  = []   # actual per-step TOTAL period (compute + sleep + jitter), seconds -- what the plant sees
    _t_loop_prev = None

    myAero2.write_led(color_run)
    startTime_noise = time.perf_counter()
    timestamp_noise = 0.0

    try:
        while timestamp_noise < t_noise_total:
            t_loop = time.perf_counter()
            if _t_loop_prev is not None:
                period_dt_log.append(t_loop - _t_loop_prev)
            _t_loop_prev = t_loop

            myAero2.read_analog_encoder_other_channels()
            y0  = myAero2.yawAngle
            yr0 = myAero2.yawRate
            p0  = myAero2.pitchAngle
            pr0 = myAero2.pitchRate

            xq0 = dither_round(cs_enc * np.array([p0, y0, pr0, yr0]), _rng_noise)

            ct_x0    = [lwe_encrypt_scalar(int(v), s4_zero, lwe_q_int, DG4) for v in xq0]
            ct_xss0  = [lwe_encrypt_scalar(int(v), s4_zero, lwe_q_int, DG4) for v in x_ss_q4_0]   # fresh draw every step, like ct_x0
            ct_dx0   = [ct_sub(ct_x0[j], ct_xss0[j], lwe_q_int) for j in range(4)]

            # LQI integral state: z += T*(dx[:2]) = T*([pitch,yaw] - x_ss[:2]),
            # i.e. integral of the RESIDUAL (not raw angle) -- see Ki note
            # above. Clamped to +/-Z_BOUND_DEG_S as anti-windup.
            z_state0 = z_state0 + T_samp * (np.array([p0, y0]) - lqr_x_ss[:2])
            # Anti-windup clamp removed by request -- z_state0 now accumulates
            # unbounded. NOTE: cs_z (the integral term's LWE quantization
            # scale) was still sized assuming |z| <= Z_BOUND_DEG_S (see
            # _worst_term_z4/_cs_z_max above). If z genuinely grows past that
            # design bound (e.g. a sustained large error or actuator
            # saturation), the quantized/encrypted integral term can exceed
            # the headroom it was sized for and wrap around mod q -- a silent
            # plaintext-corruption failure mode, not just extra noise. Keep
            # an eye on z_state0/z_state4 in the logs if you see anything
            # that looks like a sign flip or a sudden huge voltage command.
            zq0      = dither_round(cs_z * z_state0, _rng_noise)
            ct_z0    = [lwe_encrypt_scalar(int(v), s4_zero, lwe_q_int, DG4) for v in zq0]

            # Same dithered corner-selection law as the real controller.
            idx_p0   = _rng_noise.choice(len(K_lat_weights_p4), p=K_lat_weights_p4)
            idx_y0   = _rng_noise.choice(len(K_lat_weights_y4), p=K_lat_weights_y4)
            idx_uss0 = _rng_noise.choice(len(uss_lat_weights4), p=uss_lat_weights4)
            idx_ip0  = _rng_noise.choice(len(Ki_lat_weights_p4), p=Ki_lat_weights_p4)
            idx_iy0  = _rng_noise.choice(len(Ki_lat_weights_y4), p=Ki_lat_weights_y4)

            ct_u_p0  = ct_lincomb(K_lat_corners_p4[idx_p0].tolist(), ct_dx0, lwe_q_int)
            ct_u_y0  = ct_lincomb(K_lat_corners_y4[idx_y0].tolist(), ct_dx0, lwe_q_int)
            u_p_raw0 = float(int(lwe_decrypt_scalar(ct_u_p0, s4_zero, lwe_q_int)))
            u_y_raw0 = float(int(lwe_decrypt_scalar(ct_u_y0, s4_zero, lwe_q_int)))

            ct_ui_p0  = ct_lincomb(Ki_lat_corners_p4[idx_ip0].tolist(), ct_z0, lwe_q_int)
            ct_ui_y0  = ct_lincomb(Ki_lat_corners_y4[idx_iy0].tolist(), ct_z0, lwe_q_int)
            ui_p_raw0 = float(int(lwe_decrypt_scalar(ct_ui_p0, s4_zero, lwe_q_int)))
            ui_y_raw0 = float(int(lwe_decrypt_scalar(ct_ui_y0, s4_zero, lwe_q_int)))

            uss_comb0 = uss_lat_corners4[idx_uss0].astype(float)
            _term_uss0 = uss_comb0 / c_raw_enc
            _term_K0   = -np.array([u_p_raw0, u_y_raw0]) / (c_raw_enc * cs_enc)
            _term_Ki0  = -np.array([ui_p_raw0, ui_y_raw0]) / (c_raw_enc * cs_z)
            u_raw0 = _term_uss0 + _term_K0 + _term_Ki0
            u_cmd0 = np.clip(u_raw0, -V_max, V_max)

            myAero2.write_voltage(float(u_cmd0[0]), float(u_cmd0[1]))

            # ── r-y = -dx0[1]. Decrypt it -- with s4_zero this is just b (mod
            #    q, centered) = e+mu directly, i.e. exactly the residual
            #    signal the real (secret-keyed) controller/detector would
            #    also decrypt to. ──
            ctres0  = ct_neg(ct_dx0[1], lwe_q_int)
            dec_ry0 = int(lwe_decrypt_scalar(ctres0, s4_zero, lwe_q_int))

            if timestamp_noise >= _next_print_noise:
                _last_period_ms = period_dt_log[-1] * 1000 if period_dt_log else float("nan")
                _z_deg0 = np.rad2deg(z_state0)
                _sat0 = u_raw0 != u_cmd0   # True per-axis if V_max clipping is actually biting
                print(f"  [t={timestamp_noise:6.2f}s]  yaw_r={np.rad2deg(lqr_x_ss[1]):+7.3f}deg  "
                      f"yaw_y={np.rad2deg(y0):+7.3f}deg  (r-y)_decrypted={dec_ry0:+7d}  "
                      f"z_state(deg*s): pitch={_z_deg0[0]:+8.3f}(bound={Z_BOUND_DEG_S[0]:.1f}) "
                      f"yaw={_z_deg0[1]:+8.3f}(bound={Z_BOUND_DEG_S[1]:.1f})  "
                      f"u_raw(V): pitch={u_raw0[0]:+7.3f} yaw={u_raw0[1]:+7.3f}  "
                      f"[yaw terms(V): uss={_term_uss0[1]:+7.3f}  K*dx={_term_K0[1]:+7.3f}  "
                      f"Ki*z={_term_Ki0[1]:+8.3f}]  "
                      f"u_applied(V): pitch={u_cmd0[0]:+6.3f}{'*' if _sat0[0] else ' '} "
                      f"yaw={u_cmd0[1]:+6.3f}{'*' if _sat0[1] else ' '}  "
                      f"period={_last_period_ms:6.1f}ms (target={T_samp*1000:.1f}ms)")
                _next_print_noise += PRINT_INTERVAL_S

            if timestamp_noise >= t_settle_noise:   # only log once r-y has settled
                if not _noise_logging_started:
                    _noise_logging_started = True
                    print(f"[noise-cal] settle period done ({t_settle_noise:.1f}s) — "
                          f"logging r-y now for {t_noise_test:.1f}s", flush=True)
                ry_log.append(dec_ry0)

            _tg = _gt()
            scope.axes[0].sample(_tg, [TARGET_REF_DEG, np.rad2deg(y0)])
            MultiScope.refreshAll()

            elapsed = time.perf_counter() - t_loop
            compute_dt_log.append(elapsed)
            if T_samp - elapsed > 0:
                time.sleep(T_samp - elapsed)
            timestamp_noise = time.perf_counter() - startTime_noise
    except KeyboardInterrupt:
        print("\nNoise-calibration run interrupted — analysing what was collected.")
    finally:
        myAero2.write_voltage(0.0, 0.0)
        myAero2.write_led(color_done)

    compute_dt_arr = np.array(compute_dt_log, dtype=float)
    period_dt_arr  = np.array(period_dt_log, dtype=float)
    _n_overrun = int(np.sum(compute_dt_arr > T_samp))
    print("═══════════════════════════════════════════════════════════════")
    print(" LOOP TIMING — is the loop rate actually steady?")
    print("═══════════════════════════════════════════════════════════════")
    print(f"  target period = {T_samp*1000:.1f}ms ({frequency}Hz)")
    print(f"  compute time  : mean={compute_dt_arr.mean()*1000:.1f}ms  std={compute_dt_arr.std()*1000:.1f}ms  "
          f"max={compute_dt_arr.max()*1000:.1f}ms  (overruns budget in {_n_overrun}/{len(compute_dt_arr)} steps)")
    print(f"  actual period : mean={period_dt_arr.mean()*1000:.1f}ms  std={period_dt_arr.std()*1000:.1f}ms  "
          f"max={period_dt_arr.max()*1000:.1f}ms")
    print("═══════════════════════════════════════════════════════════════\n")

    ry_arr = np.array(ry_log, dtype=float)
    print("═══════════════════════════════════════════════════════════════")
    print(" NOISE CALIBRATION (A.s=0) — measured r-y (yaw) residual, std(r-y)")
    print("═══════════════════════════════════════════════════════════════")
    print(f"  samples = {len(ry_arr)}  (~{t_noise_test:.1f}s @ {frequency}Hz, "
          f"after {t_settle_noise:.1f}s settle)")
    print(f"  mean (counts, cs_enc={cs_enc} scale) = {ry_arr.mean():+.3f}")
    print(f"  std  (counts)                        = {ry_arr.std():.3f}")
    print(f"  configured lwe_sigma                 = {lwe_sigma}")
    print(f"  theoretical sigma_eff (sqrt(2)*sigma) = {np.sqrt(2) * lwe_sigma:.3f}  "
          f"(LWE-only floor -- std(r-y) above also includes real tracking error + dither)")
    print(f"  std in physical units                = {ry_arr.std() / cs_enc:.6f} rad "
          f"({np.rad2deg(ry_arr.std() / cs_enc):.4f} deg)")
    print("═══════════════════════════════════════════════════════════════\n")


if MODE in ("control_detect", "both"):
    # ════════════════════════════════════════════════════════════════════════
    # REAL LWE-ENCRYPTED CONTROL + LIVE LATTICE ANOMALY DETECTOR.
    #
    # Genuine random secret, real encryption throughout. r's plaintext
    # quantization is fixed, but its CIPHERTEXT is re-encrypted FRESH every
    # step below (like aero2_unified_detect.py's ct_r would need to be, but
    # isn't there) -- so both r and y carry independent per-step noise
    # draws, and r-y genuinely has variance 2*sigma^2 at every sample.
    #
    # The live detector is ported from aero2_unified_detect.py's background
    # BLASTER/LLL worker, unchanged in mechanism, scoped down to just the
    # YAW residual r-y (that script's single controlled axis). Here r-y for
    # yaw is -dx4[1] (dx4[1] = y_yaw - yaw_ss = y - r), so the worker is fed
    # ct_neg(ct_dx4[1]) each step -- no extra encryption, reuses the
    # controller's own residual ciphertext.
    #
    # For a window of DET_WINDOW consecutive ciphertexts (a_k, b_k) under the
    # same secret s4, the worker computes the left nullspace mod q of the
    # stacked a-vectors: any d with d.A=0 (mod q) satisfies d.b_vec =
    # d.(e_vec+mu_vec) -- the secret-dependent term cancels, so a short/
    # reduced d turns d.b_vec into a pure-noise statistic. An anomalously
    # large statistic (relative to the expected LWE noise budget) signals
    # ciphertext tampering that a secret-less attacker cannot forge around
    # on every projection simultaneously. Runs in a background thread so
    # BLASTER's LLL never blocks the control loop; always grabs the LATEST
    # DET_WINDOW-sample window (no backlog), so detection latency never
    # compounds.
    #
    # DET_WINDOW is sized to what plain LLL (not a BKZ-beta attacker) can
    # best exploit -- see aero2_unified_detect.py's own note; this is a demo
    # of the detection MECHANISM, not the crypto-hardness-grade
    # parameterization. The LWE secret s4 is periodically rotated (see
    # KEY_ROTATION_WINDOWS in USER CONFIG) to cap how many ciphertexts ever
    # accumulate under one key.
    # ════════════════════════════════════════════════════════════════════════
    s4 = (np.zeros(lwe_n, dtype=np.int64) if DEBUG_ZERO_KEYS
          else lwe_keygen(lwe_n, lwe_q_int, secret_dist="small"))

    x_ss_q4 = np.round(cs_enc * lqr_x_ss).astype(np.int64)

    z_state4 = np.zeros(2)   # LQI integral state [z_pitch, z_yaw], rad*s, plaintext-local (see Ki note above)

    DET_M_OPT  = int(round(math.sqrt(lwe_n * math.log(lwe_q_int) / math.log(DET_DELTA_LLL))))
    DET_WINDOW = max(DET_M_OPT, lwe_n + 2)   # floor: need free_dim = window - n >= 2

    # ctres4 = ct_neg(ct_dx4[1]) = ct_r[1] - ct_x4[1], noise = e_r - e_y --
    # BOTH r and y are re-encrypted fresh every step (see main loop below),
    # so each is an independent DG(sigma) draw and the residual's noise
    # variance is 2*sigma^2 -- IF the only source of variance were LWE
    # encryption noise. sigma_eff_det is the number the chi^2 test actually
    # calibrates against; see DET_SIGMA_EFF_OVERRIDE in USER CONFIG for the
    # three ways to set it (override / measured-this-run via "both" /
    # theoretical fallback).
    if DET_SIGMA_EFF_OVERRIDE is not None:
        sigma_eff_det = float(DET_SIGMA_EFF_OVERRIDE)
        _sigma_eff_src = "user override (DET_SIGMA_EFF_OVERRIDE)"
    elif MODE == "both" and "ry_arr" in globals() and len(ry_arr) > 0:
        sigma_eff_det = float(ry_arr.std())
        _sigma_eff_src = f"measured this run (noise_cal std(r-y), {len(ry_arr)} samples)"
    else:
        sigma_eff_det = float(np.sqrt(2) * lwe_sigma)
        _sigma_eff_src = ("theoretical sqrt(2)*LWE_SIGMA (no measured noise_cal data -- "
                           "run MODE=\"both\" or set DET_SIGMA_EFF_OVERRIDE for a real number)")
    norm_thrs_det = int(lwe_q_int / (2 * sigma_eff_det * DET_Z_THRESH))
    chi2_thr_det  = float(_chi2.ppf(0.999, df=DET_NUM_DOF))

    KEY_ROTATION_PERIOD = (int(KEY_ROTATION_WINDOWS) * DET_WINDOW) if KEY_ROTATION_WINDOWS else None

    print("═══════════════════════════════════════════════════════════════")
    print(" LIVE ANOMALY DETECTOR (LLL-only, demo-mode, yaw r-y only)")
    print("═══════════════════════════════════════════════════════════════")
    print(f"  DET_DELTA_LLL={DET_DELTA_LLL}  m_opt={DET_M_OPT}  window=DET_WINDOW={DET_WINDOW}  "
          f"free_dim={DET_WINDOW - lwe_n}  DET_NUM_DOF={DET_NUM_DOF}")
    print(f"  starts after {DET_WINDOW} samples (~{DET_WINDOW*T_samp:.1f}s)")
    print(f"  sigma_eff_det={sigma_eff_det:.2f} ({_sigma_eff_src})")
    print(f"  norm_thrs_det={norm_thrs_det}  chi2_thr(0.999,df={DET_NUM_DOF})={chi2_thr_det:.2f}")
    print("  Always analyzes the LATEST available DET_WINDOW-sample window (no backlog).")
    if KEY_ROTATION_PERIOD:
        print(f"  Key rotation: every {KEY_ROTATION_PERIOD} samples "
              f"({KEY_ROTATION_WINDOWS} x DET_WINDOW, ~{KEY_ROTATION_PERIOD*T_samp:.1f}s)")
    else:
        print("  Key rotation: disabled")
    print("═══════════════════════════════════════════════════════════════\n")

    def _blaster_lll_phase1(B_np, norm_thrs, num_dof,
                             max_cycles=DET_MAX_CYCLES, increment=DET_INCREMENT,
                             verbose=DET_VERBOSE, label=""):
        """Adaptive-depth BLASTER LLL loop (Phase 1 of aero2_detect.py, no BKZ handoff).
        Returns (reduced_np, converged: bool)."""
        total_rows  = B_np.shape[0]
        _depth      = 0
        _prev_short = None
        current_np  = B_np.copy()
        _pass, _cycle = 0, 0
        while True:
            if DET_PAUSE_ON_SLOW_LOOP:
                # det_stop is checked alongside det_pause so a shutdown landing
                # mid-pause can't strand this thread here forever.
                while det_pause.is_set() and not det_stop.is_set():
                    time.sleep(DET_PAUSE_POLL_S)
            _, current_np, _ = _blaster_reduce(current_np.T, delta=0.99999, cores=1,
                                                use_seysen=False, depth=_depth)
            current_np = current_np.T
            _norms   = np.sort(np.linalg.norm(current_np.astype(float), axis=1))
            _ndof_nm = float(_norms[num_dof - 1])
            _pass += 1
            if verbose:
                print(f"    [det{label}] LLL pass {_pass}: depth={_depth}  "
                      f"{num_dof}th={_ndof_nm:.3e}  thr={norm_thrs}", flush=True)
            if _ndof_nm <= norm_thrs:
                return current_np, True
            if _prev_short is None or _ndof_nm < _prev_short * DET_IMPROVE_THRESH:
                _prev_short = _ndof_nm
            else:
                _depth = min(_depth + increment, total_rows)
                _prev_short = _ndof_nm
                if _depth == total_rows:
                    _cycle += 1
                    _depth = 0
                    _prev_short = None
                    if _cycle >= max_cycles:
                        return current_np, False

    def detect_window(window, tag="", t_start=None, t_end=None):
        """window: list of DET_WINDOW (a_ints: list[int] len lwe_n, b_int: int) pairs,
        plain Python ints, all encrypted under the same secret. t_start/t_end are
        the scope-timeline (_gt()) bounds this window covers, purely for the
        printout below -- pass None if unknown. Returns a results dict and
        prints a summary block."""
        t0d   = time.perf_counter()
        m_sub = len(window)
        print(f"  [detect{tag}] window received: m={m_sub}  computing nullspace...", flush=True)

        t_ns = time.perf_counter()
        Av_full = [[int(a) % lwe_q_int for a in row[0]] for row in window]
        basis_d_full = left_nullspace_mod_q(Av_full, lwe_q_int)
        basis_d_center_full = recenter_mod_q(
            np.array([[int(x) for x in d] for d in basis_d_full], dtype=np.int64), lwe_q_int)
        free_dim_full = len(basis_d_center_full)
        dim = m_sub
        print(f"  [detect{tag}] nullspace: {free_dim_full}×{dim}  ({time.perf_counter()-t_ns:.2f}s)", flush=True)

        m_null = min(m_sub, free_dim_full)
        m_qI   = m_sub - m_null

        qI_np = np.zeros((m_qI, dim), dtype=np.int64)
        np.fill_diagonal(qI_np, lwe_q_int)
        B_np = np.vstack([qI_np, basis_d_center_full[:m_null].astype(np.int64)])
        print(f"  [detect{tag}] lattice: {B_np.shape[0]}×{dim}  ({m_null} null + {m_qI} q·I)  "
              f"norm_thrs={norm_thrs_det} — starting LLL...", flush=True)

        Bred_np, converged = _blaster_lll_phase1(B_np, norm_thrs_det, DET_NUM_DOF, label=tag)

        norms      = np.linalg.norm(Bred_np.astype(float), axis=1)
        idx        = np.argsort(norms)
        sorted_nms = norms[idx]
        V          = Bred_np[idx[:DET_NUM_DOF]]

        bvr_obj = np.array([int(centered_mod_int(int(row[1]), lwe_q_int)) for row in window], dtype=object)
        y_f = np.array([
            float(int(centered_mod_int(int(np.asarray(V[ii], dtype=object) @ bvr_obj), lwe_q_int)))
            for ii in range(DET_NUM_DOF)
        ])

        V_f  = V.astype(float)
        nrms = np.linalg.norm(V_f, axis=1)
        z_diag     = y_f / (sigma_eff_det * nrms)
        Stats_diag = float(np.sum(z_diag ** 2))

        U_svd, sv, _ = np.linalg.svd(V_f, full_matrices=False)
        mask_sv   = sv > sv[0] / 30.0
        y_rot     = U_svd[:, mask_sv].T @ y_f
        z_svd     = y_rot / (sigma_eff_det * sv[mask_sv])
        Stats_svd = float(np.sum(z_svd ** 2))
        eff_dof   = int(mask_sv.sum())

        thr     = float(_chi2.ppf(0.999, df=eff_dof)) if eff_dof != DET_NUM_DOF else chi2_thr_det
        flagged = Stats_svd > thr
        dt      = time.perf_counter() - t0d

        result = {
            "tag": tag, "converged": converged, "shortest": float(sorted_nms[0]),
            "ndof_norm": float(sorted_nms[DET_NUM_DOF - 1]), "Stats_svd": Stats_svd,
            "Stats_diag": Stats_diag, "eff_dof": eff_dof, "thr": thr,
            "flagged": flagged, "dt": dt,
        }
        if t_start is not None and t_end is not None:
            result["t_start"], result["t_end"] = t_start, t_end
            window_str = f"t=[{t_start:7.2f}, {t_end:7.2f}]s  (span={t_end - t_start:.2f}s)"
        else:
            window_str = "t=[unknown]"
        vlen_reached = converged   # LLL hit norm_thrs_det (target DET_NUM_DOF-th vector length) before giving up
        flag_str = "!!! ANOMALY FLAGGED !!!" if flagged else "nominal"
        bar = "=" * 68
        print(
            f"\n  [detect{tag}] {bar}\n"
            f"  [detect{tag}] window          {window_str}\n"
            f"  [detect{tag}] >>> SVD_STATS = {Stats_svd:10.3f}  <<<   threshold={thr:8.3f}  "
            f"(eff_dof={eff_dof}/{DET_NUM_DOF})\n"
            f"  [detect{tag}] vector length reached (norm<=thr): "
            f"{'YES' if vlen_reached else 'NO -- gave up, best-effort reduction used'}  "
            f"[{DET_NUM_DOF}th-norm={sorted_nms[DET_NUM_DOF-1]:.3e}  target<={norm_thrs_det:.3e}]\n"
            f"  [detect{tag}] anomaly:        {flag_str}\n"
            f"  [detect{tag}] {bar}  ({dt:.2f}s)\n",
            flush=True,
        )
        return result

    det_window_deque = deque(maxlen=DET_WINDOW)   # holds the MOST RECENT DET_WINDOW yaw-residual ciphertexts
    det_time_deque   = deque(maxlen=DET_WINDOW)   # parallel deque of scope-timeline (_gt()) timestamps,
                                                    # kept in lockstep with det_window_deque

    det_full_log      = []   # (t, key_epoch, a, b) for EVERY detector-feed sample, for post-run gap backfill
    det_results       = []
    det_results_lock  = threading.Lock()
    det_window_count  = 0
    det_new_data      = threading.Event()   # set whenever det_window_deque reaches length DET_WINDOW
    det_stop          = threading.Event()   # set to tell the worker to exit after its current pass
    det_pause         = threading.Event()   # set by the main loop to tell the worker to hold off starting
                                              # its next LLL pass (see DET_PAUSE_* config above)
    det_marker_queue  = queue.Queue()   # (t_start, t_end, flagged, tag) tuples -- worker thread PRODUCES,
                                          # main loop CONSUMES and does the actual pyqtgraph/Qt drawing

    # Install the is_weakly_lll_reduced monkeypatch now that det_pause exists
    # (see the long comment up by the imports for why this is needed at all).
    # Every internal iteration of BLASTER's own lll_reduce() while-loop ends
    # by calling this to decide whether to keep going -- we piggyback the
    # det_pause wait on that call so a pause can take effect *inside* a
    # single _blaster_reduce() call, not just between our own outer passes.
    _orig_is_weakly_lll_reduced = _blaster_internals.is_weakly_lll_reduced

    def _paused_is_weakly_lll_reduced(R, delta):
        # det_stop is checked alongside det_pause so a shutdown landing mid-
        # pause can't strand this thread here forever -- see the matching
        # comment in _blaster_lll_phase1's pause check for why that matters.
        if DET_PAUSE_ON_SLOW_LOOP:
            while det_pause.is_set() and not det_stop.is_set():
                time.sleep(DET_PAUSE_POLL_S)
        return _orig_is_weakly_lll_reduced(R, delta)

    _blaster_internals.is_weakly_lll_reduced = _paused_is_weakly_lll_reduced

    def _detector_worker():
        global det_window_count
        # Lower this thread's OS scheduling priority (raise its niceness) so
        # that whenever the GIL actually does become free to hand off (e.g.
        # between the QR / native-LLL / reduce / matmul sub-steps inside a
        # single _blaster_reduce() call, or at any of the det_pause
        # checkpoints above), the OS scheduler is biased toward waking the
        # real-time control loop's thread first. Raising your OWN niceness
        # needs no special privileges, so this is safe to leave on always.
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
        except Exception as _ex:
            print(f"[detect] note: couldn't lower detector thread priority "
                  f"({type(_ex).__name__}: {_ex}) -- continuing anyway", flush=True)
        while True:
            det_new_data.wait(timeout=0.5)
            if len(det_window_deque) < DET_WINDOW:
                if det_stop.is_set():
                    break
                continue
            det_new_data.clear()
            window   = list(det_window_deque)
            t_window = list(det_time_deque)
            det_window_count += 1
            print(f"\n[detect] worker picked up window #{det_window_count} "
                  f"(t={time.perf_counter()-startTime4:.1f}s)", flush=True)
            try:
                if len(t_window) >= len(window):
                    t_start, t_end = t_window[0], t_window[-1]
                else:
                    t_start, t_end = None, None
                res = detect_window(window, tag=f"#{det_window_count}", t_start=t_start, t_end=t_end)
                if t_start is not None:
                    det_marker_queue.put((t_start, t_end, res["flagged"], f"#{det_window_count}"))
                with det_results_lock:
                    det_results.append(res)
            except Exception as ex:
                print(f"  [detect] window {det_window_count} FAILED: {type(ex).__name__}: {ex}", flush=True)
            if det_stop.is_set():
                break

    det_thread = threading.Thread(target=_detector_worker, daemon=True)
    det_thread.start()

    t_lat4_test = T_CONTROL_RUN_S    # s — total run length

    for _ax in scope.axes:
        _ax.timeWindow += t_lat4_test

    cty_log, ctr_log = [], []   # ciphertext-of-measurement / ciphertext-of-residual logs, per step (see header)

    u_lat4_log, y_lat4_log, yr_lat4_log, t_lat4_log = [], [], [], []
    pitch_lat4_log, pitchrate_lat4_log = [], []
    pitch_hat_lat4_log, yaw_hat_lat4_log = [], []
    pitchrate_hat_lat4_log, yawrate_hat_lat4_log = [], []

    _lat4_rng = np.random.default_rng(RNG_SEED_CONTROL)   # fixed seed -> reproducible dithering

    _next_print4 = 0.0   # throttled diagnostic prints of the LQI integral state z (see below)

    rekey_count = 0
    _steps_since_rekey = 0
    _det_feed_count = 0   # mirrors len(det_window_deque), kept for the progress prints below

    # Seed the observer's ENCRYPTED dy=y-y_ss history window with the first
    # real [pitch,yaw] sample, repeated. Built the same way ct_dx4 is built
    # at runtime (encrypt r and y fresh, then ct_sub) so it's a genuine
    # residual ciphertext, with independent noise on each side, from the
    # start.
    myAero2.read_analog_encoder_other_channels()
    _y0_4  = np.array([myAero2.pitchAngle, myAero2.yawAngle])
    _yq0_4 = np.round(cs_enc * _y0_4).astype(np.int64)
    _ct_y0_4   = [lwe_encrypt_scalar(int(v), s4, lwe_q_int, DG4) for v in _yq0_4]
    _ct_xss0_4 = [lwe_encrypt_scalar(int(v), s4, lwe_q_int, DG4) for v in x_ss_q4[:2]]
    _ct_dy0_4  = [ct_sub(_ct_y0_4[j], _ct_xss0_4[j], lwe_q_int) for j in range(2)]
    dy_hist4_ct = [list(_ct_dy0_4) for _ in range(DEADBEAT_LEN)]   # [ct_dy[k-1], ..., ct_dy[k-DEADBEAT_LEN]], each a [ct_pitch,ct_yaw] pair

    myAero2.write_led(color_run)
    startTime4 = time.perf_counter()
    timestamp4 = 0.0
    _attack_was_active = False   # edge-detects attack on/off for the console prints + one-time scope shading
    _last_slow_t = time.perf_counter()   # timestamp of the most recent slow (GIL-contention) iteration;
                                          # det_pause is only CLEARed once we've been on-schedule for
                                          # DET_PAUSE_RECOVER_S straight since this

    try:
        while timestamp4 < t_lat4_test:
            t_loop = time.perf_counter()
            _tg = _gt()   # single per-iteration scope timestamp (also tags detector windows, see below)

            myAero2.read_analog_encoder_other_channels()
            y4  = myAero2.yawAngle
            yr4 = myAero2.yawRate
            p4  = myAero2.pitchAngle

            # ── Sensor attack: spoof yawAngle with a constant bias while
            #    active (see ATTACK_* / DEBUG_ZERO_ATTACK config above). This
            #    happens BEFORE quantization/encryption, so the corrupted
            #    value flows through the real controller (physical
            #    mis-tracking) and the real encrypted detector feed alike --
            #    exactly what a real FDI sensor attacker would produce. ──
            _attack_active = (not DEBUG_ZERO_ATTACK) and \
                              (ATTACK_TIME_START <= _tg < ATTACK_TIME_START + ATTACK_DURATION)
            if _attack_active:
                y4 = y4 + np.deg2rad(ATTACK_BIAS_DEG)
                if not _attack_was_active:
                    print(f"\n[ATTACK] sensor spoof ACTIVE at t={_tg:.1f}s -- injecting "
                          f"{ATTACK_BIAS_DEG:+.2f} deg bias into yawAngle for "
                          f"{ATTACK_DURATION:.1f}s", flush=True)
                    if DET_MARK_ENABLE:
                        _atk_plot = scope.axes[0].plot
                        _atk_region = pg.LinearRegionItem(
                            values=(_tg, _tg + ATTACK_DURATION),
                            movable=False,
                            brush=pg.mkBrush(255, 140, 0, 60),
                            pen=pg.mkPen(color=(200, 100, 0), width=2),
                        )
                        _atk_region.setZValue(-5)   # above detection-window shading, below signal traces
                        _atk_plot.addItem(_atk_region)
                        _det_mark_items.append(_atk_region)
            elif _attack_was_active:
                print(f"[ATTACK] sensor spoof ENDED at t={_tg:.1f}s", flush=True)
            _attack_was_active = _attack_active
            pr4 = myAero2.pitchRate

            xq4 = dither_round(cs_enc * np.array([p4, y4, pr4, yr4]), _lat4_rng)   # [pitch,yaw,pitch_rate,yaw_rate], dithered quantization

            # Encrypt the live measurement AND re-encrypt the equilibrium
            # (both fresh, independent noise draws each step -- x_ss4's
            # PLAINTEXT never changes, only its ciphertext), then form the
            # residual dx = x - x_ss HOMOMORPHICALLY (ciphertext subtraction)
            # -- the plaintext delta is never computed in the clear.
            ct_x4    = [lwe_encrypt_scalar(int(v), s4, lwe_q_int, DG4) for v in xq4]
            ct_xss4  = [lwe_encrypt_scalar(int(v), s4, lwe_q_int, DG4) for v in x_ss_q4]
            ct_dx4   = [ct_sub(ct_x4[j], ct_xss4[j], lwe_q_int) for j in range(4)]
            cty_log.append(ct_x4)
            ctr_log.append(ct_dx4)

            # ── Live detector feed: r - y, yaw only. dx4[1] = y_yaw - yaw_ss
            #    = y - r, so r - y = ct_neg(ct_dx4[1]) -- no extra
            #    encryption, just reuses the controller's own residual
            #    ciphertext. ──
            ctres4 = ct_neg(ct_dx4[1], lwe_q_int)
            _ctres4_a = [int(a) for a in ctres4[0]]
            _ctres4_b = int(ctres4[1])
            det_window_deque.append((_ctres4_a, _ctres4_b))
            det_time_deque.append(_tg)
            det_full_log.append((_tg, rekey_count, _ctres4_a, _ctres4_b))
            if len(det_window_deque) >= DET_WINDOW:
                det_new_data.set()
            _det_feed_count = len(det_window_deque)
            _r4 = float(lqr_x_ss[1])   # yaw target (rad), same value ct_xss4[1] encrypts
            # print(f"[detect-feed t={_tg:7.2f}s] appended b={_ctres4_b}  a[:4]={_ctres4_a[:4]}...  |  "
            #       f"y={np.rad2deg(y4):+7.3f}deg  r={np.rad2deg(_r4):+7.3f}deg  "
            #       f"(r-y)={np.rad2deg(_r4 - y4):+7.3f}deg  window={_det_feed_count}/{DET_WINDOW}",
            #       flush=True)
            if _det_feed_count % 50 == 0 and _det_feed_count < DET_WINDOW:   # periodic progress while filling the first window
                print(f"[detect] collecting initial window: {_det_feed_count}/{DET_WINDOW} samples",
                      flush=True)

            # LQI integral state: z += T*(dx[:2]) = T*([pitch,yaw] - x_ss[:2]),
            # i.e. integral of the RESIDUAL (not raw angle) -- see Ki note
            # above. Clamped to +/-Z_BOUND_DEG_S as anti-windup. Same
            # fresh-encrypt-every-step treatment as x_ss4.
            z_state4 = z_state4 + T_samp * (np.array([p4, y4]) - lqr_x_ss[:2])
            # Anti-windup clamp removed by request -- z_state4 now accumulates
            # unbounded (see the matching noise_cal-mode comment above z_state0
            # for the cs_z/headroom caveat -- same risk applies here).
            zq4      = dither_round(cs_z * z_state4, _lat4_rng)
            ct_z4    = [lwe_encrypt_scalar(int(v), s4, lwe_q_int, DG4) for v in zq4]

            # Dithering (as noise_cal mode): draw ONE corner per row/u_ss, P(corner i)=w_i.
            idx_p4   = _lat4_rng.choice(len(K_lat_weights_p4), p=K_lat_weights_p4)
            idx_y4   = _lat4_rng.choice(len(K_lat_weights_y4), p=K_lat_weights_y4)
            idx_uss4 = _lat4_rng.choice(len(uss_lat_weights4), p=uss_lat_weights4)
            idx_ip4  = _lat4_rng.choice(len(Ki_lat_weights_p4), p=Ki_lat_weights_p4)
            idx_iy4  = _lat4_rng.choice(len(Ki_lat_weights_y4), p=Ki_lat_weights_y4)

            # Controller operates on the ENCRYPTED residual: ct_u_i = Sum_j K_i[j]*ct_dx[j],
            # a ciphertext-space linear combination -- decrypted only once, right
            # here, immediately before it's needed to drive the plant. Same
            # pattern for the integral term: ct_ui_i = Sum_j Ki_i[j]*ct_z[j].
            ct_u_p4  = ct_lincomb(K_lat_corners_p4[idx_p4].tolist(), ct_dx4, lwe_q_int)
            ct_u_y4  = ct_lincomb(K_lat_corners_y4[idx_y4].tolist(), ct_dx4, lwe_q_int)
            u_p_raw4 = float(int(lwe_decrypt_scalar(ct_u_p4, s4, lwe_q_int)))   # scale c_raw_enc*cs_enc
            u_y_raw4 = float(int(lwe_decrypt_scalar(ct_u_y4, s4, lwe_q_int)))

            ct_ui_p4  = ct_lincomb(Ki_lat_corners_p4[idx_ip4].tolist(), ct_z4, lwe_q_int)
            ct_ui_y4  = ct_lincomb(Ki_lat_corners_y4[idx_iy4].tolist(), ct_z4, lwe_q_int)
            ui_p_raw4 = float(int(lwe_decrypt_scalar(ct_ui_p4, s4, lwe_q_int)))   # scale c_raw_enc*cs_z
            ui_y_raw4 = float(int(lwe_decrypt_scalar(ct_ui_y4, s4, lwe_q_int)))

            uss_comb4 = uss_lat_corners4[idx_uss4].astype(float)   # plaintext constant, scale c_raw_enc, len-2

            _term_uss4 = uss_comb4 / c_raw_enc                                              # steady-state feedforward, volts
            _term_K4   = -np.array([u_p_raw4, u_y_raw4]) / (c_raw_enc * cs_enc)             # proportional (K*dx), volts
            _term_Ki4  = -np.array([ui_p_raw4, ui_y_raw4]) / (c_raw_enc * cs_z)             # integral (Ki*z), volts
            u_raw4 = _term_uss4 + _term_K4 + _term_Ki4   # descaled, volts
            u_cmd4 = np.clip(u_raw4, -V_max, V_max)   # [V_pitch, V_yaw], each its own channel limit

            myAero2.write_voltage(float(u_cmd4[0]), float(u_cmd4[1]))

            if _tg >= _next_print4:
                _z_deg4 = np.rad2deg(z_state4)
                _sat4 = u_raw4 != u_cmd4   # True per-axis if V_max clipping is actually biting
                print(f"[z t={_tg:7.2f}s] "
                      f"yaw: r={np.rad2deg(lqr_x_ss[1]):+7.3f}deg  y={np.rad2deg(y4):+7.3f}deg  "
                      f"(r-y)={np.rad2deg(lqr_x_ss[1] - y4):+7.3f}deg   "
                      f"z_state(deg*s): "
                      f"pitch={_z_deg4[0]:+9.3f}(bound={Z_BOUND_DEG_S[0]:.1f})  "
                      f"yaw={_z_deg4[1]:+9.3f}(bound={Z_BOUND_DEG_S[1]:.1f})   "
                      f"u_raw(V): pitch={u_raw4[0]:+7.3f}  yaw={u_raw4[1]:+7.3f}   "
                      f"[yaw terms(V): uss={_term_uss4[1]:+7.3f}  K*dx={_term_K4[1]:+7.3f}  "
                      f"Ki*z={_term_Ki4[1]:+8.3f}]   "
                      f"u_applied(V): pitch={u_cmd4[0]:+6.3f}{'*' if _sat4[0] else ' '} "
                      f"yaw={u_cmd4[1]:+6.3f}{'*' if _sat4[1] else ' '}  "
                      f"(*=V_max clip, V_max={V_max})", flush=True)
                _next_print4 += PRINT_INTERVAL_S

            # Static deadbeat/autonomous shadow observer -- ENCRYPTED,
            # ciphertext-space. Reuses ct_dx4[0]/ct_dx4[1] (the [pitch,yaw]
            # RESIDUAL ciphertexts, dy=y-y_ss, already computed above for the
            # controller) as the newest entry of a DEADBEAT_LEN-deep
            # ciphertext window; the dithered-selected integer corner of row
            # [W1|...|W_{DEADBEAT_LEN}] is applied to that window via
            # ct_lincomb, decrypted, descaled, and offset by the known
            # constant c_bar_obs[row].
            ct_dy8_4 = sum(dy_hist4_ct, [])   # [ct_dy[k-1]|...|ct_dy[k-DEADBEAT_LEN]]
            x_hat = np.empty(4)
            for _i in range(4):
                idx_obs4 = _lat4_rng.choice(len(Wobs_lat_weights4[_i]), p=Wobs_lat_weights4[_i])
                ct_xhat_i = ct_lincomb(Wobs_lat_corners4[_i][idx_obs4].tolist(), ct_dy8_4, lwe_q_int)
                x_hat[_i] = float(int(lwe_decrypt_scalar(ct_xhat_i, s4, lwe_q_int))) / (c_raw_enc * cs_enc) + c_bar_obs[_i]
            dy_hist4_ct = [ct_dx4[0:2]] + dy_hist4_ct[:-1]   # push newest [ct_pitch,ct_yaw] residual, drop oldest

            # ── LWE key rotation: after KEY_ROTATION_PERIOD samples, replace
            #    s4 with a fresh secret. Any ciphertext still alive under the
            #    old key (the detector's in-flight window, the observer's
            #    persisted dy history) is invalid under the new key, so both
            #    are dropped/re-seeded here -- the detector just starts
            #    collecting a fresh window, and the observer takes another
            #    DEADBEAT_LEN-step warm-up, exactly like at startup. ──
            _steps_since_rekey += 1
            if KEY_ROTATION_PERIOD and _steps_since_rekey >= KEY_ROTATION_PERIOD:
                rekey_count += 1
                print(f"\n[rekey] rotating LWE secret (#{rekey_count}) after {_steps_since_rekey} samples "
                      f"(~{KEY_ROTATION_WINDOWS} x DET_WINDOW={DET_WINDOW}) -- resetting detector window "
                      f"and observer history...", flush=True)
                s4 = (np.zeros(lwe_n, dtype=np.int64) if DEBUG_ZERO_KEYS
                      else lwe_keygen(lwe_n, lwe_q_int, secret_dist="small"))
                det_window_deque.clear()   # drop in-flight window (ciphertexts under the old key)
                det_time_deque.clear()
                _det_feed_count = 0
                _ct_y0_rk   = [lwe_encrypt_scalar(int(v), s4, lwe_q_int, DG4)
                               for v in np.round(cs_enc * np.array([p4, y4])).astype(np.int64)]
                _ct_xss0_rk = [lwe_encrypt_scalar(int(v), s4, lwe_q_int, DG4) for v in x_ss_q4[:2]]
                _ct_dy0_rk  = [ct_sub(_ct_y0_rk[j], _ct_xss0_rk[j], lwe_q_int) for j in range(2)]
                dy_hist4_ct = [list(_ct_dy0_rk) for _ in range(DEADBEAT_LEN)]
                _steps_since_rekey = 0

            u_lat4_log.append(u_cmd4.copy())
            y_lat4_log.append(y4)
            yr_lat4_log.append(yr4)
            t_lat4_log.append(timestamp4)
            pitch_lat4_log.append(p4)
            pitchrate_lat4_log.append(pr4)
            pitch_hat_lat4_log.append(x_hat[0])
            yaw_hat_lat4_log.append(x_hat[1])
            pitchrate_hat_lat4_log.append(x_hat[2])
            yawrate_hat_lat4_log.append(x_hat[3])

            scope.axes[0].sample(_tg, [TARGET_REF_DEG, np.rad2deg(y4)])

            # Drain any detection-window markers the worker thread produced --
            # the actual pg.LinearRegionItem/InfiniteLine creation happens
            # HERE, on the main/GUI thread, never inside _detector_worker
            # itself (Qt objects must only be touched from the main thread).
            while True:
                try:
                    _mt0, _mt1, _mflag, _mtag = det_marker_queue.get_nowait()
                except queue.Empty:
                    break
                _mark_detection_window(_mt0, _mt1, _mflag, _mtag)

            MultiScope.refreshAll()

            elapsed = time.perf_counter() - t_loop
            if elapsed > DET_PAUSE_SLOW_FACTOR * T_samp:
                # Anomalously slow iteration -- the detector thread shares this
                # process's ONE GIL with us. Its big-int mod arithmetic
                # (nullspace computation, ct_lincomb bookkeeping) doesn't
                # release the GIL the way numpy/BLAS calls do, so a live LLL
                # pass can visibly steal scheduler time from this loop for its
                # whole ~multi-second duration -- during which write_voltage()
                # isn't being refreshed, so physical disturbances (gravity on
                # the yaw arm, etc.) go briefly uncorrected. This is the
                # prime suspect for any periodic "drop" you see on the scope
                # that isn't otherwise explained -- check whether its
                # timestamp lines up with a "[detect] worker picked up
                # window ..." print.
                print(f"[timing t={_tg:7.2f}s] SLOW loop: elapsed={elapsed*1000:7.1f}ms "
                      f"(target={T_samp*1000:.1f}ms) -- likely GIL contention from the "
                      f"live LLL detector thread", flush=True)
                _last_slow_t = time.perf_counter()
                if DET_PAUSE_ON_SLOW_LOOP and not det_pause.is_set():
                    det_pause.set()
                    print(f"[timing t={_tg:7.2f}s] pausing LLL detector until the control "
                          f"loop has been back on schedule for {DET_PAUSE_RECOVER_S:.1f}s",
                          flush=True)
            elif DET_PAUSE_ON_SLOW_LOOP and det_pause.is_set():
                if time.perf_counter() - _last_slow_t >= DET_PAUSE_RECOVER_S:
                    det_pause.clear()
                    print(f"[timing t={_tg:7.2f}s] control loop back on schedule -- "
                          f"resuming LLL detector", flush=True)
            if T_samp - elapsed > 0:
                time.sleep(T_samp - elapsed)
            timestamp4 = time.perf_counter() - startTime4
    except KeyboardInterrupt:
        print("\nControl+detect run interrupted — analysing what was collected.")
    finally:
        myAero2.write_voltage(0.0, 0.0)
        myAero2.write_led(color_done)

    # ── Let the live detector finish its in-flight window ────────────────────
    print("\nControl run finished -- letting the live detector finish its current window...")
    det_pause.clear()    # nothing left to protect -- run at full speed
    det_stop.set()
    det_new_data.set()   # wake the worker if it's blocked in wait()
    try:
        while det_thread.is_alive():
            det_thread.join(timeout=5.0)
            if det_thread.is_alive():
                print("  ...detector still working", flush=True)
    except KeyboardInterrupt:
        print("  interrupted -- not waiting for the detector any longer")
    while True:   # draw any verdicts the worker queued that the loop never got to
        try:
            _mt0, _mt1, _mflag, _mtag = det_marker_queue.get_nowait()
        except queue.Empty:
            break
        _mark_detection_window(_mt0, _mt1, _mflag, _mtag)

    # ── Backfill holes the live detector never covered ───────────────────────
    if DET_FILL_GAPS and det_full_log and not det_thread.is_alive():
        _W = DET_WINDOW
        _tt = np.array([r[0] for r in det_full_log])
        _ep = np.array([r[1] for r in det_full_log])
        _jobs = []   # (i_start, i_end, gap_t0, gap_t1)
        for _e in np.unique(_ep):
            _ii = np.where(_ep == _e)[0]
            _i0, _i1 = int(_ii[0]), int(_ii[-1]) + 1
            if _i1 - _i0 < _W:
                continue
            _e0, _e1 = _tt[_i0], _tt[_i1 - 1]
            _cov = sorted((r["t_start"], r["t_end"]) for r in det_results
                          if "t_start" in r and r["t_start"] >= _e0 - 1e-6 and r["t_end"] <= _e1 + 1e-6)
            _gaps, _cur = [], _e0
            for _a, _b in _cov:
                if _a > _cur:
                    _gaps.append((_cur, _a))
                _cur = max(_cur, _b)
            if _cur < _e1:
                _gaps.append((_cur, _e1))
            for _g0, _g1 in _gaps:
                if _g1 - _g0 < DET_FILL_MIN_GAP_S:
                    continue
                _ga = int(np.searchsorted(_tt, _g0, side="left"))
                _gb = int(np.searchsorted(_tt, _g1, side="right"))   # exclusive
                _ga, _gb = max(_ga, _i0), min(_gb, _i1)
                _s = _ga
                while _s < _gb:
                    _en = _s + _W
                    if _en > _i1:
                        _en = _i1
                        _s = _en - _W   # shift left (overlapping covered time) to keep a full window
                    _jobs.append((_s, _en, _g0, _g1))
                    _s = _en
        if _jobs:
            print(f"\nBackfilling {len(_jobs)} window(s) over uncovered spans >= {DET_FILL_MIN_GAP_S:.0f}s...")
        else:
            print("\nNo uncovered spans >= %.0fs -- nothing to backfill." % DET_FILL_MIN_GAP_S)
        try:
            for _k, (_s, _en, _g0, _g1) in enumerate(_jobs, 1):
                _w = [(r[2], r[3]) for r in det_full_log[_s:_en]]
                _ts, _te = float(_tt[_s]), float(_tt[_en - 1])
                _tag = f"-fill{_k}"
                det_window_count += 1
                _res = detect_window(_w, tag=_tag, t_start=_ts, t_end=_te)
                _res["backfill"] = True
                det_results.append(_res)
                _mark_detection_window(_ts, _te, _res["flagged"], _tag, clip=(_g0, _g1))
                if MultiScope.activeScopes:
                    MultiScope.refreshAll()
        except KeyboardInterrupt:
            print("  backfill interrupted -- reporting what was completed")

    with det_results_lock:
        _n_windows = len(det_results)
        _n_flagged = sum(1 for r in det_results if r["flagged"])
        _flagged_tags = [r["tag"] for r in det_results if r["flagged"]]
        _atk_t0, _atk_t1 = ATTACK_TIME_START, ATTACK_TIME_START + ATTACK_DURATION
        _atk_overlap_flagged = [
            r["tag"] for r in det_results
            if r["flagged"] and "t_start" in r and r["t_start"] < _atk_t1 and r["t_end"] > _atk_t0
        ]

    print(f"Live detector summary: {_n_windows} window(s) processed, {_n_flagged} flagged.")
    if _flagged_tags:
        print(f"  Flagged windows: {_flagged_tags}")
    if not DEBUG_ZERO_ATTACK:
        print(f"Sensor attack: {ATTACK_BIAS_DEG:+.2f} deg bias injected during "
              f"t=[{_atk_t0:.1f}, {_atk_t1:.1f}]s.")
        if _atk_overlap_flagged:
            print(f"  Caught: window(s) {_atk_overlap_flagged} overlapping the attack window were flagged.")
        else:
            print("  NOT caught: no flagged window overlapped the attack window.")
    print(f"Key rotations: {rekey_count}" +
          (f" (every {KEY_ROTATION_PERIOD} samples)" if KEY_ROTATION_PERIOD else " (disabled)"))

    y_lat4_arr     = np.array(y_lat4_log, dtype=float)
    yr_lat4_arr    = np.array(yr_lat4_log, dtype=float)
    t_lat4_arr     = np.array(t_lat4_log, dtype=float)
    pitch_lat4_arr = np.array(pitch_lat4_log, dtype=float)
    pitch_hat_lat4_arr     = np.array(pitch_hat_lat4_log, dtype=float)
    yaw_hat_lat4_arr       = np.array(yaw_hat_lat4_log, dtype=float)
    pitchrate_hat_lat4_arr = np.array(pitchrate_hat_lat4_log, dtype=float)
    yawrate_hat_lat4_arr   = np.array(yawrate_hat_lat4_log, dtype=float)


if PLANT_BACKEND == "tcp_bridge":
    myAero2.close()   # clean "stop" handshake with aero2_plant_bridge.py

# ── Keep the scope window(s) open after the run finishes ───────────────────
# Without this, reaching end-of-script tears down the Qt QApplication (and
# every GraphicsLayoutWidget with it) the instant the control loop/analysis
# above finishes, so the plot would vanish immediately instead of staying up
# for you to look at. Blocking here on the Qt event loop keeps the window(s)
# fully interactive (pan/zoom still work) until you close them yourself (or
# Ctrl+C in the terminal).
if DET_EXIT_WHEN_DONE and MODE != "noise_cal":
    print("\nDetector finished -- exiting.")
    sys.exit(0)
if MultiScope.activeScopes:
    print("\nRun complete -- close the scope window(s) (or Ctrl+C here) to exit.")
    # A bare pg.exec() blocks fully inside Qt's C++ event loop, which starves
    # CPython's signal handling -- Ctrl+C in the terminal can go unnoticed
    # until you click into the window. A no-op QTimer that fires periodically
    # gives the interpreter a chance to actually process the SIGINT.
    _keepalive_timer = pg.QtCore.QTimer()
    _keepalive_timer.timeout.connect(lambda: None)
    _keepalive_timer.start(200)
    try:
        pg.exec()
    except KeyboardInterrupt:
        pass
