# Encrypted LWE Control with Live Lattice-Based Anomaly Detection (Quanser Aero 2)

A pitch/yaw LQI controller for the Quanser Aero 2 (QA2C) that runs entirely
under LWE (Learning-With-Errors) encryption: state feedback, integral
action, and the observer used for anomaly detection are all evaluated as
homomorphic operations on ciphertexts, with the secret key never appearing
on the plant side. A background thread runs a live BLASTER/LLL lattice
reduction against a sliding window of ciphertexts to detect attacks against
the encrypted control loop (e.g. key-recovery attempts via lattice
reduction) in real time, with periodic LWE key rotation to bound
ciphertext exposure per key.

## Files

- **`aero2_lwe_run.py`** — entry point. All tunables (mode, LQR weights,
  physics parameters, LWE parameters, signal scaling, detector parameters,
  key-rotation policy, run lengths, RNG seeds) live in a single `USER
  CONFIG` block near the top of the file; nothing below it should need
  editing for a normal parameter sweep.
- **`aero2_tools.py`** — helper library: physics-model discretization, LQR
  design, LWE keygen/encrypt/decrypt, ciphertext arithmetic
  (add/sub/negate/scalar-mul/linear-combination), and modular
  nullspace/recentering routines used to build the encrypted observer.
- **`charm_compat.py`** — import shim for `charm.core.math.integer`. If
  `charm-crypto` isn't installed, falls back to plain Python `int`
  (arbitrary precision, safe for this project's parameters: q ~ 2**23,
  n ~ 116). `aero2_lwe_run.py` and `aero2_tools.py` both import from this
  instead of `charm` directly, so charm-crypto is optional, not required.
- **`aero2_plant_bridge.py`** — optional. Only needed if the real Aero2
  hardware and the crypto stack (`charm`/`blaster-python`) can't run in the
  same process/OS on the lab PC. Runs natively wherever the Quanser
  HIL/USB driver lives, owns the `Aero2(hardware=1)` object, and exposes it
  over a small newline-JSON TCP protocol with its own independent safety
  cutout (voltage clamp, pitch cutout, watchdog). `aero2_lwe_run.py` talks
  to it via `RemotePlantProxy` when `PLANT_BACKEND = "tcp_bridge"`. See the
  module docstring for the wire protocol. Not needed if
  `PLANT_BACKEND = "local"` works directly on the lab PC (see Requirements
  below).

`aero2_lwe_run.py` + `aero2_tools.py` (+ `charm_compat.py`) are the minimum
needed to run an experiment. `aero2_plant_bridge.py` is only needed for the
split-process real-hardware setup described above.

## Modes

Set `MODE` in `aero2_lwe_run.py`:

- `"noise_cal"` — drives the real plant with the real encrypted controller
  but decrypts under a zeroed secret, so the measured residual is pure
  noise (no secret-dependent signal). Used to measure the empirical
  residual noise floor before running with a real secret.
- `"control_detect"` — real secret, real encryption, live LLL-based
  anomaly detection on a background thread, with periodic LWE key
  rotation.
- `"both"` — runs `noise_cal` immediately followed by `control_detect` in
  one invocation (one hardware session, one scope window), using the
  noise floor measured in the first phase to calibrate the detector's
  threshold in the second.

## Requirements

- Quanser hardware SDK (`pal`, i.e. Quanser Interactive Labs / QUARC
  Python API) with an Aero 2 (QA2C) connected.
- [`charm-crypto`](https://github.com/JHUISI/charm) (`charm.core.math.integer`)
  — **optional**. If it's not installed, `charm_compat.py` transparently
  falls back to plain Python `int`. Fine to skip on a machine where
  charm won't build (e.g. Windows) — this is the expected workaround
  there, not a blocker.
- `blaster-python` (BLASTER lattice reduction, live LLL anomaly detector)
  — **required**, no fallback exists for this one.
  `pip install blaster-python`. PyPI ships prebuilt wheels for Linux,
  macOS, and **Windows (`win_amd64`, cp311/cp312/cp313)**, so this
  installs cleanly on the Windows lab PC too — no compiler/Eigen3/OpenMP
  setup needed since it doesn't have to build from source.
  **Note for whoever sets up the lab PC:** the package name matters —
  `pip install blaster` (no suffix) grabs an unrelated multiprocessing
  task-runner from a different author. Use `pip install blaster-python`,
  and confirm `import blaster; blaster.reduce` resolves before assuming
  it's working.
- `numpy`, `scipy`

Since `blaster-python` has real Windows wheels and `charm-crypto` is now
optional, running everything as one process directly on the lab PC
(`PLANT_BACKEND = "local"`, `hardware=1`) may just work — try that first.
Fall back to the `aero2_plant_bridge.py` split-process setup only if
`blaster-python` genuinely won't install/import in whatever environment
ends up running the crypto stack there.

`aero2_lwe_run.py` adds its own directory to `sys.path` to find
`aero2_tools.py`, so the files only need to stay together.

## Running

```bash
python aero2_lwe_run.py
```

Set `MODE` and the other knobs in the `USER CONFIG` block first. A
`MultiScope` window plots reference/measured/estimated yaw and pitch,
control effort, and rates in real time.
