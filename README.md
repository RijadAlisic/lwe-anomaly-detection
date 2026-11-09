# Anomaly detection with LWE-encrypted control: reproducible examples

Code accompanying the paper *Anomaly Detection with LWE Encrypted Control* (Alisic, Kim, Sandberg).
It contains three small, self-contained examples. Nothing here needs the secret key at the detector.

| Folder | What it does | Run time |
|---|---|---|
| `table1/` | Exact Type II errors of the detector's test and of the attacker's test (Table I of the paper) | seconds |
| `fig7/` | Regenerates the numerical-results figure of the paper from the saved run | seconds |
| `four_tank/` | Encrypted control of a nonlinear four-tank process, detection of sensor and actuator attacks | about 4 minutes |

## Quick start

```bash
pip install -r requirements.txt
python table1/type2_table.py                    # reproduces Table I (max deviation 0.07 percentage points)
python fig7/make_fig7.py 3.5 3.5 fig7/out.pdf   # Fig. 7 from fig7/windowed_det_data.npz
cd four_tank && python run_demo.py --windows 8  # detection rate versus attack size
```

## Staged simulation (`four_tank/`)

The simulation is built in steps, each runnable on its own, for two plants: the linear plant of the paper and the
nonlinear four-tank process of the KTH lab.

| Step | Script | Content |
|---|---|---|
| 1 | `step1_feedback_attacks.py` | plants in closed loop with sensor and actuator attack injections (floating point) |
| 2 | `step2_quantization.py` | quantized signals and the integer controller of eq. (23); the linear plant reproduces the paper's saved run exactly |
| 3 | `encrypted.py` | encrypted control; the log adds the encrypted measurement, input and residual |
| 4 | `detect.py` | anomaly detection with the Aero2 parameters: per window of 192 residual ciphertexts, kernel lattice, LLL (blaster), 6-vector chi-square test; about 2 s per window in Python on one core (measured, see `det_results`). Run with the research venv (needs `blaster`/`fpylll`); results are stored in `det_results/<plant>_aero2.npz` and drawn by `make_gifs.py` as `*_4_detected.gif` (the flags fill in after the signals) |

### Encrypted control: parameter sets and decryption failures

`mode="encrypted"` evaluates the integer controller on LWE ciphertexts and also logs the encrypted values (second
ciphertext component) of the measurement, the input and the residual, drawn as a gray background in the gif, as in
Fig. 7 of the paper. Two parameter sets are defined in `quantized.py`:

| Preset | v | q | sigma | r | Used for |
|---|---|---|---|---|---|
| `AERO2` | 64 | 2^23 + 9 | 50 | 4096 | the gifs (Aero2 / Busan demo parameters; r chosen for this q) |
| `PAPER` | 64 | 16381 | 26 | 400 | the paper's example; `export_windows.py` and the overnight detection |

Both plants use the same kind of integer controller with scale `c = 1` (`cs = 5` for both; `cu = 5` for the linear
plant and `cu = 10` for the four-tank), so the scaled signals stay small and a modulus of 8.4 million is enough.
With `PAPER`, the noise reaching the controller input is `e_(k-2) - 2 e_(k-1)` (standard deviation about
2.2 sigma = 58, against the limit r/2 = 200), so **decryption fails in a few steps of most 6144-step runs**
(0 to 5 per run, mean about 2.4; about 5.7e-4 per step). With `AERO2` there are none. `encrypted.run` counts them
(`dec_errors`). The gifs are meant to show the cost of each stage for a fixed parameter set: the encrypted gifs print
the time per sample of encryption, homomorphic control and decryption (Python, one core). Security levels are not
claimed for `AERO2`; the paper's single 80-bit experiment is the `PAPER` one.

### Animated plots

`python make_gifs.py` writes one gif per controller version and plant to `four_tank/gifs/`, in the order
nominal, quantized (encrypted and detected follow with steps 3 and 4). All gifs share the layout, the axis limits
and a fixed attack storyline (`experiment.py`): calm start, sensor attack in the second quarter, recovery, actuator
attack in the last quarter, fixed random seed. `experiment.run(plant, mode)` is the single entry point.

## `four_tank/`: what is implemented

* **Plant**: the minimum-phase four-tank process of the KTH water-tank lab (parameters and equations from the
  lab's MATLAB sources), simulated nonlinearly with RK4. `u1` is the control input, `h2` the output, `u2` is fixed.
* **Encryption**: the secret-key LWE scheme of the paper, `Enc(m) = (P, P s + r m + e mod q)` (`lwe.py`).
* **Controller**: an observer-based controller designed for the linearized plant, rounded to integers with
  scale `c` and evaluated on ciphertexts as in eq. (23) of the paper: periodic state reset every `kp` steps,
  encrypted input `U`, encrypted prediction `Yhat`, encrypted residual `c^(kbar+1) Y - Yhat` (`loop.py`).
  The script checks that the decrypted residual equals a plaintext shadow of the controller.
* **Detector**: for each key epoch it builds the q-ary kernel lattice of the residual ciphertexts, finds a short
  filtering vector `d` with LLL, and applies the two-sided test of Theorem 1 (`detector.py`).
* **Attacks**: an additive attack on the measurement ciphertext (sensor) and an injection on the pump voltage (actuator).

## Important limits of this demo

* **The LWE parameters are toy parameters** (`v = 8`, `q` about 2^40, `sigma = 2`) so that exact LLL finishes in
  seconds. They are **not secure**. The paper uses `v = 64`, `q = 16381`, and BKZ (g6k) for the lattice step.
* **Nominal residual is exactly zero because the measurement is quantized with a deadband** (+-0.5 cm).
  Attacks whose physical effect stays inside the deadband are invisible to any residual-based detector
  (in the run below, an actuator injection of 0.05 V is not detected, as expected).
* The noise covariance used by the test is computed exactly from the controller (`Loop.noise_map`).
* Results of one run (`--windows 8 --alpha 0.01`, 96 windows): false-alarm rate 0/24, sensor bias of 1 to 5 steps
  detected in 100% of the windows, actuator injection of 0.05 V detected 0%, 0.1 V 88%, 0.2 V and above 100%
  (`four_tank/results.csv`, `four_tank/detection_vs_attack.pdf`).
* The Fig. 7 example of the paper (linear plant, `v = 64`) used the authors' research code, which depends on
  g6k; `fig7/` ships the saved detector output and the plotting script only.

## License

To be decided by the authors before publication.
