# Anomaly detection with LWE-encrypted control

Reproducible examples for the paper *Anomaly Detection with LWE Encrypted Control* (Alisic, Kim, Sandberg).
A controller that works on LWE ciphertexts, and a detector that, without the secret key, flags attacks on the loop.

<p align="center">
  <img src="four_tank/gifs/fourtank_4_detected.gif" width="640" alt="Four-tank process: encrypted control with anomaly detection">
</p>

The animation is the end of a staged simulation: the levels of the four tanks (left), the signals of the nonlinear
four-tank process under sensor and actuator attacks, the encrypted values in the gray background, and the detector's
verdict per key epoch filling in below.

## What is here

| Folder | Content |
|---|---|
| `four_tank/` | staged simulation of two plants (nominal, quantized, encrypted, detected), the stealthy-attack study, and the comparison experiment of Section VII |
| `analysis/` | exact Type II errors of the detector's test and of the attacker's test (Table I), their crossovers, and the BKZ block size for a target vector length |
| `fig8/` | regenerates the numerical-results figure of the paper (Fig. 8) from the saved run |
| `aero2_demo/` | the live hardware demo of the talks: encrypted LQI control of a Quanser Aero 2 with live lattice-based anomaly detection |

## From the paper to the code

| Paper | What | Command (from the repository root) |
|---|---|---|
| Table I | Type II error of the detector and, in parentheses, of the attacker's test | `python analysis/type2_table.py` |
| Fig. 8 | detection on ciphertexts, 6-vector test | `cd fig8 && python make_fig8.py 3.5 3.5 out.pdf` |
| Fig. 9 | alarms against the number of nonzero residuals, three attacks, sigma = 50 and 75 | `cd four_tank && python compare_fig_columns.py` (from the saved sweeps) |
| Section VII, BKZ paragraph | block size and cost for a target vector length | `python analysis/bkz_cost.py` |
| Table I discussion | Type II error against alpha and sigma-tilde, crossover of detector and attacker | `cd analysis && python beta_vs_alpha.py` (also `beta_vs_sigma.py`, `alpha_plus_beta.py`) |

## Quick start

```bash
pip install -r requirements.txt
python analysis/type2_table.py                     # Table I (matches the paper within 0.05 percentage points)
python analysis/bkz_cost.py                        # kappa = 80 gives ||d|| of about 1.4e3 at a cost of 2^23 per call
cd four_tank && python compare_fig_columns.py      # Fig. 9 from the saved sweeps (a few seconds)
python make_gifs.py                                # the gifs below (needs the saved det_results, a few minutes)
```

Re-running the detector on the staged simulation (about 2 s per window per plant, needs `blaster-python` or `fpylll`):

```bash
python detect.py linear      # and: python detect.py fourtank      -> det_results/<plant>_compact.npz
```

## The staged simulation

Two plants: the linear plant of the paper (Ts = 1 s) and the nonlinear four-tank process of the KTH lab (Ts = 2 s;
equations and parameters from the lab's MATLAB sources, `h2` controlled by pump 1, linearized around the exact equilibrium).
Each stage adds one thing and has its own script; `experiment.run(plant, mode)` is the single entry point and
`make_gifs.py` draws one gif per stage. All gifs use the same attack storyline (calm start, sensor attack, recovery,
actuator attack, calm end; 36 key epochs of 192 steps, fixed seed) and the same axis limits per plant.

| Stage | Mode | What changes | Script |
|---|---|---|---|
| 1 | `nominal` | floating-point observer-based controller, attack injections | `step1_feedback_attacks.py` |
| 2 | `quantized` | quantized signals and the integer controller of eq. (23) with periodic state reset | `step2_quantization.py` |
| 3 | `encrypted` | the integer controller evaluated on LWE ciphertexts; adds the encrypted measurement, input and residual | `encrypted.py`, `lwe.py` |
| 4 | `detected` | key-less detection per key epoch: kernel lattice of the residual ciphertexts, LLL, 6-vector chi-square test | `detect.py`, `replay.py` |

| | Linear plant | Four-tank process |
|---|---|---|
| 1. nominal | <img src="four_tank/gifs/linear_1_nominal.gif" width="330"> | <img src="four_tank/gifs/fourtank_1_nominal.gif" width="430"> |
| 2. quantized | <img src="four_tank/gifs/linear_2_quantized.gif" width="330"> | <img src="four_tank/gifs/fourtank_2_quantized.gif" width="430"> |
| 3. encrypted | <img src="four_tank/gifs/linear_3_encrypted.gif" width="330"> | <img src="four_tank/gifs/fourtank_3_encrypted.gif" width="430"> |
| 4. detected | <img src="four_tank/gifs/linear_4_detected.gif" width="330"> | <img src="four_tank/gifs/fourtank_4_detected.gif" width="430"> |

The four-tank gifs have an illustration of the tanks on the left: the water level of each tank (drawn to scale for a 30 cm
tank), its equilibrium level (dashed), the pump voltage, and a highlight on tank 2 or pump 1 while the sensor or the
actuator is attacked. Tanks 1 and 4 move most under the actuator attack, since pump 1 feeds them.

In the encrypted gifs the gray background is the second ciphertext component (centered modulo q), as in Fig. 8 of the
paper, and the footer gives the time per sample for encryption, homomorphic control and decryption (about 0.12, 0.09 and
0.07 ms in Python on one core). In the detected gifs a window is gray until its last sample has arrived, yellow while
the detector works on it, and green (no alarm) or red (alarm) afterwards. The plant runs at its sampling rate and
never waits; the detector runs beside it, newest window first, using the measured compute time per window (about
2 s). Window lengths are 192 s (linear) and 384 s (four-tank), so the flags appear about 2 s after each window closes.
`make_gifs.py --plant fourtank --time-scale S --workers N` rescales the detector time or the number of detector workers.

## The comparison experiment (Section VII, Fig. 9)

A systematic sweep of three attacks of varying stealth on one plant, against the test on the decrypted residual, which
alarms whenever the residual is nonzero (the nominal residual is exactly zero here). The plant is the linear plant of the
paper with the first element of B changed, B(1) = -A(1,2) B(2) / (z0 - A(2,2)), so that it has a transmission zero z0 = 0.7;
the loop is encrypted with `(v, r, sigma, q) = (64, 2000, 50, 2^23 + 9)` and, dashed in the figure, sigma = 75 at the same r.

| Attack | Script | Saved result |
|---|---|---|
| noise injection of length L = 0 ... 192, 100 keys each | `compare_detectors.py` | `det_results/detector_comparison_random_r2000_fine_stablezero.json` (sigma = 50), `..._sigma75_r2000_...json` |
| zero-dynamics attack with amplitude g, stable (z0 = 0.7) and unstable (z0 = 1.02) zero, 50 keys each | `zero_sweep_fine.py` | `det_results/zero_sweep_fine.json`, `zero_sweep_fine_more.json`, `zero_sweep_fine_sigma75.json` |

```bash
cd four_tank
ATTACK=random FINE=1 PLANT=stablezero R_OVER_SIGMA=40 OMP_NUM_THREADS=1 python compare_detectors.py 100                  # sigma = 50
ATTACK=random FINE=1 PLANT=stablezero SIGMA=75 R_OVER_SIGMA=26.67 OMP_NUM_THREADS=1 python compare_detectors.py 100       # sigma = 75, r = 2000
OMP_NUM_THREADS=1 python zero_sweep_fine.py 50                                                                          # g grid set by GS_JSON
SIGMA=75 TAG=_sigma75 OMP_NUM_THREADS=1 python zero_sweep_fine.py 50
```

Each detected key epoch costs about 3 s of LLL on one core, so a sweep of about 1100 runs takes a few core-hours (`WORKERS`
sets the number of processes, default 5). With sigma = 75 some keys have decryption errors; the figure plots only the
error-free runs. The unstable-zero plant uses the lower-gain controller found by `design_unstable.py`.

The detector uses only LLL, which gives filtering vectors of norm about 7.5e3 (root Hermite factor 1.019); the Gaussian
heuristic predicts vectors of about 680. `analysis/bkz_cost.py` gives the BKZ block size for a target length: kappa = 80 would give
about 1.4e3 at a cost of about 2^23 per call, and reaching the Gaussian heuristic requires kappa close to the number of samples (192),
at a cost of about 2^55.

## Aero 2 hardware demo

`aero2_demo/` is the code of the live demonstration shown in the talks (IFAC World Congress, Busan; MIT LIDS): a pitch/yaw
LQI controller for the Quanser Aero 2 that runs entirely on LWE ciphertexts, with a background lattice reduction (BLASTER/LLL)
over a sliding window of ciphertexts as the live detector and periodic key rotation. It needs the Quanser hardware SDK and
an Aero 2, so unlike the rest of the repository it cannot be run without the hardware. See `aero2_demo/README.md` for the
modes (`noise_cal`, `control_detect`, `both`), the requirements, and the optional TCP bridge for split-process setups.

## Stealthy attacks

`four_tank/STEALTH.md`: a zero-dynamics attack on a plant with a stable or an unstable transmission zero. An attacker who
knows the hidden state keeps the output and the residual at zero, the detector does not alarm, and for an unstable zero
the hidden state grows. With a wrong guess the residual is nonzero and the detector alarms (160 runs, 5 key seeds,
`stealth_sweep.py`). The sweep in the paper (above) is the larger version of this study.

<p align="center"><img src="four_tank/stealth_fig_paper.png" width="420"></p>

## Parameters

| Preset | v | q | sigma | r | Used for |
|---|---|---|---|---|---|
| `COMPACT` | 64 | 2^23 + 9 | 50 | 4096 | all gifs and `stealth_sweep.py` (r chosen for this q) |
| sweep | 64 | 2^23 + 9 | 50 and 75 | 2000 | the comparison experiment (Fig. 9), lower-gain controllers |
| `PAPER` | 64 | 16381 | 26 | 400 | the paper's example in Fig. 8; `export_windows.py` and `overnight_detection.py` |

With `COMPACT` there are no decryption failures. With `PAPER` the noise reaching the controller input has standard
deviation about 58 against the limit r/2 = 200, so decryption fails in a few steps of most 6144-step runs; `encrypted.run`
counts them (`dec_errors`). **No security level is claimed for `COMPACT` or the sweep parameters**; the paper's single
80-bit experiment is the `PAPER` one. The detector for `PAPER` needs BKZ with sieving (g6k): `export_windows.py` exports
the ciphertexts and `overnight_detection.py` reduces them window by window (resumable); it is provided here and its
results are not.

`sigma_sweep.py` and `bkz_probe.py` try larger noise on single four-tank windows (sigma = 200: LLL misses the attack and
BKZ-40 recovers it in about 4 s; at sigma = 400 to 1600, BKZ-50 is no longer enough). These are probes, not a study.

## Sanity checks

`verify_kernel.py` (the shortest vectors lie in the kernel of P mod q), `verify_zd_formula.py` (the closed form of the
zero-dynamics attack), `sigma0_check.py` and `signal_scaling_check.py` (the statistic with no noise and with scaled
signals, which show where wrap-around sets in), `diagnose_vectors.py`, `zero_nz_map.py` and `min_r_check.py`.

## Limits

* The nominal residual is exactly zero because the measurement is quantized with a deadband. Attacks whose effect stays
  inside the deadband are invisible to any residual-based detector, encrypted or not. With process and measurement
  noise the nominal residual is nonzero, which is Problem 2 of the paper; the experiments here do not include noise.
* The statistic saturates once the filtered residual wraps around modulo q: the number of nonzero residuals beyond
  the first few does not raise it further, which is why the alarm fraction in Fig. 9 grows only over the first few.
* The noise covariance used by the test is computed exactly from the controller (`encrypted.noise_map`).
* The detector times are from one Python process on one core; they show the order of magnitude.
* The Fig. 8 example of the paper used the authors' research code, which depends on g6k; `fig8/` ships the saved
  detector output and the plotting script only.

## License

To be decided by the authors before publication.
