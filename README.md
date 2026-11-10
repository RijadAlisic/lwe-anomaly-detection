# Anomaly detection with LWE-encrypted control

Code for the paper *Anomaly Detection with LWE Encrypted Control* (Alisic, Kim, Sandberg).

## 1. A practical deployment: the four-tank process

The level of tank 2 is measured and encrypted at the plant. A **controller computer** works on the ciphertexts and sends an encrypted
command back to pump 1, where it is decrypted. A **detector computer**, which never sees the key, reads the same ciphertexts and the
controller's encrypted prediction and gives a verdict per window of 192 samples. Sensor and actuator attacks hit the two links.

<p align="center"><img src="four_tank/gifs/fourtank_4_detected.gif" width="860" alt="Four-tank process with controller and detector computers"></p>

**A detector that falls behind.** With harder encryption parameters (sigma = 400) a window takes about 250 s to reduce (BKZ-60), against
a window length of 384 s. On a detector computer three times slower, windows arrive faster than they are checked: the detector takes the
newest one, the skipped windows are hatched, and they are checked after the run ends.

<p align="center"><img src="four_tank/gifs/fourtank_5_overloaded.gif" width="860" alt="One slow detector computer misses windows"></p>

**A second detector computer.** Windows are independent, so a second computer takes windows in parallel. Each still needs its time per
window, but nothing is missed.

<p align="center"><img src="four_tank/gifs/fourtank_6_two_detectors.gif" width="860" alt="Two detector computers"></p>

## 2. The build-up to anomaly detection over LWE

The same attack storyline (calm, sensor attack, calm, actuator attack, calm) in four steps. In the first three the detector works on the
residual and decides at every sample; in the last it works on ciphertexts and decides per window.

| Step | Controller | Detector | Gif |
|---|---|---|---|
| 1 | floating point | residual test at every sample | `fourtank_1_nominal.gif` |
| 2 | integer, quantized signals | residual test at every sample | `fourtank_2_quantized.gif` |
| 3 | the integer controller on LWE ciphertexts | residual test on the decrypted residual, at every sample | `fourtank_3_encrypted.gif` |
| 4 | the same, on ciphertexts | **no key**: lattice reduction of the residual ciphertexts, chi-square test, per window | `fourtank_4_detected.gif` |

<p align="center">
<img src="four_tank/gifs/fourtank_1_nominal.gif" width="430"> <img src="four_tank/gifs/fourtank_2_quantized.gif" width="430"><br>
<img src="four_tank/gifs/fourtank_3_encrypted.gif" width="430"> <img src="four_tank/gifs/fourtank_4_detected.gif" width="430">
</p>

```bash
pip install -r requirements.txt
cd four_tank && python make_gifs.py --plant fourtank
```

## 3. The figures of the numerical section

| Paper | Command (from the repository root) |
|---|---|
| Table I | `python analysis/type2_table.py` |
| Fig. 7 | `cd fig7 && python make_fig7.py 3.5 3.5 out.pdf` |
| Fig. 8 | `cd four_tank && python compare_fig_columns.py` |
