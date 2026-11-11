# Anomaly Detection with LWE Encrypted Control

Code for the paper *Anomaly Detection with LWE Encrypted Control* (Alisic, Kim, Sandberg).

## 1. A practical deployment example: the four-tank process

In this example we consider a feedback controller of a four-tank system, but we restrict ourselves to the SISO system where Pump 1
controls the level of Tank 2 (Pump 2 is fixed at steady state).

The system is controlled over homomorphically encrypted signals (using the LWE scheme). The controller uses
$u_k = \hat y_{k-1} - 2y_{k-1}$ (up to scaling), where the output observer is simply $\hat y_k = y_{k-1}$.

We consider a scenario where an attacker (a) injects random noise into the sensor signal, and (b) injects random noise into the
actuator signal.

<p align="center"><img src="four_tank/gifs/fourtank_4_detected.gif" width="860" alt="Four-tank process with controller and detector computers"></p>

In gray one can see the encrypted signal, and the signal in blue is the plaintext signal (if no encryption was applied). The two attack
windows are colored in yellow and orange.

The encryption parameters here, $(v, q, \sigma) = (64, 2^{23}+9, 50)$, are weak: the best known attacks cost about $2^{39}$ operations (a
textbook estimate, not a security claim). They are weak enough for the detection to succeed with the polynomial-time LLL algorithm,
and the detector works in roughly real time. However, that also means that an attacker can break the system quickly.

Next we make the noise larger, $\sigma = 400$ ($r$ scaled from 4096 to 16000). LLL is no longer sufficient, and the detector has to rely on
BKZ-60. A window now takes about 250 s to reduce, against a window length of 384 s. On a detector computer three times slower than ours
(stated on the gif), the detector falls behind: it always takes the most recent window, so it misses some windows and the detection
becomes patchy. The gaps are filled in afterwards, but any attack found then happened a long time ago.

<p align="center"><img src="four_tank/gifs/fourtank_5_overloaded.gif" width="860" alt="One slow detector computer misses windows"></p>

However, the detection can be parallelized. The windows are independent, so a second detector computer takes windows in parallel. Each
computer still needs its time per window, but no window is missed.

<p align="center"><img src="four_tank/gifs/fourtank_6_two_detectors.gif" width="860" alt="Two detector computers"></p>

## 2. The build-up to anomaly detection over LWE

We build up the story in four steps, with the same attacks as above: random noise in the sensor signal, then random noise in the actuator
signal.

**Step 1: nominal control.** A floating-point controller, no quantization, no encryption. The detector is a residual test: it raises an
alarm whenever $|\rho_k| > 0.05$ cm, where $\rho_k = y_k - \hat y_k$ is the difference between the measurement and the prediction. It
decides at every sample, so the alarms (the red strip below) appear the moment an attack starts.

<p align="center"><img src="four_tank/gifs/fourtank_1_nominal.gif" width="860" alt="Step 1: nominal control with a residual test at every sample"></p>

**Step 2: quantized control.** The signals are now integers and the controller is the integer controller, with a state reset every 64
steps. In calm periods the residual is zero (apart from the start-up), so any nonzero residual is an alarm. The detector still decides at
every sample.

<p align="center"><img src="four_tank/gifs/fourtank_2_quantized.gif" width="860" alt="Step 2: quantized control with a residual test at every sample"></p>

**Step 3: encrypted control.** The integer controller now runs on LWE ciphertexts (in gray). The plant side holds the key, so it can
decrypt the residual, and the same residual test works at every sample: encryption does not change the plaintext signals. This is the
benchmark that any detector on ciphertexts is compared with.

<p align="center"><img src="four_tank/gifs/fourtank_3_encrypted.gif" width="860" alt="Step 3: encrypted control with a residual test on the decrypted residual"></p>

**Step 4: detection without the key.** The detector computer only has the ciphertexts. It builds a lattice from the residual
ciphertexts of one window, reduces it to find short vectors that cancel the secret, and applies a chi-square test to the result. It
therefore needs a whole window of 192 samples and gives one verdict per window, after the window has closed. That is the price of not
having the key.

<p align="center"><img src="four_tank/gifs/fourtank_4_detected.gif" width="860" alt="Step 4: detection on ciphertexts, one verdict per window"></p>

## 3. The figures of the numerical section

| Paper | Command (from the repository root) |
|---|---|
| Table I | `python analysis/type2_table.py` |
| Fig. 7 | `cd fig7 && python make_fig7.py 3.5 3.5 out.pdf` |
| Fig. 8 | `cd four_tank && python compare_fig_columns.py` |
