# Draft: reply to Reviewer 4, comment 2 (stealthy attacks), and the paper text it refers to

Files: `stealth_fig_paper.pdf` (figure), `stealth_table.tex` (table), `stealth_example.py` / `stealth_sweep.py` (code).

## Paper paragraph (Section numerical, after the Fig. 7 discussion; wrap in \newtext)

Attacks that keep the output of the closed loop unchanged leave the residual at zero and cannot be detected by any
residual-based test, with or without encryption. A zero-dynamics attack is of this kind. To illustrate this and its
limits, we change the input matrix of the plant to B = [b_1 0.4722]^T, with b_1 chosen so that the plant has a
transmission zero at z_0 = 1.02, outside the unit circle, and keep C = [1 0]. The integer controller is redesigned for
this plant. In the middle of a key epoch a disturbance moves the hidden state to (0, xi), and the attacker injects the
input that keeps the output at zero and cancels the controller's own input. If the attacker knows the hidden state, the
output and the residual stay at zero, the detector does not alarm, and the hidden state grows (Fig. X). If the attacker
guesses half of the state, the output deviates by more than one quantization step, the residual is nonzero, and the
detector alarms. Table X repeats the experiment for guesses between 0.5 and 1, for a stable zero (z_0 = 0.7) and the
unstable one, over five keys. The detector alarms exactly when the output deviation reaches the quantization step.
Encryption does not make a zero-dynamics attack detectable. It keeps the attacker from learning the state that is
needed to construct it.

## Reply (replaces / extends the action text of Reviewer 4, comment 2)

Stealthy attacks are those that leave the residual unchanged, so no residual-based detector can detect them, with or
without encryption. What matters is what an attacker must know to build one. We now show this for zero-dynamics attacks.
In a modified plant with a transmission zero, Fig. X shows a disturbance followed by an attacker who knows the hidden
state: the output and the residual stay at zero, the detector does not alarm, and, for an unstable zero, the hidden state
grows. With a wrong guess of the state the residual is nonzero and the detector alarms. Table X varies the attacker's
guess, the zero location (stable and unstable), and the kick amplitude over five keys: the calm window never alarmed,
the disturbance alone always alarmed, and an attack alarmed exactly when the output deviation reached one quantization
step. The paragraph [location] states that undetectability of such attacks is inherent, and that the protection is that
the state is hidden by the encryption. The remaining part of the comment, on quantitative comparisons with other
detectors, is answered as before (no other residual-based detector works on ciphertexts without the key).

## Open points before this can go in
* The experiments use the Aero2 parameters (q = 2^23+9, sigma = 50) with LLL, not the paper's (q = 16381, sigma = 26)
  that need BKZ. Either rerun at the paper's parameters or state this.
* The plant is the paper's with a changed B and a redesigned controller; the controller gains are the integer ones with
  the smallest spectral radius, not placed poles.
* Kick amplitude: xi = 3 (stable) and 0.7 (unstable) give a similar output transient (about 1.2-1.3); the unstable one is
  limited by the ciphertext range.
* "Attacker knows the state" = plaintext access; the threat model in the paper should say what the attacker is assumed
  to know.
* The figure shows 100 steps; the hidden state reaches 28 within the 192-step window (table).
