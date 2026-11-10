"""Where does the size of X_i come from?  sigma = 0, attack of length L=1 at the start of epoch 2.
Prints the nonzero residual samples rho_k, the entries d_ik of the six vectors at those samples, and the terms r*d_ik*rho_k."""
import numpy as np
import encrypted
from quantized import QuantizedLinear
from _kernel import kernel_mod_q
from blaster import reduce as blaster_lll

T = 192
spec = QuantizedLinear(); spec.lwe = dict(spec.lwe, sigma=0.0)
q, r = spec.lwe["q"], spec.lwe["r"]
cen = lambda x: ((x + q // 2) % q) - q // 2
out = encrypted.run(spec, 3 * T, sensor_attack=lambda k: 1 if k == 2 * T else 0, seed=3, keep_ciphertexts=True)
seg = out["res_ct"][2 * T:3 * T]
rho = np.rint(out["rho"][2 * T:3 * T] * spec.cs).astype(int)
P, b = seg[:, :-1], seg[:, -1]
R = blaster_lll(np.array(kernel_mod_q(P, q), dtype=np.int64).T, delta=0.99999, cores=1, use_seysen=False, depth=0)[1].T
D = R[np.argsort(np.linalg.norm(R.astype(float), axis=1))[:6]]
nz = np.nonzero(rho)[0]
print(f"r = {r}, q = {q}, q/2 = {q // 2}")
print("nonzero residual samples (index in epoch, value rho_k):", [(int(k), int(rho[k])) for k in nz])
print("typical |d_ik| over all 192 entries of the six vectors: median %.0f, max %d" % (np.median(np.abs(D)), np.abs(D).max()))
for i in range(6):
    terms = [r * int(D[i, k]) * int(rho[k]) for k in nz]
    tot = sum(terms)
    print(f"vec{i}: d_ik at those samples {[int(D[i, k]) for k in nz]}  terms r*d*rho = {terms}  sum = {tot}  X = cen(sum mod q) = {cen(tot)}")
