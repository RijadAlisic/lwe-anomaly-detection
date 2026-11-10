"""Debug step 2 of the detector: are the vectors really in the kernel of P, and does LLL keep the lattice?
Checks on one real epoch of residual ciphertexts (no attack):
  (1) every row d of the basis from kernel_mod_q satisfies d^T P = 0 mod q
  (2) every row of the LLL-reduced basis satisfies it too
  (3) |det| of both bases equals q^v (the index of the kernel lattice in Z^N), so LLL kept the same lattice
Usage: OMP_NUM_THREADS=1 python verify_kernel.py [SIGMA]"""
import sys
import numpy as np
import encrypted
from quantized import QuantizedLinear
from _kernel import kernel_mod_q
from blaster import reduce as blaster_lll

T = 192
sg = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
spec = QuantizedLinear()
spec.lwe = dict(spec.lwe, sigma=sg, r=int(round(4096 * sg / 50)))
q, v = spec.lwe["q"], spec.lwe["v"]
out = encrypted.run(spec, 2 * T, seed=3, keep_ciphertexts=True)
seg = out["res_ct"][T:2 * T]
P = seg[:, :-1]
print(f"P shape {P.shape}, q={q}, entries in [{P.min()}, {P.max()}]")

def in_kernel(M):                                   # exact integer arithmetic (Python ints)
    Mo, Po = M.astype(object), P.astype(object)
    return bool((((Mo @ Po) % q) == 0).all())

def logdet(M):                                      # log |det| of a square integer matrix (QR, floating point)
    return float(np.sum(np.log(np.abs(np.diag(np.linalg.qr(M.astype(float))[1])))))

B = np.array(kernel_mod_q(P, q), dtype=np.int64)
print(f"\nbasis from kernel_mod_q: shape {B.shape}; rows with d^T P = 0 mod q: {in_kernel(B)}")
R = blaster_lll(B.T, delta=0.99999, cores=1, use_seysen=False, depth=0)[1].T
print(f"LLL-reduced basis:       shape {R.shape}; rows with d^T P = 0 mod q: {in_kernel(R)}")
print(f"log det(B) = {logdet(B):.2f}, log det(R) = {logdet(R):.2f}, v*log q = {v * np.log(q):.2f}")
nrm = np.linalg.norm(R.astype(float), axis=1)
print(f"row norms of R: shortest {nrm.min():.0f}, median {np.median(nrm):.0f}, longest {nrm.max():.0f};  norms of B: shortest {np.linalg.norm(B.astype(float), axis=1).min():.0f}")
print("six shortest rows of R in kernel:", in_kernel(R[np.argsort(nrm)[:6]]))

# ---- centering the basis before LLL ----
# Centering ALL entries is wrong: the 64 rows q*e_pivot become 0 and the lattice shrinks.  Only the 128 rows with a 1 in a free
# coordinate may be centered (that subtracts multiples of the rows q*e_pivot, which stay in the basis).
import time
N = P.shape[0]
nfree = N - v
cen = lambda M: ((M + q // 2) % q) - q // 2
Bwrong = cen(B)
print(f"\ncentering every row: {int((~Bwrong.any(axis=1)).sum())} of {N} rows become the zero vector (log det of the result would be -inf)")
Bc = B.copy(); Bc[:nfree] = cen(B[:nfree])
print(f"centering only the {nfree} free rows: entries in [{Bc.min()}, {Bc.max()}], rows in kernel: {in_kernel(Bc)}, log det {logdet(Bc):.2f}")
nb = np.linalg.norm(B.astype(float), axis=1); nc = np.linalg.norm(Bc.astype(float), axis=1)
print(f"row norms before LLL: uncentered median {np.median(nb):.0f}; centered-free-rows median {np.median(nc):.0f}")
for name, M in (("uncentered", B), ("centered free rows", Bc)):
    t0 = time.perf_counter()
    Rm = blaster_lll(M.T, delta=0.99999, cores=1, use_seysen=False, depth=0)[1].T
    dt = time.perf_counter() - t0
    n = np.sort(np.linalg.norm(Rm.astype(float), axis=1))
    print(f"LLL on {name:18s}: {dt:5.2f} s, in kernel {in_kernel(Rm)}, log det {logdet(Rm):.2f}, six shortest norms {np.round(n[:6]).astype(int).tolist()}, max |entry| of those {int(np.abs(Rm[np.argsort(np.linalg.norm(Rm.astype(float), axis=1))[:6]]).max())}")
