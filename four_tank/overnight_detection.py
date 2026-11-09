"""Heavy part of step 4: short filtering vectors for the paper's parameters (v = 64, N = 192, q = 16381).

Per detection window: build the q-ary kernel lattice of the residual ciphertexts, LLL, then progressive BKZ with
g6k sieving (the routine of the research code), and keep the K shortest vectors.  Results are written per
window, and windows that are already done are skipped, so the run can be stopped and resumed.

Run it with the Python that has g6k and fpylll (the research virtual environment):

  RES=/home/rijad/Anomaly-Detector-under-Encryption
  $RES/.venv/bin/python overnight_detection.py --research-dir $RES --windows 0-31 --threads 8 \\
        --beta-start 50 --beta-max 85 > overnight.log 2>&1 &

First export the ciphertexts with:  python export_windows.py     (creates windows_linear_paper.npz)
Memory: sieving at block size 85-90 needs several GB per process, so use --jobs 1 and all threads.
"""
import argparse, os, sys, time
import numpy as np


def parse_windows(spec, n):
    if spec == "all":
        return list(range(n))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def reduce_window(w, a):
    """Reduce window w and return the K shortest vectors."""
    sys.path.insert(0, a.research_dir)
    from fpylll import IntegerMatrix, LLL, BKZ
    from functions import run_progressive_betas_verbose
    import _kernel
    D = np.load(a.windows)
    P, q = D["P"][w], int(D["q"])
    t0 = time.time()
    basis = _kernel.kernel_mod_q(P, q)
    A = IntegerMatrix.from_matrix([[int(x) for x in row] for row in basis])
    LLL.reduction(A)
    for bs in (20, 30, 40):                                   # cheap pre-reduction with fpylll BKZ
        if bs < a.beta_start:
            BKZ.reduction(A, BKZ.Param(block_size=bs, max_loops=4))
    A, _, _ = run_progressive_betas_verbose(A, det=None, beta_start=a.beta_start, beta_step=a.beta_step,
                                            beta_max=a.beta_max, max_tours_per_beta=a.tours, threads=a.threads)
    R = np.array([[A[i, j] for j in range(A.ncols)] for i in range(A.nrows)], dtype=np.int64)
    norms = np.linalg.norm(R.astype(float), axis=1)
    idx = np.argsort(norms)[:a.k]
    V = R[idx]
    ok = all(not ((V[i].astype(object) @ P.astype(object)) % q).any() for i in range(len(V)))
    return V, norms[idx], time.time() - t0, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--research-dir", required=True, help="folder with functions.py (the research code)")
    ap.add_argument("--windows-file", dest="windows", default="windows_linear_paper.npz")
    ap.add_argument("--windows", dest="wsel", default="all", help="e.g. all, 0-31, 0,8-15")
    ap.add_argument("--out", default="det_results")
    ap.add_argument("--k", type=int, default=6, help="number of short vectors to keep")
    ap.add_argument("--beta-start", type=int, default=50)
    ap.add_argument("--beta-step", type=int, default=5)
    ap.add_argument("--beta-max", type=int, default=85)
    ap.add_argument("--tours", type=int, default=2, help="max tours per block size")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n = np.load(a.windows)["P"].shape[0]
    for w in parse_windows(a.wsel, n):
        path = os.path.join(a.out, f"window_{w:02d}.npz")
        if os.path.exists(path):
            print(f"window {w}: done, skipping", flush=True)
            continue
        print(f"\n######## window {w} ########", flush=True)
        V, norms, secs, ok = reduce_window(w, a)
        np.savez(path, vectors=V, norms=norms, seconds=secs, ok=ok, window=w, beta_max=a.beta_max)
        print(f"window {w}: {secs / 60:.1f} min, shortest norms {np.round(norms, 1)}, d^T P = 0: {ok}", flush=True)


if __name__ == "__main__":
    main()
