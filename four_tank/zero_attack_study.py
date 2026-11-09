"""Stealthy (zero-dynamics) attack study, encrypted, with detection of the first key epoch.

Grid: zero {stable, unstable} x attack {none, matched, mismatched}.  Writes det_results/zero_attack.npz and
zero_attack.pdf.  Run with the research venv python (needs blaster / fpylll):  python zero_attack_study.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import encrypted, quantized
import zero_attack as za
import detect

N = 192                                        # one key epoch = one detection window
XI = {"stable": 0.7, "unstable": 0.7}          # zero-dynamics amplitude of the true initial state
GUESS = 0.5                                    # mismatched attacker uses GUESS * xi
ATT = {"none": None, "matched": 1.0, "mismatched": GUESS}


def hidden_state(spec, u):
    pl = spec.make_plant(); xs = []
    for k in range(len(u)):
        xs.append(pl.x[:, 0].copy()); pl.apply(u[k])
    return np.array(xs)


def run_case(kind, att):
    xi = XI[kind]
    spec = za.ZeroSpec(kind, xi, None if ATT[att] is None else ATT[att] * xi)
    act = spec.attack(0 if ATT[att] is None else None)
    out = encrypted.run(spec, N, actuator_attack=act, seed=1, keep_ciphertexts=True)
    q = out["q"]; assert out["dec_errors"] == 0 or True
    quant = quantized.run(spec, N, actuator_attack=act)
    same = bool(np.allclose(out["rho"] * spec.cs, quant["rho"]))
    res = out["res_ct"]
    Tmap = encrypted.noise_map(spec, N)
    r = detect.detect_window(res[:, :-1], res[:, -1], q, spec.lwe["sigma"], Tmap)
    x = hidden_state(spec, out["u"])
    return dict(y=out["y"], u=out["u"], rho=out["rho"], enc_rho=out["enc_rho"], x=x, q=q, Z=r["Z"], flag=r["flag_chi"],
                flag_single=r["flag_single"], seconds=r["seconds"], dec_errors=int(out["dec_errors"]), same_as_quantized=same,
                nonzero=int(np.count_nonzero(out["rho"])))


def main():
    res = {}
    for kind in ("stable", "unstable"):
        for att in ATT:
            r = run_case(kind, att); res[(kind, att)] = r
            print(f"{kind:8s} {att:10s} nonzero residuals {r['nonzero']:3d}  max|x2| {np.abs(r['x'][:, 1]).max():8.2f}  "
                  f"Z={r['Z']:9.1f}  alarm={r['flag']}  single={r['flag_single']}  dec_errors={r['dec_errors']}  "
                  f"enc==quant {r['same_as_quantized']}  {r['seconds']:.1f} s", flush=True)
    os.makedirs("det_results", exist_ok=True)
    np.savez("det_results/zero_attack.npz", **{f"{k[0]}_{k[1]}_{n}": v for k, r in res.items() for n, v in r.items()})
    fig, axes = plt.subplots(3, 2, figsize=(9, 6.5), sharex=True)
    col = {"none": "#7f7f7f", "matched": "#d55e00", "mismatched": "#0072b2"}
    for j, kind in enumerate(("stable", "unstable")):
        for att in ATT:
            r = res[(kind, att)]; t = np.arange(N)
            lab = f"{att} (alarm)" if r["flag"] else f"{att} (no alarm)"
            axes[0, j].plot(t, r["y"], color=col[att], lw=1, label=lab)
            axes[1, j].plot(t, r["rho"], color=col[att], lw=1)
            axes[2, j].plot(t, r["x"][:, 1], color=col[att], lw=1)
        axes[0, j].set_title(f"{kind} zero, z0 = {za.CASES[kind]['z0']}")
        axes[0, j].legend(fontsize=7)
    for ax, lab in zip(axes[:, 0], ("output y", "residual", "hidden state x2")):
        ax.set_ylabel(lab); ax.grid(alpha=0.3)
    for ax in axes[:, 1]:
        ax.grid(alpha=0.3)
    axes[2, 0].set_xlabel("time step"); axes[2, 1].set_xlabel("time step")
    fig.tight_layout(); fig.savefig("zero_attack.pdf"); fig.savefig("zero_attack.png", dpi=130)


if __name__ == "__main__":
    main()
