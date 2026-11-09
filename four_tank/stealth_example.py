"""Example for the stealth study: a disturbance and a zero-dynamics attack that start 4 steps into key epoch (window) 1.

At step K0 the plant state is kicked to (0, xi) (a disturbance), which would produce a transient.  The attacker injects
the input that keeps the output at zero and also cancels the controller's own input:
    u_total(k) = -(A xhat)_1 / b1   (one step ahead, keeps y = 0),   U_a = round(cu * u_total) - U_c,
with xhat = (x1, g * x2): g = 1 is the matched attacker (knows the hidden state), g = 0.5 guesses half of it, g = 0
injects nothing.  Encrypted loop with the COMPACT parameters; the window is tested with the 6-vector chi-square test.
Writes stealth_example.png/pdf and det_results/stealth_example.npz.  Run with a Python that has blaster or fpylll."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import encrypted, quantized, detect
import zero_attack as za

T = 192
N, KICK, K0, GUESS = 2 * T, T + 4, T + 4, 0.5            # window 1 starts at T; at KICK = K0 the plant state is kicked into (0, xi) and the attack starts
XI = {"stable": 3.0, "unstable": 0.7}                    # kick amplitude: output transient of about 1.2-1.3 in both cases


class KickPlant(za.ZeroPlant):
    """Plant at rest that receives a state kick (0, XI) at step KICK (a disturbance), so that a transient starts there."""
    def __init__(self, z0, xi):
        super().__init__(z0, (0.0, 0.0)); self.k = 0; self.xi = xi

    def apply(self, u):
        super().apply(u); self.k += 1
        if self.k == KICK:
            self.x = self.x + np.array([[0.0], [self.xi]])


def make_spec(kind, xi, xi_att=None):
    spec = za.ZeroSpec(kind, 0.0, xi_att)
    spec.make_plant = lambda: KickPlant(spec.z0, xi)
    return spec


def plan(kind, g, xi):
    spec = make_spec(kind, xi)
    nom = quantized.run(spec, N)                                  # nominal quantized loop (prefix is what the attacker saw)
    pl = spec.make_plant(); xs = []
    for k in range(N):
        xs.append(pl.x[:, 0].copy()); pl.apply(nom["u"][k])
    xs = np.array(xs)
    D, a, b = spec.M["D"][0][0], spec.M["C"][0][0], spec.M["C"][0][1]
    A, B = pl.A, pl.B[:, 0]
    xh = np.array([xs[K0, 0], g * xs[K0, 1]])                    # attacker's belief about the state at K0
    yq = {k: int(nom["yq"][k]) for k in range(K0 + 1)}           # yq history seen up to K0 (yq of later steps: 0)
    Ua = {}
    for k in range(K0, N):
        same = lambda i: (k - i) // spec.kp == k // spec.kp            # controller state is reset every kp steps
        Uc = D * yq.get(k, 0) + (a * yq.get(k - 1, 0) if same(1) else 0) + (b * yq.get(k - 2, 0) if same(2) else 0)
        u_tot = -(A[0] @ xh) / B[0]                               # makes y(k+1) = 0 in the attacker's model
        Utot = int(np.round(spec.cu * u_tot))
        Ua[k] = Utot - Uc
        xh = A @ xh + B * (Utot / spec.cu)
    return spec, Ua


def run(kind, g, xi=None, seed=1):
    xi = XI[kind] if xi is None else xi
    if g is None:
        spec, Ua = make_spec(kind, xi), {}
    else:
        spec, Ua = plan(kind, g, xi)
    out = encrypted.run(spec, N, actuator_attack=lambda k: Ua.get(k, 0), seed=seed, keep_ciphertexts=True)
    res = out["res_ct"]
    Tm = encrypted.noise_map(spec, T)
    dd = [detect.detect_window(res[w * T:(w + 1) * T, :-1], res[w * T:(w + 1) * T, -1], out["q"], spec.lwe["sigma"], Tm) for w in range(2)]
    d = dd[1]                                                     # the window in which the attack starts
    pl = spec.make_plant(); xs = []
    for k in range(N):
        xs.append(pl.x[:, 0].copy()); pl.apply(out["u"][k])
    return dict(y=out["y"], u=out["u"], rho=out["rho"], enc=out["enc_rho"], x2=np.array(xs)[:, 1], q=out["q"], Z=d["Z"],
                flag=bool(d["flag_chi"]), flag0=bool(dd[0]["flag_chi"]), Z0=dd[0]["Z"],
                nz=int(np.count_nonzero(out["rho"][T:])), dec=int(out["dec_errors"]))


if __name__ == "__main__":
    store = {}
    fig, axes = plt.subplots(4, 2, figsize=(10, 8.5), sharex=True)
    cases = (("none", None, "#7f7f7f"), ("mismatched", GUESS, "#0072b2"), ("matched", 1.0, "#d55e00"))
    for j, kind in enumerate(("stable", "unstable")):
        for name, g, col in cases:
            r = run(kind, g); store[f"{kind}_{name}"] = r
            lab = {"none": "no attack", "mismatched": "attack, state guess 0.5 x true", "matched": "attack, state known"}[name]
            lab += f"  [window 1: {'alarm' if r['flag'] else 'no alarm'}, Z={r['Z']:.0f}]"
            t = np.arange(N)
            axes[0, j].plot(t, r["y"], color=col, lw=1.1, label=lab)
            axes[1, j].plot(t, r["u"], color=col, lw=1.1)
            axes[3, j].plot(t, r["x2"], color=col, lw=1.1)
            if name == "matched":
                axes[2, j].plot(t, r["enc"], color="#8c8c8c", lw=0.4, alpha=0.5); axes[2, j].set_ylim(-r["q"] / 2, r["q"] / 2)
                axes[2, j].set_yticks([])
        pax = axes[2, j].twinx(); pax.set_zorder(axes[2, j].get_zorder() + 1); pax.patch.set_visible(False)
        for name, g, col in cases:
            pax.plot(np.arange(N), store[f"{kind}_{name}"]["rho"], color=col, lw=1.0)
        pax.set_ylim(-1.6, 1.6)
        axes[0, j].set_title(f"{kind} zero (z0 = {za.CASES[kind]['z0']})", fontsize=10)
        axes[0, j].legend(fontsize=7, loc="upper right")
        for ax in axes[:, j]:
            ax.axvline(T, color="k", ls="--", lw=0.7); ax.axvline(K0, color="k", ls=":", lw=0.9); ax.grid(alpha=0.3); ax.set_xlim(T - 8, T + 100)
        axes[3, j].set_xlabel("time step")
    for j, kind in enumerate(("stable", "unstable")):
        for i, key in ((0, "y"), (1, "u"), (3, "x2")):
            v = np.concatenate([store[f"{kind}_{n}"][key][T - 8:T + 100] for n in ("none", "mismatched", "matched")])
            pad = 0.08 * (v.max() - v.min() + 1e-9); axes[i, j].set_ylim(v.min() - pad, v.max() + pad)
    for ax, lab in zip(axes[:, 0], ("output y", "applied input u", "encrypted residual (gray)", "hidden state x2")):
        ax.set_ylabel(lab)
    axes[2, 1].yaxis.set_label_position("right")
    fig.suptitle("Zero-dynamics attack starting with a disturbance 4 steps into window 1 (dotted); encrypted loop", fontsize=11, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig("stealth_example.png", dpi=130); fig.savefig("stealth_example.pdf")
    np.savez("det_results/stealth_example.npz", **{f"{k}_{n}": v for k, r in store.items() for n, v in r.items()})
    for k, r in store.items():
        print(f"{k:22s} window0 alarm={r['flag0']} Z0={r['Z0']:.0f} | window1 nonzero residuals {r['nz']:3d}  Z={r['Z']:7.1f}  alarm={r['flag']}  dec_errors={r['dec']}  max|x2|={np.abs(r['x2']).max():.1f}")
