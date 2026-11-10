"""Animated plots, one gif per controller version, in the order nominal -> quantized -> (encrypted -> detected).

Every gif has the same layout and the same attack storyline (same scenario, same axis limits per plant), so
consecutive gifs can be compared by flipping through them.  Usage:  python make_gifs.py [--frames 100] [--seconds 10]
Writes gifs/<plant>_<number>_<mode>.gif
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import experiment as ex
import replay
from scene import Scene

TITLES = {"linear": "Linear plant of the paper", "fourtank": "Nonlinear four-tank process"}
MODE_LABEL = {"nominal": "nominal control (floating point)", "quantized": "quantized control (integer controller)",
              "encrypted": "encrypted control (LWE ciphertexts)",
              "detected": "encrypted control with anomaly detection"}
ENC_KEY = {"y": "enc_y", "u": "enc_u", "rho": "enc_rho"}
ROWS = (("y", "output y"), ("u", "applied input u"), ("rho", "residual"))
UNITS = {"linear": ("", "", ""), "fourtank": (" [cm]", " [V]", " [cm]")}
TIME = {"linear": "time [s]", "fourtank": "time [s]"}


def shared_limits(plant):
    """Axis limits from all modes, so that the gifs of one plant are directly comparable."""
    logs = [ex.run(plant, m) for m in ex.MODES if m != "detected"]
    lim = {}
    for key, _ in ROWS:
        lo = min(l[key].min() for l in logs); hi = max(l[key].max() for l in logs)
        pad = 0.08 * (hi - lo + 1e-9)
        lim[key] = (lo - pad, hi + pad)
    return lim


TAU = 0.05                                          # per-sample alarm threshold on the residual [output units]; the quantization step is 0.2


def alarm_bins(rho, n_bins=1200):
    """Per-sample alarm |rho_k| > TAU, reduced to n_bins bins (a bin alarms if any of its samples does) for the strip."""
    al = np.abs(np.asarray(rho)) > TAU
    edges = np.linspace(0, len(al), n_bins + 1).astype(int)
    return np.array([al[edges[i]:max(edges[i + 1], edges[i] + 1)].any() for i in range(n_bins)]), al


def make_gif(plant, mode, number, lim, frames, outdir, seconds=10.0, workers=1, time_scale=1.0,
             lwe="compact", det_tag="compact", drop=False, name=None, catchup=False):
    from matplotlib.colors import ListedColormap
    L = ex.run(plant, mode, lwe=lwe, det_tag=det_tag)
    t, Ts = L["t"], L["Ts"]
    detected = mode == "detected"
    scene_on = plant == "fourtank" and "h" in L          # the four-tank gifs show the deployment scene on the left
    n_det = workers if (detected and scene_on) else 1
    if scene_on:
        fig = plt.figure(figsize=(11.8, 7.9))
        gs = fig.add_gridspec(4, 2, width_ratios=[0.9, 1.6], height_ratios=[3, 3, 3, 1.15], wspace=0.04)
        axes = [fig.add_subplot(gs[0, 1])] + [fig.add_subplot(gs[i, 1], sharex=None) for i in range(1, 4)]
        scene = Scene(fig.add_subplot(gs[:, 0]), L, mode, n_det=n_det)
    else:
        fig, axes = plt.subplots(4, 1, figsize=(6.4, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 3, 3, 1.15]))
        scene = None
    sax, axes = axes[3], axes[:3]
    for a in axes:
        a.set_xlim(t[0], t[-1] + Ts)
    sax.set_xlim(t[0], t[-1] + Ts)
    for a in axes:
        a.tick_params(labelbottom=False)
    lines, enc_lines = [], []
    encrypted = "enc_y" in L
    for ax, (key, lab), unit in zip(axes, ROWS, UNITS[plant]):
        ax.axvspan(L["sensor_window"][0] * Ts, L["sensor_window"][1] * Ts, color="#e69f00", alpha=0.15, lw=0)
        ax.axvspan(L["actuator_window"][0] * Ts, L["actuator_window"][1] * Ts, color="#d55e00", alpha=0.15, lw=0)
        ax.grid(alpha=0.3)
        pax = ax
        if encrypted:                                   # encrypted values as a gray background (left), plaintext on top (right)
            half = L["q"] / 2
            (eln,) = ax.plot([], [], color="#8c8c8c", lw=0.5, alpha=0.4)
            ax.set_ylim(-half, half); ax.set_yticks([]); ax.set_ylabel("encrypted", color="#8c8c8c", fontsize=8)
            enc_lines.append((eln, ENC_KEY[key]))
            pax = ax.twinx(); pax.set_zorder(ax.get_zorder() + 1); pax.patch.set_visible(False)
        (ln,) = pax.plot([], [], color="#0072b2", lw=0.8)
        (dot,) = pax.plot([], [], "o", color="#0072b2", ms=3.5)
        pax.set_ylim(*lim[key]); pax.set_ylabel(lab + unit)
        if encrypted:
            pax.yaxis.set_label_position("right")
        lines.append((ln, dot, key))
    fig.suptitle(f"{TITLES[plant]}: {MODE_LABEL[mode]}", fontsize=10, weight="bold", x=0.02, ha="left")
    tr = axes[0].get_xaxis_transform()                  # x in data, y as fraction of the axis height
    axes[0].text(0.5 * (L["sensor_window"][0] + L["sensor_window"][1]) * Ts, 0.97, "sensor attack", ha="center",
                 va="top", fontsize=7, color="#a06a00", transform=tr, zorder=10)
    axes[0].text(0.5 * (L["actuator_window"][0] + L["actuator_window"][1]) * Ts, 0.97, "actuator attack", ha="center",
                 va="top", fontsize=7, color="#a03a00", transform=tr, zorder=10)
    sax.set_yticks([]); sax.set_xlabel(TIME[plant])
    if detected:                                        # our detector: one verdict per window of T samples
        d = L["det"]; Tw = int(d["T"]) * Ts; nw = len(d["window"])
        patches = []
        for w in range(nw):
            patch = plt.Rectangle((w * Tw, 0.12), Tw, 0.76, facecolor="#ececec", edgecolor="white", lw=1.0)
            sax.add_patch(patch); patches.append(patch)
        sax.set_ylim(0, 1); sax.set_ylabel("detection\n(per window)", rotation=0, ha="right", va="center", fontsize=8)
        sax.text(1.01, 0.72, "alarm", transform=sax.transAxes, color="#d55e00", fontsize=8, va="center")
        sax.text(1.01, 0.28, "no alarm", transform=sax.transAxes, color="#009e73", fontsize=8, va="center")
        sax.text(1.01, 0.5, "computing", transform=sax.transAxes, color="#a08c00", fontsize=8, va="center")
        if drop:
            sax.text(1.01, 0.06, "missed", transform=sax.transAxes, color="#555555", fontsize=8, va="center")
    else:                                               # the residual test: one verdict per sample
        bins, alarm = alarm_bins(L["rho"])
        im = sax.imshow(np.zeros((1, len(bins))), aspect="auto", cmap=ListedColormap(["#ececec", "#d55e00"]), vmin=0, vmax=1,
                        extent=[t[0], t[-1] + Ts, 0, 1], interpolation="nearest", zorder=1)
        sax.set_ylim(0, 1); sax.set_ylabel("detection\n(every sample)", rotation=0, ha="right", va="center", fontsize=8)
        sax.text(1.01, 0.7, "alarm", transform=sax.transAxes, color="#d55e00", fontsize=8, va="center")
        sax.text(1.01, 0.3, r"$|\rho_k|>%g$" % TAU, transform=sax.transAxes, color="#555555", fontsize=7, va="center")
    if encrypted:                                       # what one sample costs, for this parameter set
        tm = L["timing"]
        fig.text(0.02, 0.012, f"LWE n={L['lwe']['v']}, q=2^23+9, sigma={L['lwe']['sigma']:g}, r={L['lwe']['r']} | per sample (Python, 1 core): encrypt {tm['encrypt']:.2f} ms, "
                 f"control {tm['evaluate']:.2f} ms, decrypt {tm['decrypt']:.2f} ms", fontsize=6, color="#555555")
    if detected:
        fig.text(0.02, 0.036, f"detection per window of {int(d['T'])} samples ({str(d['method']) if 'method' in d else 'kernel lattice + LLL'}, Python, 1 core): "
                 f"{np.mean(d['seconds']):.0f} s on average, {int(d['K'])}-vector chi-square test at alpha = {float(d['alpha']):g}; "
                 f"{workers} detector computer(s), newest window first", fontsize=6, color="#555555")
        note = []
        if time_scale != 1:
            note.append(f"detector time x{time_scale:g} (slower machine)")
        if drop:
            note.append("skipped windows are checked after the run" if catchup else "windows it cannot start in time are missed")
        if note:
            fig.text(0.02, 0.060, "; ".join(note), fontsize=6, color="#a03a00")
    if scene_on:
        fig.subplots_adjust(left=0.01, right=0.925, top=0.95, bottom=0.115, hspace=0.12)
    else:
        fig.subplots_adjust(left=0.13, right=0.86, top=0.93, bottom=0.17, hspace=0.12)
    idx = np.unique(np.linspace(1, len(t), frames).astype(int))
    if detected:
        # one common simulation clock: the plant runs at the sample rate, window w is available when its last
        # sample has arrived, the detector computers start and finish at the times given by replay.schedule
        Tn = int(d["T"]); t_run = len(t) * Ts
        sch = replay.schedule((np.arange(nw) + 1) * Tn * Ts, d["seconds"], t_run, workers, time_scale, drop=drop, catchup=catchup)
        dt = t_run / len(idx)
        taus = np.arange(1, int(np.ceil(max(t_run, np.nanmax(sch["finish"])) / dt)) + 1) * dt
        seq = [float(x) for x in taus] + [float(taus[-1])] * 15
    else:
        seq = [int(i) for i in idx]

    def draw(f):
        lamps = None
        if detected:
            n = max(1, min(len(t), int(f / Ts)))
            for w in range(nw):
                patches[w].set_hatch(None); patches[w].set_edgecolor("white")
                if f >= sch["finish"][w]:
                    c = "#d55e00" if d["flag_chi"][w] else "#009e73"
                elif f >= sch["start"][w]:
                    c = "#f0e442"
                elif drop and sch["missed"][w] and f >= sch["missed_at"][w]:
                    c = "#ffffff"; patches[w].set_hatch("xxx"); patches[w].set_edgecolor("#b0b0b0")
                elif f >= sch["avail"][w]:
                    c = "#c8c8c8"
                else:
                    c = "#ececec"
                patches[w].set_facecolor(c)
            lamps = []                                  # one lamp per detector computer
            for k in range(n_det):
                mine = [w for w in range(nw) if sch["worker"][w] == k]
                if any(sch["start"][w] <= f < sch["finish"][w] for w in mine):
                    lamps.append("#f0e442")
                else:
                    done = [w for w in mine if f >= sch["finish"][w]]
                    if done:
                        last = max(done, key=lambda w: sch["finish"][w])
                        lamps.append("#d55e00" if d["flag_chi"][last] else "#009e73")
                    else:
                        lamps.append("#c8c8c8")
            extra = []
        else:
            n = f
            cur = np.zeros(len(bins)); cur[: int(np.ceil(n / len(t) * len(bins)))] = bins[: int(np.ceil(n / len(t) * len(bins)))]
            im.set_data(cur[None, :])
            lamps = ["#d55e00" if alarm[max(0, n - 10):n].any() else "#009e73"]
            extra = [im]
        for ln, dot, key in lines:
            ln.set_data(t[:n], L[key][:n]); dot.set_data([t[n - 1]], [L[key][n - 1]])
        for eln, ek in enc_lines:
            eln.set_data(t[:n], L[ek][:n])
        extra_artists = (scene.update(n, lamps) if scene_on else []) + extra
        return [x for ln, dot, _ in lines for x in (ln, dot)] + [eln for eln, _ in enc_lines] + extra_artists

    anim = FuncAnimation(fig, draw, frames=seq, blit=False)
    path = os.path.join(outdir, name or f"{plant}_{number}_{mode}.gif")
    anim.save(path, writer=PillowWriter(fps=len(idx) / seconds))
    if detected:
        lat = sch["finish"] - sch["avail"]
        print(f"  detector latency after the window closes: mean {np.nanmean(lat):.1f} s, max {np.nanmax(lat):.1f} s (simulation seconds); "
              f"last flag at {np.nanmax(sch['finish']):.0f} s, run ends at {t_run:.0f} s; missed windows: {int(sch['missed'].sum())} of {nw}")
    plt.close(fig)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=10.0, help="duration of each gif")
    ap.add_argument("--workers", type=int, default=1, help="detector workers running beside the plant")
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help="simulation seconds per measured detector second (1 = as measured)")
    ap.add_argument("--only", default=None, help="render only this mode (e.g. detected)")
    ap.add_argument("--plant", default=None, choices=("linear", "fourtank"), help="render only this plant")
    ap.add_argument("--sigma", type=float, default=None, help="LWE noise for the encrypted and detected gifs (r = 40 sigma)")
    ap.add_argument("--det-tag", default="compact", help="use det_results/<plant>_<tag>.npz for the detected gif")
    ap.add_argument("--drop", action="store_true", help="live detector: windows it cannot start in time are missed, no catching up")
    ap.add_argument("--catchup", action="store_true", help="with --drop: the detector checks the missed windows after the run ends")
    ap.add_argument("--name", default=None, help="file name of the gif (with --only)")
    a = ap.parse_args()
    os.makedirs("gifs", exist_ok=True)
    for plant in ("linear", "fourtank"):
        if a.plant and plant != a.plant:
            continue
        lim = shared_limits(plant)
        for number, mode in enumerate(ex.MODES, start=1):
            if a.only and mode != a.only:
                continue
            from quantized import COMPACT
            lw = dict(COMPACT, sigma=a.sigma, r=int(round(40 * a.sigma))) if a.sigma else "compact"
            p = make_gif(plant, mode, number, lim, a.frames, "gifs", a.seconds, a.workers, a.time_scale,
                         lwe=lw, det_tag=a.det_tag, drop=a.drop, name=a.name, catchup=a.catchup)
            print(p, f"{os.path.getsize(p) / 1e3:.0f} kB")
