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


TANK_H = 30.0                                            # drawn tank height [cm]
TANK_POS = {3: (0.30, 40.0), 4: (2.90, 40.0), 1: (0.30, 0.0), 2: (2.90, 0.0)}      # lower-left corner of each tank
TANK_W = 1.40


def build_tanks(ax, L):
    """Four-tank illustration (tanks 3 and 4 above 1 and 2, as in the lab): the water level of each tank, its
    equilibrium level (dashed) and the pump voltage.  Returns update(n) that draws sample n and the artists it changes."""
    ax.set_xlim(0, 4.6); ax.set_ylim(-12, 80); ax.axis("off")
    h0, u0 = L["h0"], float(L["u0"][0])
    Ts = L["Ts"]; sw, aw = L["sensor_window"], L["actuator_window"]
    water, texts, frames_ = {}, {}, {}
    for i, (x, y) in TANK_POS.items():
        water[i] = plt.Rectangle((x, y), TANK_W, 0.0, facecolor="#6fb4e8", edgecolor="none", zorder=1)
        ax.add_patch(water[i])
        frames_[i] = plt.Rectangle((x, y), TANK_W, TANK_H, facecolor="none", edgecolor="#333333", lw=1.6, zorder=3)
        ax.add_patch(frames_[i])
        ax.plot([x - 0.06, x + TANK_W + 0.06], [y + h0[i - 1]] * 2, color="#333333", lw=0.8, ls=(0, (3, 2)), zorder=4)
        ax.text(x + TANK_W / 2, y + TANK_H + 1.6, f"tank {i}" + (" (h2 measured)" if i == 2 else ""), ha="center", va="bottom", fontsize=8, color="#333333")
        texts[i] = ax.text(x + TANK_W / 2, y + 2.0, "", ha="center", va="bottom", fontsize=8, color="#0b3d63", zorder=5)
    for xs in (0.3 + TANK_W / 2, 2.9 + TANK_W / 2):                     # drain of tank 3 into 1 and of tank 4 into 2
        ax.annotate("", xy=(xs, TANK_H + 0.3), xytext=(xs, 40.0), arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.0))
    pump = ax.text(1.0, -5.0, "", ha="center", va="center", fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#333333", lw=1.2))
    ax.text(1.0, -9.8, "pump 1: tanks 1, 4", ha="center", va="center", fontsize=6.5, color="#555555")
    ax.text(3.6, -5.0, "pump 2: const.", ha="center", va="center", fontsize=7, color="#555555")
    ax.text(3.6, -9.8, "tanks 2, 3", ha="center", va="center", fontsize=6.5, color="#555555")
    note = ax.text(2.3, 77.5, "dashed line: equilibrium level", ha="center", va="center", fontsize=6.5, color="#555555")
    stat = ax.text(0.02, 0.5, "", transform=ax.transAxes)                # placeholder, keeps the artist list uniform

    def update(n):
        k = n - 1
        for i in water:
            water[i].set_height(float(L["h"][k][i - 1]))
            texts[i].set_text(f"{L['h'][k][i - 1]:.1f} cm")
        pump.set_text(f"{u0 + L['u'][k]:.2f} V")
        sensor_on = sw[0] <= k < sw[1] + 1; act_on = aw[0] <= k < aw[1] + 1
        frames_[2].set_edgecolor("#e69f00" if sensor_on else "#333333"); frames_[2].set_linewidth(3.0 if sensor_on else 1.6)
        pump.get_bbox_patch().set_edgecolor("#d55e00" if act_on else "#333333")
        pump.get_bbox_patch().set_linewidth(3.0 if act_on else 1.2)
        note.set_text("sensor attack on h2" if sensor_on else "actuator attack on u1" if act_on else "dashed line: equilibrium level")
        note.set_color("#a06a00" if sensor_on else "#a03a00" if act_on else "#555555")
        return list(water.values()) + list(texts.values()) + list(frames_.values()) + [pump, note, stat]
    return update


def build_diagram(ax, L, mode):
    """How the plant connects to the computer: the plant side holds the key (Enc at the sensor, Dec at the actuator), the
    computer runs the controller on ciphertexts and, in the last stage, the key-less detector.  Returns update(n, lamp)."""
    from matplotlib.patches import FancyBboxPatch, Circle
    ax.set_xlim(-0.06, 4.66); ax.set_ylim(0, 2.45); ax.axis("off")
    enc = mode in ("encrypted", "detected"); det = mode == "detected"
    Ts = L["Ts"]; sw, aw = L["sensor_window"], L["actuator_window"]

    def box(x0, y0, w, h, fc, ec, lw=1.2, ls="-"):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=lw, ls=ls))
    box(0.02, 0.08, 1.56, 2.3, "#eef5fb", "#333333")
    ax.text(0.80, 2.30, "plant side", ha="center", va="top", fontsize=7.5, weight="bold")
    ax.text(0.80, 1.78, "four tanks\nsensor h2\nactuator u1", ha="center", va="top", fontsize=7, color="#333333")
    ax.text(0.80, 0.62, "secret key s\nEnc / Dec" if enc else "no encryption", ha="center", va="center", fontsize=7,
            color="#0b3d63" if enc else "#777777", weight="bold" if enc else "normal")
    box(2.26, 0.08, 2.32, 2.3, "#f6f6f6", "#333333")
    ax.text(3.42, 2.30, "computer (untrusted)" if enc else "computer", ha="center", va="top", fontsize=7.5, weight="bold")
    ctrl = {"nominal": "controller\n(floating point)", "quantized": "integer\ncontroller"}.get(mode, "controller on\nciphertexts")
    box(2.38, 1.12, 2.08, 0.84, "white", "#333333")
    ax.text(3.42, 1.54, ctrl, ha="center", va="center", fontsize=7)
    box(2.38, 0.18, 2.08, 0.78, "white" if det else "#f6f6f6", "#333333" if det else "#aaaaaa", ls="-" if det else (0, (3, 2)))
    ax.text(3.18, 0.57, "anomaly detector\n(no key)" if det else "detector\n(stage 4)", ha="center", va="center", fontsize=7,
            color="#000000" if det else "#999999")
    lamp = Circle((4.22, 0.57), 0.13, fc="#c8c8c8", ec="#555555", lw=1.0, visible=det); ax.add_patch(lamp)
    arr = dict(arrowstyle="-|>", color="#333333", lw=1.6, mutation_scale=9)
    ay = ax.annotate("", xy=(2.26, 1.66), xytext=(1.58, 1.66), arrowprops=arr)      # measurement to the computer
    au = ax.annotate("", xy=(1.58, 1.30), xytext=(2.26, 1.30), arrowprops=arr)      # control signal back to the plant
    ytxt = ax.text(1.92, 1.74, r"$y^e$" if enc else r"$y$", ha="center", va="bottom", fontsize=7.5, linespacing=0.95)
    utxt = ax.text(1.92, 1.22, r"$u^e$" if enc else r"$u$", ha="center", va="top", fontsize=7.5, linespacing=0.95)
    if det:                                                      # the detector gets the controller's prediction and reads the measurement
        ax.annotate("", xy=(3.42, 0.97), xytext=(3.42, 1.12), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2, mutation_scale=7))
        ax.text(3.50, 1.045, r"$\hat y^e$", ha="left", va="center", fontsize=6.5, color="#555555")

    def update(n, lamp_color=None):
        k = n - 1
        s_on = sw[0] <= k < sw[1] + 1; a_on = aw[0] <= k < aw[1] + 1
        ay.arrow_patch.set_color("#e69f00" if s_on else "#333333"); ay.arrow_patch.set_linewidth(3.2 if s_on else 1.6)
        au.arrow_patch.set_color("#d55e00" if a_on else "#333333"); au.arrow_patch.set_linewidth(3.2 if a_on else 1.6)
        base_y, base_u = (r"$y^e$", r"$u^e$") if enc else (r"$y$", r"$u$")
        ytxt.set_text(base_y + "\nattacked" if s_on else base_y); utxt.set_text(base_u + "\nattacked" if a_on else base_u)
        ytxt.set_color("#a06a00" if s_on else "#000000"); utxt.set_color("#a03a00" if a_on else "#000000")
        if lamp_color:
            lamp.set_facecolor(lamp_color)
        return [ay, au, ytxt, utxt, lamp]
    return update


def make_gif(plant, mode, number, lim, frames, outdir, seconds=10.0, workers=1, time_scale=1.0):
    L = ex.run(plant, mode)
    t, Ts = L["t"], L["Ts"]
    detected = mode == "detected"
    tanks = plant == "fourtank" and "h" in L             # the four-tank gifs get a tank illustration on the left
    if tanks:
        fig = plt.figure(figsize=(10.4, 7.4 if detected else 6.6))
        nrow = 4 if detected else 3
        gs = fig.add_gridspec(nrow, 2, width_ratios=[1.25, 2.4], height_ratios=[3, 3, 3, 1.1][:nrow], wspace=0.40)
        axes = [fig.add_subplot(gs[0, 1])]
        axes += [fig.add_subplot(gs[i, 1], sharex=axes[0]) for i in range(1, nrow)]
        for a in axes[:-1]:
            a.tick_params(labelbottom=False)
        left = gs[:, 0].subgridspec(2, 1, height_ratios=[3.5, 1.5], hspace=0.03)
        tank_update = build_tanks(fig.add_subplot(left[0]), L)
        diag_update = build_diagram(fig.add_subplot(left[1]), L, mode)
        sax, axes = (axes[3], axes[:3]) if detected else (None, axes)
    elif detected:                                       # a fourth row for the detection flags
        fig, axes = plt.subplots(4, 1, figsize=(6.4, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 3, 3, 1.1]))
        sax, axes = axes[3], axes[:3]
    else:
        fig, axes = plt.subplots(3, 1, figsize=(6.4, 5.4), sharex=True)
    lines, enc_lines = [], []
    encrypted = "enc_y" in L
    for ax, (key, lab), unit in zip(axes, ROWS, UNITS[plant]):
        ax.axvspan(L["sensor_window"][0] * Ts, L["sensor_window"][1] * Ts, color="#e69f00", alpha=0.15, lw=0)
        ax.axvspan(L["actuator_window"][0] * Ts, L["actuator_window"][1] * Ts, color="#d55e00", alpha=0.15, lw=0)
        ax.set_xlim(t[0], t[-1] + Ts); ax.grid(alpha=0.3)
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
    if detected:
        d = L["det"]; Tw = int(d["T"]) * Ts; nw = len(d["window"])
        patches = []
        for w in range(nw):
            patch = plt.Rectangle((w * Tw, 0.12), Tw, 0.76, facecolor="#ececec", edgecolor="white", lw=1.0)
            sax.add_patch(patch); patches.append(patch)
        sax.set_ylim(0, 1); sax.set_yticks([]); sax.set_ylabel("detection", rotation=0, ha="right", va="center", fontsize=9)
        sax.set_xlabel(TIME[plant]); sax.tick_params(axis="x", labelbottom=True)
        sax.text(1.01, 0.72, "alarm", transform=sax.transAxes, color="#d55e00", fontsize=8, va="center")
        sax.text(1.01, 0.28, "no alarm", transform=sax.transAxes, color="#009e73", fontsize=8, va="center")
        sax.text(1.01, 0.5, "computing", transform=sax.transAxes, color="#a08c00", fontsize=8, va="center")
    else:
        axes[2].set_xlabel(TIME[plant])
    if encrypted:                                       # what one sample costs, for this parameter set
        tm = L["timing"]
        fig.text(0.02 if tanks else 0.13, 0.012, f"LWE n=64, q=2^23+9, sigma=50 | per sample (Python, 1 core): encrypt {tm['encrypt']:.2f} ms, "
                 f"control {tm['evaluate']:.2f} ms, decrypt {tm['decrypt']:.2f} ms", fontsize=6, color="#555555")
    if detected:
        fig.text(0.02 if tanks else 0.13, 0.038, f"detection per window of {int(d['T'])} samples (kernel lattice + LLL, Python, 1 core): "
                 f"{np.mean(d['seconds']):.0f} s on average, {int(d['K'])}-vector chi-square test at alpha = {float(d['alpha']):g}; "
                 f"plant and {workers} detector worker(s) in parallel, newest window first",
                 fontsize=6, color="#555555")
    if tanks:
        fig.subplots_adjust(left=0.01, right=0.90, top=0.93, bottom=0.17 if detected else 0.13, hspace=0.12)
    else:
        fig.subplots_adjust(left=0.13, right=0.86, top=0.93, bottom=0.17 if detected else 0.13, hspace=0.12)
    idx = np.unique(np.linspace(1, len(t), frames).astype(int))
    if detected:
        # one common simulation clock: the plant runs at the sample rate, window w is available when its last
        # sample has arrived, the detector workers start and finish at the times given by replay.schedule
        Tn = int(d["T"]); t_run = len(t) * Ts
        sch = replay.schedule((np.arange(nw) + 1) * Tn * Ts, d["seconds"], t_run, workers, time_scale)
        dt = t_run / len(idx)
        taus = np.arange(1, int(np.ceil(max(t_run, sch["finish"].max()) / dt)) + 1) * dt
        seq = [float(x) for x in taus] + [float(taus[-1])] * 15
    else:
        seq = [int(i) for i in idx]

    def draw(f):
        lamp = None
        if detected:
            n = max(1, min(len(t), int(f / Ts)))
            for w in range(nw):
                if f >= sch["finish"][w]:
                    c = "#d55e00" if d["flag_chi"][w] else "#009e73"
                elif f >= sch["start"][w]:
                    c = "#f0e442"
                elif f >= sch["avail"][w]:
                    c = "#c8c8c8"
                else:
                    c = "#ececec"
                patches[w].set_facecolor(c)
            done = [w for w in range(nw) if f >= sch["finish"][w]]
            busy = any(sch["start"][w] <= f < sch["finish"][w] for w in range(nw))
            if busy:
                lamp = "#f0e442"
            elif done:
                last = max(done, key=lambda w: sch["finish"][w])
                lamp = "#d55e00" if d["flag_chi"][last] else "#009e73"
            else:
                lamp = "#c8c8c8"
        else:
            n = f
        for ln, dot, key in lines:
            ln.set_data(t[:n], L[key][:n]); dot.set_data([t[n - 1]], [L[key][n - 1]])
        for eln, ek in enc_lines:
            eln.set_data(t[:n], L[ek][:n])
        extra_artists = (tank_update(n) + diag_update(n, lamp)) if tanks else []
        return [x for ln, dot, _ in lines for x in (ln, dot)] + [eln for eln, _ in enc_lines] + extra_artists

    anim = FuncAnimation(fig, draw, frames=seq, blit=not detected)
    path = os.path.join(outdir, f"{plant}_{number}_{mode}.gif")
    anim.save(path, writer=PillowWriter(fps=len(idx) / seconds))
    if detected:
        lat = sch["finish"] - sch["avail"]
        print(f"  detector latency after the window closes: mean {lat.mean():.1f} s, max {lat.max():.1f} s (simulation seconds); "
              f"last flag at {sch['finish'].max():.0f} s, run ends at {t_run:.0f} s")
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
    a = ap.parse_args()
    os.makedirs("gifs", exist_ok=True)
    for plant in ("linear", "fourtank"):
        if a.plant and plant != a.plant:
            continue
        lim = shared_limits(plant)
        for number, mode in enumerate(ex.MODES, start=1):
            if a.only and mode != a.only:
                continue
            p = make_gif(plant, mode, number, lim, a.frames, "gifs", a.seconds, a.workers, a.time_scale)
            print(p, f"{os.path.getsize(p) / 1e3:.0f} kB")
