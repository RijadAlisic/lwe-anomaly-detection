"""Deployment scene of the four-tank experiment, drawn after the KTH lab manual (tanks 3 and 4 above 1 and 2, pumps with three-way
valves, reservoir) and extended with the sensor, the encryption blocks, the controller computer and the detector computer(s).

    scene = Scene(ax, log, mode, n_det=1)   # mode: nominal | quantized | encrypted | detected
    scene.update(n, lamps)                  # draw sample n (1-based); lamps: one colour per detector computer
    scene.artists                           # for blitting

Pixel coordinates of the KTH figure (y pointing down); the tank and pump geometry follows that figure.
"""
import numpy as np
from matplotlib.patches import Rectangle, Circle, Polygon, FancyBboxPatch, Wedge

TW, TH = 97, 190                                    # tank width and height [px]; the drawn tank is 30 cm high
PXCM = TH / 30.0
TANKS = {3: (215, 113), 4: (407, 113), 1: (215, 398), 2: (407, 398)}      # top-left corner of each tank
WATER, BLUE, ORANGE, RED, GREEN, YEL = "#8ec3e8", "#0072b2", "#e69f00", "#d55e00", "#009e73", "#f0e442"
INK = "#111111"


class Scene:
    def __init__(self, ax, log, mode, n_det=1):
        self.ax, self.log, self.mode, self.n_det = ax, log, mode, n_det
        self.enc = mode in ("encrypted", "detected")
        self.det_key = mode == "encrypted"                       # stage 3: the detector holds the key (decrypts); stage 4: it does not
        self.sw, self.aw = log["sensor_window"], log["actuator_window"]
        self.h0, self.u0 = log["h0"], float(log["u0"][0])
        ax.set_xlim(-10, 705); ax.set_ylim(1000, -5); ax.set_aspect("equal"); ax.axis("off")
        self.artists = []
        self._tanks(); self._reservoir(); self._pumps_and_pipes(); self._computers(); self._wires()

    # ---------- static parts ----------
    def _line(self, xs, ys, lw=1.5, color=INK, ls="-", z=2, **kw):
        (ln,) = self.ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="butt", **kw)
        return ln

    def _tanks(self):
        ax = self.ax
        self.water, self.level_txt = {}, {}
        for i, (x, y) in TANKS.items():
            self.water[i] = Rectangle((x, y + TH), TW, 0, facecolor=WATER, edgecolor="none", zorder=1); ax.add_patch(self.water[i])
            self._line([x, x, x + TW / 2 - 5], [y, y + TH, y + TH], lw=2.6, z=3)                  # walls and bottom with the outlet gap
            self._line([x + TW / 2 + 5, x + TW, x + TW], [y + TH, y + TH, y], lw=2.6, z=3)
            self._line([x + TW / 2 - 5] * 2, [y + TH, y + TH + 12], lw=1.6); self._line([x + TW / 2 + 5] * 2, [y + TH, y + TH + 12], lw=1.6)
            ax.text(x + TW / 2, y + 28, f"Tank {i}", ha="center", va="center", fontsize=8, color=INK, zorder=4)
            self.level_txt[i] = ax.text(x + TW / 2, y + TH - 11, "", ha="center", va="center", fontsize=7, color=INK, zorder=4)
            self.dash = ax.plot([x - 4, x + TW + 4], [y + TH - self.h0[i - 1] * PXCM] * 2, color="#555555", lw=0.8, ls=(0, (3, 2)), zorder=4)
        for xs in (263.5, 455.5):                                                              # the upper tanks drain into the lower ones
            ax.annotate("", xy=(xs, 396), xytext=(xs, 318), arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.0, ls=(0, (2, 2)), mutation_scale=7))
        for xs in (263.5, 455.5):                                                              # the lower tanks drain into the reservoir
            ax.annotate("", xy=(xs, 640), xytext=(xs, 596), arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.0, ls=(0, (2, 2)), mutation_scale=7))
        self.sensor = Circle((TANKS[2][0] + TW, TANKS[2][1] + 100), 7, facecolor="white", edgecolor=INK, lw=1.4, zorder=5); ax.add_patch(self.sensor)
        ax.text(TANKS[2][0] + TW + 14, TANKS[2][1] + 78, r"$y$", fontsize=9, ha="left", va="center", color=BLUE)
        ax.text(TANKS[1][0] + TW + 10, TANKS[1][1] + 62, r"$y_1$", fontsize=8, ha="left", va="center", color="#aaaaaa")

    def _reservoir(self):
        ax = self.ax
        ax.add_patch(Rectangle((122, 658), 474, 46, facecolor=WATER, edgecolor="none", zorder=1))
        self._line([120, 120, 598, 598], [635, 706, 706, 635], lw=2.6, z=3)

    def _pumps_and_pipes(self):
        ax = self.ax
        self.pump_txt = {}
        for k, (px, name, sgn) in enumerate(((145, "Pump 1", -1), (572, "Pump 2", 1)), start=1):
            ax.add_patch(Circle((px, 462), 18, facecolor="white", edgecolor=INK, lw=1.6, zorder=3))
            ax.add_patch(Polygon([(px - 10, 470), (px + 10, 470), (px, 452)], closed=True, facecolor="white", edgecolor=INK, lw=1.4, zorder=4))
            self._line([px, px], [682, 480]); self._line([px, px], [444, 386])
            ax.text(px + sgn * 38, 436, name, ha="center", va="center", fontsize=8)
            self.pump_txt[k] = ax.text(px + sgn * 45, 492, f"$u_{k}$", ha="center", va="center", fontsize=8)
            self._valve(px, 367, sgn)
        self._line([145, 145, 455, 455], [348, 60, 60, 92]); self._line([572, 572, 263, 263], [348, 30, 30, 92])
        self._line([160, 241, 241], [367, 367, 390]); self._line([557, 478, 478], [367, 367, 388])
        for xs, ys in ((455, 92), (263, 92), (241, 390), (478, 388)):
            self.ax.add_patch(Polygon([(xs - 4, ys - 7), (xs + 4, ys - 7), (xs, ys)], closed=True, facecolor=INK, edgecolor=INK, zorder=3))
        self.pump_box = FancyBboxPatch((px_ := 145 - 24, 440), 48, 44, boxstyle="round,pad=2,rounding_size=8", facecolor="none", edgecolor=RED, lw=0, zorder=2)
        ax.add_patch(self.pump_box)
        self.pump_volt = ax.text(145, 508, "", ha="center", va="center", fontsize=8, color="#0b3d63")

    def _valve(self, x, y, sgn):
        a = self.ax
        for tri in ([(x - 9, y - 12), (x + 9, y - 12), (x, y)], [(x - 9, y + 12), (x + 9, y + 12), (x, y)], [(x - sgn * 14, y - 9), (x - sgn * 14, y + 9), (x, y)]):
            a.add_patch(Polygon(tri, closed=True, facecolor="white", edgecolor=INK, lw=1.3, zorder=3))
        a.add_patch(Wedge((x + sgn * 11, y), 13, 90 if sgn < 0 else -90, 270 if sgn < 0 else 90, facecolor="white", edgecolor=INK, lw=1.3, zorder=3))

    def _monitor(self, x0, y0, w, h, title, sub=None, dashed=False):
        a = self.ax
        a.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=1,rounding_size=7", facecolor="#f3f3f3", edgecolor=INK, lw=1.6, ls="--" if dashed else "-", zorder=3))
        a.add_patch(Rectangle((x0 + 8, y0 + 8), w - 16, h - 30, facecolor="white", edgecolor="#666666", lw=0.8, zorder=4))
        a.plot([x0 + w / 2 - 18, x0 + w / 2 + 18], [y0 + h + 6, y0 + h + 6], color=INK, lw=2.4, zorder=3)
        a.text(x0 + w / 2, y0 + 25, title, ha="center", va="center", fontsize=8, weight="bold", zorder=5)
        if sub:
            a.text(x0 + w / 2, y0 + 48, sub, ha="center", va="center", fontsize=6.8, color="#444444", zorder=5)

    def _computers(self):
        a = self.ax
        a.add_patch(FancyBboxPatch((8, 742), 684, 232, boxstyle="round,pad=2,rounding_size=12", facecolor="none", edgecolor="#777777", lw=1.2, ls=(0, (5, 3)), zorder=1))
        a.text(18, 760, "network and computers" + (" (untrusted)" if self.enc else ""), fontsize=7.5, color="#555555", va="center")
        self._monitor(70, 800, 190, 120, "Controller", "on ciphertexts" if self.enc else ("integer controller" if self.mode == "quantized" else "floating point"))
        slots = [(330, 520)] if self.n_det == 1 else [(300, 470), (500, 670)]
        sub = "holds the key" if self.det_key else ("no key" if self.enc else "residual test")
        self.lamps, self.det_boxes = [], []
        for k, (x0, x1) in enumerate(slots[: self.n_det]):
            title = "Detector" if self.n_det == 1 else f"Detector {k + 1}"
            self._monitor(x0, 800, x1 - x0, 120, title, sub)
            lamp = Circle((x1 - 26, 880), 11, facecolor="#c8c8c8", edgecolor="#444444", lw=1.2, zorder=6); a.add_patch(lamp)
            self.lamps.append(lamp); self.det_boxes.append((x0, x1))

    def _wires(self):
        a = self.ax; enc = self.enc
        wire = dict(color=BLUE, lw=1.7, zorder=2)
        # measurement: sensor -> (Enc) -> network -> controller and detectors
        self.ywire = [a.plot([505, 632], [498, 498], **wire)[0], a.plot([655, 655, 165], [498, 760, 760], **wire)[0]]
        a.annotate("", xy=(165, 800), xytext=(165, 760), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.7, mutation_scale=9))
        for (x0, x1) in ([(330, 520)] if self.n_det == 1 else [(300, 470), (500, 670)])[: self.n_det]:
            xc = (x0 + x1) / 2
            a.plot([xc, xc], [760, 800], **wire)
            a.annotate("", xy=(xc, 800), xytext=(xc, 770), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.7, mutation_scale=9))
        if enc:
            a.add_patch(Rectangle((632, 484), 46, 28, facecolor="white", edgecolor=INK, lw=1.6, zorder=4)); a.text(655, 498, "Enc", ha="center", va="center", fontsize=8, zorder=5)
            a.add_patch(Rectangle((22, 456), 46, 28, facecolor="white", edgecolor=INK, lw=1.6, zorder=4)); a.text(45, 470, "Dec", ha="center", va="center", fontsize=8, zorder=5)
        else:
            a.plot([632, 678], [498, 498], **wire)
        a.text(612, 515, r"$y^e$" if enc else r"$y$", fontsize=8.5, color=BLUE, ha="center")
        # control signal: controller -> (Dec) -> pump 1
        self.uwire = [a.plot([70, 45], [850, 850], **wire)[0], a.plot([45, 45], [850, 484 if enc else 470], **wire)[0]]
        a.annotate("", xy=(45, 484 if enc else 470), xytext=(45, 520), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.7, mutation_scale=9))
        a.plot([68, 127], [470, 470], color="#777777", lw=1.1, ls=(0, (4, 3)), zorder=2)                  # dashed link into the pump, as in the KTH figure
        a.text(64, 700, r"$u^e$" if enc else r"$u$", fontsize=8.5, color=BLUE, ha="center")
        # predicted output: controller -> detectors, on a bus below the computers
        slots = [(330, 520)] if self.n_det == 1 else [(300, 470), (500, 670)]
        last = slots[: self.n_det][-1][0] - 14
        a.plot([260, 285, 285, last], [905, 905, 952, 952], **dict(color=BLUE, lw=1.4, zorder=2))
        for (x0, x1) in slots[: self.n_det]:
            a.plot([x0 - 14, x0 - 14], [952, 880], color=BLUE, lw=1.4, zorder=2)
            a.annotate("", xy=(x0, 880), xytext=(x0 - 14, 880), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4, mutation_scale=8))
        a.text(272, 968, r"$\hat y^e$" if enc else r"$\hat y$", fontsize=8, color=BLUE, ha="center", va="center")
        # attack markers
        self.bolt_y = Polygon([(690, 560), (670, 585), (682, 585), (670, 615), (698, 578), (685, 578)], closed=True, facecolor="none", edgecolor="#bbbbbb", lw=1.0, zorder=5)
        self.bolt_u = Polygon([(33, 560), (13, 585), (25, 585), (13, 615), (41, 578), (28, 578)], closed=True, facecolor="none", edgecolor="#bbbbbb", lw=1.0, zorder=5)
        a.add_patch(self.bolt_y); a.add_patch(self.bolt_u)
        self.att_txt_y = a.text(655, 632, "", fontsize=7, color=ORANGE, ha="center"); self.att_txt_u = a.text(2, 632, "", fontsize=7, color=RED, ha="left")

    # ---------- per frame ----------
    def update(self, n, lamps=None):
        k = n - 1; h = self.log["h"][k]
        for i, r in self.water.items():
            hh = float(h[i - 1]) * PXCM; y0 = TANKS[i][1] + TH
            r.set_y(y0 - hh); r.set_height(hh); self.level_txt[i].set_text(f"{h[i - 1]:.1f} cm")
        u = self.u0 + float(self.log["u"][k]); self.pump_volt.set_text(f"{u:.2f} V")
        s_on = self.sw[0] <= k <= self.sw[1]; a_on = self.aw[0] <= k <= self.aw[1]
        self.sensor.set_facecolor(ORANGE if s_on else "white")
        self.pump_box.set_edgecolor(RED); self.pump_box.set_linewidth(2.6 if a_on else 0)
        for ln in self.ywire: ln.set_color(ORANGE if s_on else BLUE)
        for ln in self.uwire: ln.set_color(RED if a_on else BLUE)
        self.bolt_y.set_facecolor(ORANGE if s_on else "none"); self.bolt_y.set_edgecolor(ORANGE if s_on else "#bbbbbb")
        self.bolt_u.set_facecolor(RED if a_on else "none"); self.bolt_u.set_edgecolor(RED if a_on else "#bbbbbb")
        self.att_txt_y.set_text("sensor attack" if s_on else ""); self.att_txt_u.set_text("actuator attack" if a_on else "")
        if lamps:
            for lp, c in zip(self.lamps, lamps): lp.set_facecolor(c)
        self.artists = (list(self.water.values()) + list(self.level_txt.values()) + [self.pump_volt, self.sensor, self.pump_box, self.bolt_y, self.bolt_u,
                        self.att_txt_y, self.att_txt_u] + self.ywire + self.uwire + self.lamps)
        return self.artists
