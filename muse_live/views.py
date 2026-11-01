"""The window: one figure, several views, two sets of radio buttons, and one tick function."""
import time

import numpy as np

from .signals import FS, FS_IMU, FS_OPT, CH, BANDS, eeg, acc, gyro, optics
from .analysis import (filtered_raw, band_powers, contact_quality, best_channel, eeg_spectrogram, HeadPose,
                       FaceDetector, NodDetector, EpochAverager, SpectrumAverager, BAD_CONTACT_UV)
from .sound import SOUND_MODES, SOUND_LABELS

VIEWS = ["overview", "spectrogram", "zoo", "nod", "erp", "assr"]
VIEW_LABELS = ["overview", "EEG spectrogram", "artefact zoo", "nod metronome", "evoked response", "40 Hz entrainment"]

HEADERS = {
    "overview": (
        "Four dry electrodes read the electrical activity of the brain: TP9 and TP10 behind the ears, AF7 and AF8 on the forehead, "
        "in microvolts, filtered to 1–40 Hz.\nBlinks, eye movements and jaw tension make the biggest waves (artefacts); a red label "
        "means an electrode is not touching the skin. Band power splits the last two seconds into the\nclassic rhythms; alpha "
        "(8–13 Hz) grows when the eyes close and the visual cortex idles. Try it: eyes closed for 15 s, then open, and watch the red "
        "curve step down.\nThe headband also carries an accelerometer and a gyroscope: tilting the head shifts the three lines, a nod "
        "makes a bump. Keys: v next view, 0–7 sound, r record, space marker."),
    "spectrogram": (
        "The same picture you have read for sound, now of a brain signal: time across, frequency up to 40 Hz, power as colour, "
        "one-second windows stepped by a quarter second.\nClose your eyes and a horizontal band appears near 10 Hz: alpha. A blink "
        "is a vertical smear at the bottom; a jaw clench fills the top. Everything you know about window size applies."),
    "zoo": (
        "Most of what an untrained eye calls brain activity is the face. Follow the prompts; each leaves its own signature in the "
        "traces, and the detector marks what it recognises.\nBlinks are large slow waves on the forehead pair; eye movements are "
        "slow steps; a jaw clench is a burst of fast activity on every channel; chewing and speech are rhythmic bursts."),
    "nod": (
        "The gyroscope sees every nod as a spike of rotation speed. Nod steadily and the metronome (key m) follows your rate; then "
        "keep nodding with the clicks.\nThe histogram shows where each nod lands relative to the nearest click, in milliseconds. "
        "Most people land a little early: the negative mean asynchrony from the chapter on time and rhythm."),
    "erp": (
        "A click sounds about once a second (key e). Around each click the EEG is cut out and averaged. A single response is buried "
        "in noise; after forty or fifty clicks a negative dip near 100 ms and a positive bump\nnear 200 ms should rise out of it on the "
        "forehead channels. This is how event-related potentials are made. Bluetooth timing jitter smears the peaks; sit still and avoid "
        "blinking.\nKey c resets the average."),
    "assr": (
        "A 1 kHz tone pulses 40 times a second (key a). The auditory brainstem and cortex follow that rhythm, so an averaged spectrum "
        "of the EEG should grow a peak at exactly 40 Hz.\nThis is the auditory steady-state response, an entrainment measured in the "
        "brain. With four dry electrodes it is marginal; the number on the right is the peak against its\nneighbours. The tall peak at 50 Hz is the mains hum, picked up by the body as an antenna, not the brain; it is notched out of the average. Key c resets."),
}
ZOO_PROMPTS = [("sit still, eyes open", 6), ("blink three times", 6), ("look left, then right, twice", 6),
               ("clench your jaw for two seconds", 6), ("chew", 6), ("say a sentence aloud", 6),
               ("nod three times", 6), ("close your eyes", 10)]


def _clean(ax, left=True):
    for s in ("top", "right") + (() if left else ("left",)):
        ax.spines[s].set_visible(False)


class Window:
    def __init__(self, source, fake=False, athena=False, engine=None, recorder=None, view="overview", sound=None):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RadioButtons
        self.plt = plt
        self.source, self.fake, self.athena, self.engine, self.recorder = source, fake, athena, engine, recorder
        self.view = view
        self.t0 = time.monotonic()
        self.pose = HeadPose(); self.face = FaceDetector(); self.nods = NodDetector()
        self.epochs = EpochAverager(); self.spectrum = SpectrumAverager()
        self.alpha_hist = []            # (t, alpha dB)
        self.band_hist = []             # (t, 4 bands)
        self.events = []                # (kind, time)
        self.zoo_start = None
        self.metronome_on = False
        self.last_period = None

        plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "toolbar": "None"})
        self.fig = plt.figure(figsize=(13.5, 10.5))
        try:
            self.fig.canvas.manager.set_window_title("Muse live")
        except Exception:
            pass
        self.title = self.fig.text(0.5, 0.98, "Muse live", ha="center", va="top", fontsize=15, weight="bold")
        self.header = self.fig.text(0.02, 0.945, "", ha="left", va="top", fontsize=9.2, linespacing=1.4,
                                    bbox=dict(boxstyle="round,pad=0.6", facecolor="#f3f3f3", edgecolor="none"))
        self.status = self.fig.text(0.01, 0.005, "", fontsize=9, color="#555")
        # controls on the right
        ax_view = self.fig.add_axes([0.80, 0.56, 0.18, 0.24]); ax_view.set_title("View (key v cycles)", fontsize=10)
        ax_snd = self.fig.add_axes([0.80, 0.14, 0.18, 0.36]); ax_snd.set_title("Sound (click, or press the key)", fontsize=10)
        for ax in (ax_view, ax_snd):
            ax.set_facecolor("#f3f3f3")
            for s in ax.spines.values(): s.set_visible(False)
        self.r_view = RadioButtons(ax_view, VIEW_LABELS, active=VIEWS.index(view))
        self.r_snd = RadioButtons(ax_snd, SOUND_LABELS, active=SOUND_MODES.index(sound))
        for r in (self.r_view, self.r_snd):
            for lbl in r.labels: lbl.set_fontsize(9)
        self.r_view.on_clicked(lambda label: self.set_view(VIEWS[VIEW_LABELS.index(label)]))
        self.r_snd.on_clicked(lambda label: self.set_sound(SOUND_MODES[SOUND_LABELS.index(label)]))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.axes = []
        self.update = lambda: None
        if engine is not None and sound:
            engine.set_mode(sound)
        self.set_view(view)

    # ------------------------------------------------------------------ controls
    def on_key(self, event):
        k = event.key
        if k in "01234567" and len(k) == 1:
            self.r_snd.set_active(int(k))
        elif k == "v":
            self.r_view.set_active((VIEWS.index(self.view) + 1) % len(VIEWS))
        elif k == "r" and self.recorder is not None:
            self.recorder.stop() if self.recorder.active else self.recorder.start()
        elif k == " " and self.recorder is not None:
            self.recorder.mark(self.zoo_prompt()[0] if self.view == "zoo" else "mark")
        elif k == "m" and self.engine is not None:
            self.metronome_on = not self.metronome_on
            if not self.metronome_on:
                self.engine.set_metronome(None)
        elif k == "e" and self.engine is not None:
            self.engine.set_erp(self.engine.click_mode != "erp")
        elif k == "a" and self.engine is not None:
            self.engine.set_assr(not self.engine.assr_on)
        elif k == "c":
            self.epochs.reset(); self.spectrum.reset()

    def set_sound(self, mode):
        if self.engine is not None:
            self.engine.set_mode(mode)

    def set_view(self, view):
        for ax in self.axes:
            ax.remove()
        self.axes = []
        self.view = view
        self.header.set_text(HEADERS[view])
        rows = {"overview": 0.80, "spectrogram": 0.85, "zoo": 0.85, "nod": 0.85, "erp": 0.84, "assr": 0.84}[view]
        self.gs = self.fig.add_gridspec(4, 2, left=0.06, right=0.76, top=rows, bottom=0.06, hspace=0.55, wspace=0.3)
        build = getattr(self, "build_" + view)
        self.update = build()
        if view != "erp" and self.engine is not None and self.engine.click_mode == "erp":
            self.engine.set_erp(False)
        if view != "assr" and self.engine is not None:
            self.engine.set_assr(False)
        if view != "nod" and self.engine is not None and self.metronome_on:
            self.metronome_on = False; self.engine.set_metronome(None)
        self.fig.canvas.draw_idle()

    def add(self, spec):
        ax = self.fig.add_subplot(spec); self.axes.append(ax); return ax

    # ------------------------------------------------------------------ shared pieces
    def raw_axes(self, spec, seconds=4, title=None):
        ax = self.add(spec)
        offsets = np.arange(4)[::-1] * 100
        t = np.arange(FS * seconds) / FS - seconds
        lines = [ax.plot(t, np.zeros_like(t) + o, lw=0.8)[0] for o in offsets]
        labels = [ax.text(-seconds + 0.05, o + 25, ch, fontsize=10, weight="bold") for ch, o in zip(CH, offsets)]
        ax.set_ylim(-80, 380); ax.set_xlim(-seconds, 0); ax.set_yticks([]); ax.set_xlabel("seconds")
        ax.set_title(title or "Raw EEG, 1–40 Hz (100 µV between traces; a red label means a poor contact)")
        _clean(ax, left=False)

        def upd():
            x = filtered_raw(seconds)
            if x.shape[1] < FS:
                return
            seg = x[:, -FS * seconds:]; tt = t[-seg.shape[1]:]
            sd = contact_quality()
            for c in range(4):
                lines[c].set_data(tt, np.clip(seg[c], -60, 60) + offsets[c])
                bad = sd[c] > BAD_CONTACT_UV or sd[c] < 0.5
                labels[c].set_color("#c33" if bad else "#000")
                labels[c].set_text(f"{CH[c]}  (check contact)" if bad else CH[c])
        return ax, upd

    def alpha_axes(self, spec):
        ax = self.add(spec)
        (line,) = ax.plot([], [], color="#c33", lw=1.5)
        ax.set_xlim(-60, 0); ax.set_ylim(-10, 40); ax.set_xlabel("seconds"); ax.set_ylabel("alpha power (dB)")
        ax.set_title("Alpha (8–13 Hz) over the last minute"); _clean(ax)

        def upd():
            if self.alpha_hist:
                now = time.monotonic() - self.t0
                tt = np.array([h[0] for h in self.alpha_hist]) - now
                line.set_data(tt, [h[1] for h in self.alpha_hist])
        return ax, upd

    # ------------------------------------------------------------------ views
    def build_overview(self):
        gs = self.gs
        _, u_raw = self.raw_axes(gs[0, :])
        ax_bar = self.add(gs[1, 0])
        bars = ax_bar.bar([b[0] for b in BANDS], np.zeros(4), color=["#888", "#888", "#c33", "#888"])
        ax_bar.set_ylim(-10, 40); ax_bar.set_ylabel("power (dB)"); ax_bar.set_title("Band power, last 2 s, mean of 4 channels"); _clean(ax_bar)
        _, u_alpha = self.alpha_axes(gs[1, 1])
        ax_imu = self.add(gs[2, :])
        t_imu = np.arange(FS_IMU * 4) / FS_IMU - 4
        imu_lines = [ax_imu.plot(t_imu, np.zeros_like(t_imu), lw=1, label=l)[0] for l in ("x", "y", "z")]
        ax_imu.set_xlim(-4, 0); ax_imu.set_ylim(-1.5, 1.5); ax_imu.set_ylabel("g"); ax_imu.set_xlabel("seconds")
        ax_imu.set_title("Head motion: accelerometer (tilt shifts the lines, a nod makes a bump)")
        ax_imu.legend(loc="upper left", ncol=3, frameon=False, fontsize=8); _clean(ax_imu)
        tilt_text = ax_imu.text(0.99, 0.95, "", transform=ax_imu.transAxes, ha="right", va="top", fontsize=9, color="#555")
        ax_opt, opt_lines, opt_note = None, [], None
        if self.athena:
            ax_opt = self.add(gs[3, :])
            t_opt = np.arange(FS_OPT * 30) / FS_OPT - 30
            opt_lines = [ax_opt.plot(t_opt, np.zeros_like(t_opt), lw=1, label=f"optode {c + 1}")[0] for c in range(4)]
            ax_opt.set_xlim(-30, 0); ax_opt.set_ylim(-5, 5); ax_opt.set_xlabel("seconds"); ax_opt.set_ylabel("% change")
            ax_opt.set_title("Athena fNIRS: near-infrared light through the forehead, % change over 30 s (slow: hold your breath for 20 s)")
            ax_opt.legend(loc="upper left", ncol=4, frameon=False, fontsize=8); _clean(ax_opt)
            opt_note = ax_opt.text(0.99, 0.95, "", transform=ax_opt.transAxes, ha="right", va="top", fontsize=9, color="#555")
        else:
            ax_note = self.add(gs[3, :]); ax_note.axis("off")
            ax_note.text(0, 0.6, "A Muse S Athena would add its near-infrared (fNIRS) channels here.", fontsize=9, color="#777")

        def upd():
            u_raw()
            bp = self.bp
            if np.all(np.isfinite(bp)):
                for b, v in zip(bars, bp): b.set_height(v)
            u_alpha()
            a = acc.last(4)
            if a.shape[1] > 2:
                ta = t_imu[-a.shape[1]:]
                for c in range(3): imu_lines[c].set_data(ta, a[c])
                tilt_text.set_text(f"tilt {self.tilt:4.0f}°   turning {self.speed:5.0f}°/s")
            if ax_opt is not None:
                o = optics.last(30)
                if o.shape[1] > FS_OPT * 2:
                    base = np.median(o, axis=1, keepdims=True)
                    pct = 100 * (o - base) / np.where(base == 0, 1, base)
                    to = t_opt[-o.shape[1]:]
                    for c in range(4): opt_lines[c].set_data(to, pct[c])
                    opt_note.set_text(f"{optics.count / FS_OPT:5.1f} s of optics at {FS_OPT} Hz")
                else:
                    opt_note.set_text("waiting for optics data")
        return upd

    def build_spectrogram(self):
        gs = self.gs
        _, u_raw = self.raw_axes(gs[0, :])
        ax = self.add(gs[1:3, :])
        img = ax.imshow(np.zeros((41, 80)), aspect="auto", origin="lower", extent=[-20, 0, 0, 40], vmin=-10, vmax=30, cmap="magma")
        ax.set_xlabel("seconds"); ax.set_ylabel("frequency (Hz)")
        ax.axhline(8, color="w", lw=0.5, ls=":"); ax.axhline(13, color="w", lw=0.5, ls=":")
        ax.text(-19.8, 10.5, "alpha", color="w", fontsize=9, va="center")
        title = ax.set_title("Spectrogram of the best channel, last 20 s, 0–40 Hz (1 s window, 0.25 s step, dB)")
        _, u_alpha = self.alpha_axes(gs[3, :])
        state = {"last": 0}

        def upd():
            u_raw(); u_alpha()
            if eeg.count - state["last"] >= FS // 4:
                state["last"] = eeg.count
                ch = best_channel()
                r = eeg_spectrogram(ch, 20)
                if r is not None:
                    f, t, S = r
                    img.set_data(S); img.set_extent([t[0], 0, f[0], f[-1]])
                    title.set_text(f"Spectrogram of {CH[ch]} (best contact), last 20 s, 0–40 Hz (1 s window, 0.25 s step, dB)")
        return upd

    def zoo_prompt(self):
        if self.zoo_start is None:
            self.zoo_start = time.monotonic()
        total = sum(d for _, d in ZOO_PROMPTS)
        tt = (time.monotonic() - self.zoo_start) % total
        acc_t = 0
        for label, d in ZOO_PROMPTS:
            if tt < acc_t + d:
                return label, acc_t + d - tt
            acc_t += d
        return ZOO_PROMPTS[0][0], 0

    def build_zoo(self):
        gs = self.gs
        self.zoo_start = time.monotonic()
        ax_p = self.add(gs[0, :]); ax_p.axis("off")
        prompt = ax_p.text(0.5, 0.5, "", ha="center", va="center", fontsize=22, weight="bold")
        count = ax_p.text(0.98, 0.1, "", ha="right", va="bottom", fontsize=10, color="#777")
        ax, u_raw = self.raw_axes(gs[1:3, :], seconds=8, title="Raw EEG, 1–40 Hz, last 8 s")
        marks = []
        ax_g = self.add(gs[3, :])
        t_g = np.arange(FS_IMU * 8) / FS_IMU - 8
        (gline,) = ax_g.plot(t_g, np.zeros_like(t_g), lw=1, color="#555")
        ax_g.set_xlim(-8, 0); ax_g.set_ylim(0, 300); ax_g.set_ylabel("°/s"); ax_g.set_xlabel("seconds")
        ax_g.set_title("Head rotation speed from the gyroscope (nods and head turns)"); _clean(ax_g)

        def upd():
            label, left = self.zoo_prompt()
            prompt.set_text(label); count.set_text(f"{left:3.0f} s")
            u_raw()
            for m in marks: m.remove()
            marks.clear()
            now = time.monotonic()
            for kind, t in self.events[-30:]:
                rel = t - now
                if rel > -8:
                    marks.append(ax.annotate(kind, (rel, 360), ha="center", fontsize=8, color="#c33" if kind == "blink" else "#36c"))
            g = gyro.last(8)
            if g.shape[1] > 2:
                gline.set_data(t_g[-g.shape[1]:], np.linalg.norm(g, axis=0))
        return upd

    def build_nod(self):
        gs = self.gs
        ax_g = self.add(gs[0:2, :])
        t_g = np.arange(FS_IMU * 8) / FS_IMU - 8
        (gline,) = ax_g.plot(t_g, np.zeros_like(t_g), lw=1, color="#555")
        ax_g.set_xlim(-8, 0); ax_g.set_ylim(0, 400); ax_g.set_ylabel("°/s"); ax_g.set_xlabel("seconds")
        ax_g.set_title("Rotation speed of the head; each detected nod is a tick, each click a dashed line"); _clean(ax_g)
        info = ax_g.text(0.99, 0.95, "", transform=ax_g.transAxes, ha="right", va="top", fontsize=10)
        ticks = []
        ax_h = self.add(gs[2:4, :])
        ax_h.set_xlim(-300, 300); ax_h.set_xlabel("nod minus nearest click (ms); negative is early"); ax_h.set_ylabel("count")
        ax_h.set_title("Asynchrony of the last 20 nods"); _clean(ax_h)
        ax_h.axvline(0, color="#999", lw=1)
        hist_art = {"bars": None}
        mtext = ax_h.text(0.99, 0.95, "", transform=ax_h.transAxes, ha="right", va="top", fontsize=10)

        def upd():
            g = gyro.last(8)
            now = time.monotonic()
            if g.shape[1] > 2:
                gline.set_data(t_g[-g.shape[1]:], np.linalg.norm(g, axis=0))
            for a in ticks: a.remove()
            ticks.clear()
            for t in self.nods.times[-12:]:
                if now - t < 8:
                    ticks.append(ax_g.axvline(t - now, color="#c33", lw=1.5, ymax=0.15))
            if self.engine is not None:
                for t in self.engine.recent_clicks(now - 8, now):
                    ticks.append(ax_g.axvline(t - now, color="#36c", lw=1, ls="--", ymax=0.6))
            per = self.nods.period()
            state = "metronome on" if self.metronome_on else "press m to start the metronome"
            info.set_text(f"nod period {per:.2f} s = {60 / per:.0f} BPM   {state}" if per else f"nod steadily to set a tempo   {state}")
            if self.metronome_on and per and self.engine is not None:
                if self.last_period is None or abs(per - self.last_period) / self.last_period > 0.05:
                    self.engine.set_metronome(per); self.last_period = per
            if self.engine is not None and self.engine.click_log and len(self.nods.times) > 1:
                clicks = np.array(self.engine.click_log)
                diffs = []
                for t in self.nods.times[-20:]:
                    j = np.argmin(np.abs(clicks - t))
                    d = (t - clicks[j]) * 1000
                    if abs(d) < 300:
                        diffs.append(d)
                if hist_art["bars"] is not None:
                    hist_art["bars"].remove(); hist_art["bars"] = None
                if diffs:
                    _, _, hist_art["bars"] = ax_h.hist(diffs, bins=np.arange(-300, 301, 25), color="#c33", alpha=0.8)
                    ax_h.set_ylim(0, max(3, max(np.histogram(diffs, bins=np.arange(-300, 301, 25))[0]) + 1))
                    mtext.set_text(f"mean {np.mean(diffs):+.0f} ms, sd {np.std(diffs):.0f} ms, n = {len(diffs)}")
        return upd

    def build_erp(self):
        gs = self.gs
        ax = self.add(gs[0:3, :])
        lines = [ax.plot([], [], lw=1.2, label=ch)[0] for ch in CH]
        ax.set_xlim(-100, 500); ax.set_ylim(-15, 15); ax.set_xlabel("milliseconds after the click"); ax.set_ylabel("µV")
        ax.axvline(0, color="#999", lw=1); ax.axhline(0, color="#ccc", lw=0.5)
        ax.axvspan(80, 130, color="#c33", alpha=0.08); ax.text(105, 13.5, "N1", ha="center", color="#c33", fontsize=9)
        ax.axvspan(160, 240, color="#36c", alpha=0.08); ax.text(200, 13.5, "P2", ha="center", color="#36c", fontsize=9)
        ax.legend(loc="lower right", ncol=4, frameon=False, fontsize=9); _clean(ax)
        title = ax.set_title("Average EEG around each click")
        note = ax.text(0.01, 0.97, "", transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#555")
        _, u_raw = self.raw_axes(gs[3, :], seconds=4, title="Raw EEG (sit still, blink between clicks if you must)")
        if self.engine is not None:
            self.engine.set_erp(True)

        def upd():
            lags, m, n = self.epochs.mean()
            for c in range(4):
                lines[c].set_data(lags * 1000, m[c])
            if n.max() > 0:
                lim = max(8.0, min(60.0, 1.3 * np.nanmax(np.abs(m))))
                ax.set_ylim(-lim, lim)
            on = self.engine is not None and self.engine.click_mode == "erp"
            title.set_text(f"Average EEG around each click   ({'clicks on, key e stops them' if on else 'press e to start the clicks'})")
            note.set_text("epochs kept per channel: " + ", ".join(f"{CH[c]} {n[c]}" for c in range(4))
                          + f"   rejected {self.epochs.rejected} (peak-to-peak over {self.epochs.reject_uv:.0f} µV)"
                          + ("   keep still, relax the jaw, improve the contacts" if self.epochs.rejected > 2 * max(1, n.max()) else ""))
            u_raw()
        return upd

    def build_assr(self):
        gs = self.gs
        ax = self.add(gs[0:3, :])
        (line,) = ax.plot([], [], lw=1.5, color="#333")
        ax.set_xlim(20, 60); ax.set_ylim(-20, 20); ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("power (dB)")
        ax.axvline(40, color="#c33", lw=1, ls="--"); _clean(ax)
        from .analysis import MAINS_HZ
        ax.axvspan(MAINS_HZ - 1.5, MAINS_HZ + 1.5, color="#999", alpha=0.15)
        ax.text(MAINS_HZ, 0.02, f"mains {MAINS_HZ:.0f} Hz, notched", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color="#777")
        title = ax.set_title("Averaged EEG spectrum, 20–60 Hz")
        snr = ax.text(0.99, 0.95, "", transform=ax.transAxes, ha="right", va="top", fontsize=11)
        note = ax.text(0.01, 0.97, "", transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#555")
        _, u_raw = self.raw_axes(gs[3, :], seconds=4, title="Raw EEG")
        if self.engine is not None:
            self.engine.set_assr(True)

        def upd():
            r = self.spectrum.result()
            on = self.engine is not None and self.engine.assr_on
            if r is not None:
                f, p, n, s = r
                line.set_data(f, p); ax.set_ylim(*self.spectrum.ylim())
                snr.set_text(f"40 Hz against neighbours: {s:+.1f} dB")
                note.set_text("windows kept per channel: " + ", ".join(f"{CH[c]} {n[c]}" for c in range(4))
                              + f"   rejected {self.spectrum.rejected}")
            else:
                note.set_text(f"no window kept yet   rejected {self.spectrum.rejected} (peak-to-peak over {self.spectrum.reject_uv:.0f} µV): keep still, improve the contacts")
            title.set_text(f"Averaged EEG spectrum, 20–60 Hz   ({'tone on, key a stops it' if on else 'press a to start the 40 Hz tone'})")
            u_raw()
        return upd

    # ------------------------------------------------------------------ the tick
    def tick(self):
        self.source.pump(0.05)
        now = time.monotonic() - self.t0
        self.bp = band_powers()
        if np.all(np.isfinite(self.bp)):
            self.alpha_hist.append((now, self.bp[2])); self.band_hist.append((now, self.bp))
            self.alpha_hist = [h for h in self.alpha_hist if h[0] > now - 60]
            self.band_hist = self.band_hist[-1200:]
            if self.recorder is not None:
                self.recorder.log_bands(self.bp)
        self.tilt, self.speed = self.pose.read()
        for kind, idx in self.face.update():
            self.events.append((kind, eeg.time_of(idx)))
            if self.engine is not None and self.engine.mode == "drum":
                self.engine.trigger("kick" if kind == "blink" else "snare")
        self.events = self.events[-200:]
        self.nods.update()
        eng = self.engine
        if eng is not None:
            if self.alpha_hist:
                a = [h[1] for h in self.alpha_hist]; eng.set_alpha(self.bp[2], min(a), max(a))
            eng.set_head(self.tilt, self.speed)
            if eng.mode == "pan":
                pc = band_powers(per_channel=True)
                if np.all(np.isfinite(pc)):
                    l, r = pc[2, [0, 1]].mean(), pc[2, [2, 3]].mean()
                    eng.set_pan(0.5 + np.clip((r - l) / 10, -0.5, 0.5))
            if eng.mode == "brainmix" and len(self.band_hist) > 5:
                hist = np.array([h[1] for h in self.band_hist[-240:]])
                lo, hi = hist.min(axis=0), hist.max(axis=0)
                eng.set_bands((self.bp - lo) / np.maximum(hi - lo, 3))
            for t in eng.pop_new_clicks():
                if eng.click_mode == "erp":
                    self.epochs.add(t)
            self.epochs.update()
            if eng.assr_on:
                self.spectrum.update()
        if self.recorder is not None:
            self.recorder.update()
        self.update()
        src = "synthetic signal" if self.fake else ("Muse S Athena" if self.athena else "Muse")
        rec = f"   recording to {self.recorder.dir} ({self.recorder.markers} markers)" if self.recorder is not None and self.recorder.active else "   r to record"
        self.status.set_text(f"{src}   EEG {eeg.count / FS:6.1f} s   motion {acc.count / FS_IMU:6.1f} s{rec}   {time.strftime('%H:%M:%S')}")
        self.fig.canvas.draw_idle()

    def run(self):
        timer = self.fig.canvas.new_timer(interval=60)
        timer.add_callback(self.tick)
        timer.start()
        self.plt.show()
