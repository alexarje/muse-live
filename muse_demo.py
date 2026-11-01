#!/usr/bin/env python
"""Live signals from a Muse headband for the classroom: EEG, head motion, band powers, alpha over time, and sound.

Works with the Muse 2016, Muse 2 and Muse S (one protocol) and the Muse S Athena (another): the script
probes the headband after connecting and picks the right decoder. It talks to the device directly over
Bluetooth LE (bleak), no browser, no LSL.

    .venv/bin/python muse_demo.py                    # scan, connect to the first Muse, open the window
    .venv/bin/python muse_demo.py --address AA:BB:CC:DD:EE:FF
    .venv/bin/python muse_demo.py --sound pitch      # start with a sound mode on (pitch, am, head)
    .venv/bin/python muse_demo.py --fake             # synthetic signals, for testing without a headband
    .venv/bin/python muse_demo.py --test 8           # connect, capture 8 s, print statistics, no window
    .venv/bin/python muse_demo.py --fake --save frame.png   # render one frame to a file

In the window: the sound buttons at the bottom right switch the sonification; the keys 0, 1, 2, 3 do the same.

The headband must be switched on and must not be paired in the system Bluetooth settings,
nor connected to the Muse phone app.
"""
import argparse
import sys
import time

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

FS = 256                     # EEG sampling rate, Hz
FS_IMU = 52                  # accelerometer and gyroscope rate, Hz
FS_OPT = 64                  # Athena near-infrared (optics) rate, Hz
CH = ["TP9", "AF7", "AF8", "TP10"]   # left ear, left forehead, right forehead, right ear
BANDS = [("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30)]
RAW_SECONDS = 4              # window of raw traces shown
PSD_SECONDS = 2              # window for the band-power estimate
HISTORY_SECONDS = 60         # alpha history shown
BAD_CONTACT_UV = 150         # standard deviation above which a channel is flagged
AUDIO_FS = 44100
SOUND_MODES = [None, "pitch", "am", "head"]
SOUND_LABELS = ["off", "1  pitch follows alpha", "2  tremolo of the raw EEG", "3  head tilt and motion"]

EXPLANATION = (
    "What you see: the electrical activity of the brain picked up by four dry electrodes, TP9 and TP10 behind the ears and "
    "AF7 and AF8 on the forehead, in microvolts,\n"
    "filtered to 1–40 Hz. Blinks, eye movements and jaw tension make the biggest waves (artefacts); a red label means an "
    "electrode is not touching the skin.\n"
    "Band power splits the last two seconds into the classic rhythms. Alpha (8–13 Hz) grows when the eyes close and the visual "
    "cortex idles. Try it: eyes closed for 15 s, then open,\n"
    "and watch the red curve step down. The headband also carries an accelerometer and a gyroscope: tilting the head shifts "
    "the three lines, a nod makes a bump.\n"
    "Sound: 1 is a tone whose pitch follows alpha (220–880 Hz); 2 is a 200 Hz tone whose loudness follows the raw EEG, so alpha "
    "is a tremolo and a blink a thump;\n"
    "3 is a tone whose pitch follows how far the head is tilted and whose loudness follows how fast it moves."
)
EXPLANATION_ATHENA = (
    "\nThis is a Muse S Athena: the bottom row adds its five near-infrared light sensors (fNIRS), which read blood volume and "
    "oxygenation in the frontal cortex. That signal changes over seconds, not milliseconds."
)


class Ring:
    """A ring buffer of the last `seconds` of a multichannel signal."""

    def __init__(self, channels, fs, seconds):
        self.fs = fs
        self.n = fs * seconds
        self.buf = np.zeros((channels, self.n))
        self.i = 0
        self.count = 0           # total samples ever written (absolute index of the next sample)

    def push(self, block):            # block: (channels, k)
        k = block.shape[1]
        idx = (self.i + np.arange(k)) % self.n
        self.buf[:, idx] = block
        self.i = (self.i + k) % self.n
        self.count += k

    def last(self, seconds):
        m = int(min(self.fs * seconds, self.count, self.n))
        idx = (self.i - m + np.arange(m)) % self.n
        return self.buf[:, idx]

    def at(self, ch, abs_idx):
        """Samples of one channel at absolute indices (held at the newest sample if asked for the future)."""
        abs_idx = np.minimum(abs_idx, self.count - 1)
        return self.buf[ch, abs_idx % self.n]


eeg = Ring(4, FS, HISTORY_SECONDS)
acc = Ring(3, FS_IMU, HISTORY_SECONDS)      # g
gyro = Ring(3, FS_IMU, HISTORY_SECONDS)     # degrees per second
optics = Ring(4, FS_OPT, HISTORY_SECONDS)   # Athena only: raw near-infrared light readings


def on_eeg(data, timestamps):
    """(channels, samples) in microvolts; the first four channels are TP9, AF7, AF8, TP10."""
    eeg.push(np.asarray(data)[:4, :])


def on_acc(data, timestamps):
    acc.push(np.asarray(data)[:3, :])


def on_gyro(data, timestamps):
    gyro.push(np.asarray(data)[:3, :])


def on_optics(data, timestamps):
    d = np.asarray(data)
    if d.ndim == 2 and d.shape[0] >= 4:
        optics.push(d[:4, :])


# ----------------------------------------------------------------------------- sources
class FakeMuse:
    """Synthetic signals: EEG noise with alpha on for 10 s and off for 10 s and a blink every 7 s;
    a head that tilts slowly and nods every 9 s."""

    def __init__(self, athena=False):
        self.t = 0.0
        self.athena = athena
        self.rng = np.random.default_rng(0)

    def pump(self, seconds):
        k = int(seconds * FS)
        t = self.t + np.arange(k) / FS
        alpha_gain = 15 * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * t / 20)))
        blink = 120 * np.exp(-((t % 7 - 3.5) ** 2) / (2 * 0.08 ** 2))
        block = np.empty((4, k))
        for c in range(4):
            drift = np.cumsum(self.rng.normal(0, 1, k)); drift -= np.linspace(drift[0], drift[-1], k)
            block[c] = 8 * self.rng.normal(0, 1, k) + 0.3 * drift + alpha_gain * np.sin(2 * np.pi * 10 * t + c)
            if c in (1, 2):
                block[c] += blink
        on_eeg(np.vstack([block, np.zeros(k)]), t)
        ki = max(1, int(round(seconds * FS_IMU)))
        ti = self.t + np.arange(ki) / FS_IMU
        nod = 0.25 * np.exp(-((ti % 9 - 4.5) ** 2) / (2 * 0.15 ** 2))
        tilt = 0.3 * np.sin(2 * np.pi * ti / 16)
        a = np.vstack([tilt + nod, 0.05 * np.sin(2 * np.pi * ti / 11), np.sqrt(np.clip(1 - tilt ** 2, 0, 1))])
        a += 0.01 * self.rng.normal(0, 1, a.shape)
        g = np.vstack([np.gradient(a[0]) * FS_IMU * 57.3, np.gradient(a[1]) * FS_IMU * 57.3, np.zeros(ki)])
        g += 2 * self.rng.normal(0, 1, g.shape)
        on_acc(a, ti); on_gyro(g, ti)
        if self.athena:
            ko = max(1, int(round(seconds * FS_OPT)))
            to = self.t + np.arange(ko) / FS_OPT
            slow = 1 + 0.02 * np.sin(2 * np.pi * to / 25) + 0.004 * np.sin(2 * np.pi * 1.1 * to)   # breath-hold-like swell and a pulse ripple
            o = np.vstack([slow * (40000 + 3000 * c) for c in range(4)]) + 30 * self.rng.normal(0, 1, (4, ko))
            on_optics(o, to)
        self.t += k / FS
        time.sleep(seconds)


def connect_muse(address, model="auto"):
    """Connect and start streaming. Returns (device, is_athena). device.adapter.pump() must be called often.

    muselsl's own auto mode subscribes to the old EEG characteristics before it probes, which fails on an Athena,
    so the probe is done here with a bare connection and no callbacks."""
    from muselsl.muse import Muse
    from muselsl.athena import Athena
    from muselsl.devices import probe_athena
    from muselsl import backends
    if address is None:
        print("Scanning for a Muse (12 s)...")
        found = [d for d in backends.BleakBackend().scan(timeout=12) if d["name"] and "muse" in d["name"].lower()]
        if not found:
            sys.exit("No Muse found. Is it switched on, unpaired from the system, and not in the phone app?")
        address = found[0]["address"]
        print(f"Found {found[0]['name']} at {address}")
    if model == "auto":
        print(f"Probing {address} (the first connection can take 20 s)...")
        probe = Muse(address, backend="bleak")
        if not probe.connect(retries=3):
            sys.exit("Could not connect. Turn the headband off and on, and try again.")
        model = "athena" if probe_athena(probe.device) else "legacy"
        probe.disconnect()
    cls = Athena if model == "athena" else Muse
    kwargs = dict(callback_eeg=on_eeg, callback_acc=on_acc, callback_gyro=on_gyro, backend="bleak")
    if model == "athena":
        kwargs["callback_optics"] = on_optics
    muse = cls(address, **kwargs)
    print(f"Connecting as {'Muse S Athena' if model == 'athena' else 'Muse (2016, 2 or S)'}...")
    if not muse.connect(retries=3):
        sys.exit("Could not connect. Turn the headband off and on, and try again.")
    muse.start()
    print("Streaming. Close the window or press Ctrl-C to stop.")
    return muse, model == "athena"


# ----------------------------------------------------------------------------- analysis
SOS = butter(4, [1, 40], btype="band", fs=FS, output="sos")


def filtered_raw():
    x = eeg.last(RAW_SECONDS)
    if x.shape[1] < FS:
        return x
    x = x - x.mean(axis=1, keepdims=True)
    return sosfiltfilt(SOS, x, axis=1)


def band_powers():
    """Mean band power over the four channels in dB, from the last PSD_SECONDS."""
    x = eeg.last(PSD_SECONDS)
    if x.shape[1] < FS:
        return np.full(len(BANDS), np.nan)
    x = x - x.mean(axis=1, keepdims=True)
    f, p = welch(x, fs=FS, nperseg=FS, axis=1)
    out = []
    for _, lo, hi in BANDS:
        sel = (f >= lo) & (f < hi)
        out.append(10 * np.log10(p[:, sel].mean(axis=1) + 1e-12))
    return np.array(out).mean(axis=1)


def best_channel():
    """The channel with the smallest spread in the last second, which is the best contact."""
    x = eeg.last(1)
    if x.shape[1] < FS // 2:
        return 1
    return int(np.argmin(x.std(axis=1)))


class HeadPose:
    """Tilt in degrees away from the orientation seen in the first second, and rotation speed in degrees per second."""

    def __init__(self):
        self.ref = None

    def read(self):
        a = acc.last(0.25)
        if a.shape[1] < 3:
            return 0.0, 0.0
        v = a.mean(axis=1)
        n = np.linalg.norm(v)
        if n < 0.3:
            return 0.0, 0.0
        v = v / n
        if self.ref is None:
            if acc.count > FS_IMU:
                self.ref = v
            return 0.0, 0.0
        tilt = float(np.degrees(np.arccos(np.clip(np.dot(v, self.ref), -1, 1))))
        g = gyro.last(0.25)
        speed = float(np.linalg.norm(g, axis=0).mean()) if g.shape[1] else 0.0
        return tilt, speed


# ----------------------------------------------------------------------------- sound
class Sonifier:
    """Three mappings. `pitch`: a sine whose pitch follows alpha power, 220 Hz for the lowest alpha seen in the last
    minute and 880 Hz for the highest. `am`: a 200 Hz tone whose loudness follows the raw EEG of the best channel, so
    the alpha rhythm is heard as a 10 Hz tremolo and a blink as a thump. `head`: a sine whose pitch follows head tilt
    (0 to 60 degrees over 220 to 880 Hz) and whose loudness follows how fast the head turns."""

    def __init__(self, mode):
        import sounddevice as sd
        self.mode = mode
        self.phase = 0.0
        self.freq = 440.0
        self.target = 440.0
        self.amp = 0.2
        self.amp_target = 0.2
        self.pos = None          # absolute EEG read position (float), am mode
        self.dc = 0.0
        self.level = 20.0
        self.stream = sd.OutputStream(samplerate=AUDIO_FS, channels=1, blocksize=1024, callback=self._callback)
        self.stream.start()

    def set_mode(self, mode):
        self.mode = mode
        self.pos = None

    def set_alpha(self, alpha_db, lo, hi):
        if self.mode != "pitch":
            return
        if hi - lo < 3:
            mid = 0.5 * (lo + hi); lo, hi = mid - 1.5, mid + 1.5
        x = float(np.clip((alpha_db - lo) / (hi - lo), 0, 1))
        self.target = 220.0 * 2 ** (2 * x)
        self.amp_target = 0.2

    def set_head(self, tilt_deg, speed_dps):
        if self.mode != "head":
            return
        x = float(np.clip(tilt_deg / 60.0, 0, 1))
        self.target = 220.0 * 2 ** (2 * x)
        self.amp_target = 0.04 + 0.3 * float(np.clip(speed_dps / 150.0, 0, 1))

    def _callback(self, out, frames, t, status):
        n = np.arange(1, frames + 1)
        if self.mode in ("pitch", "head"):
            self.freq += (self.target - self.freq) * 0.15
            self.amp += (self.amp_target - self.amp) * 0.2
            ph = self.phase + 2 * np.pi * self.freq * n / AUDIO_FS
            out[:, 0] = self.amp * np.sin(ph)
            self.phase = ph[-1] % (2 * np.pi)
            return
        if eeg.count < FS // 4:
            out[:] = 0; return
        step = FS / AUDIO_FS
        if self.pos is None or eeg.count - self.pos > FS * 0.5 or self.pos > eeg.count:
            self.pos = eeg.count - FS * 0.1
        idx = self.pos + np.arange(frames) * step
        i0 = np.floor(idx).astype(int)
        frac = idx - i0
        ch = best_channel()
        e = eeg.at(ch, i0) * (1 - frac) + eeg.at(ch, i0 + 1) * frac
        self.pos = idx[-1] + step
        self.dc += (e.mean() - self.dc) * 0.05
        e = e - self.dc
        self.level += (np.abs(e).mean() + 1e-6 - self.level) * 0.02
        env = np.clip(e / (4 * self.level), -1, 1)
        ph = self.phase + 2 * np.pi * 200.0 * n / AUDIO_FS
        out[:, 0] = 0.3 * (0.5 + 0.5 * env) * np.sin(ph)
        self.phase = ph[-1] % (2 * np.pi)

    def close(self):
        try:
            self.stream.stop(); self.stream.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------- window
def build_window(source, fake, son=None, athena=False):
    """Create the figure and return it with the function that updates it."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "toolbar": "None"})
    fig = plt.figure(figsize=(12, 12.5 if athena else 10.5))
    try:
        fig.canvas.manager.set_window_title("Muse live")
    except Exception:
        pass
    fig.text(0.5, 0.98, "Muse live: brain and head", ha="center", va="top", fontsize=15, weight="bold")
    fig.text(0.03, 0.945, EXPLANATION + (EXPLANATION_ATHENA if athena else ""), ha="left", va="top", fontsize=9.2, linespacing=1.4,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#f3f3f3", edgecolor="none"))
    rows = [1.3, 1, 0.8] + ([0.9] if athena else [])
    gs = fig.add_gridspec(len(rows), 2, height_ratios=rows, hspace=0.5, wspace=0.25, top=0.82 if athena else 0.79, bottom=0.05)
    ax_raw = fig.add_subplot(gs[0, :])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[1, 1])
    ax_imu = fig.add_subplot(gs[2, 0])
    ax_ctl = fig.add_subplot(gs[2, 1])
    ax_opt = fig.add_subplot(gs[3, :]) if athena else None

    offsets = np.arange(4)[::-1] * 100
    t_raw = np.arange(FS * RAW_SECONDS) / FS - RAW_SECONDS
    lines = [ax_raw.plot(t_raw, np.zeros_like(t_raw) + o, lw=0.8)[0] for o in offsets]
    labels = [ax_raw.text(-RAW_SECONDS + 0.05, o + 25, ch, fontsize=10, weight="bold") for ch, o in zip(CH, offsets)]
    ax_raw.set_ylim(-80, 380); ax_raw.set_xlim(-RAW_SECONDS, 0)
    ax_raw.set_yticks([]); ax_raw.set_xlabel("seconds")
    ax_raw.set_title("Raw EEG, 1–40 Hz (100 µV between traces; a red label means a poor contact)")
    for s in ("top", "right", "left"): ax_raw.spines[s].set_visible(False)

    bars = ax_bar.bar([b[0] for b in BANDS], np.zeros(len(BANDS)), color=["#888", "#888", "#c33", "#888"])
    ax_bar.set_ylim(-10, 40); ax_bar.set_ylabel("power (dB)"); ax_bar.set_title("Band power, last 2 s, mean of 4 channels")
    for s in ("top", "right"): ax_bar.spines[s].set_visible(False)

    hist_t, hist_a = [], []
    (hist_line,) = ax_hist.plot([], [], color="#c33", lw=1.5)
    ax_hist.set_xlim(-HISTORY_SECONDS, 0); ax_hist.set_ylim(-10, 40)
    ax_hist.set_xlabel("seconds"); ax_hist.set_ylabel("alpha power (dB)")
    ax_hist.set_title("Alpha (8–13 Hz) over the last minute: close your eyes and watch it rise")
    for s in ("top", "right"): ax_hist.spines[s].set_visible(False)

    t_imu = np.arange(FS_IMU * RAW_SECONDS) / FS_IMU - RAW_SECONDS
    imu_lines = [ax_imu.plot(t_imu, np.zeros_like(t_imu), lw=1, label=l)[0] for l in ("x", "y", "z")]
    ax_imu.set_xlim(-RAW_SECONDS, 0); ax_imu.set_ylim(-1.5, 1.5); ax_imu.set_xlabel("seconds"); ax_imu.set_ylabel("g")
    ax_imu.set_title("Head motion: accelerometer (tilt shifts the lines, a nod makes a bump)")
    ax_imu.legend(loc="upper left", ncol=3, frameon=False, fontsize=8)
    for s in ("top", "right"): ax_imu.spines[s].set_visible(False)
    tilt_text = ax_imu.text(0.99, 0.95, "", transform=ax_imu.transAxes, ha="right", va="top", fontsize=9, color="#555")

    opt_lines, opt_note = [], None
    if ax_opt is not None:
        t_opt = np.arange(FS_OPT * 30) / FS_OPT - 30
        opt_lines = [ax_opt.plot(t_opt, np.zeros_like(t_opt), lw=1, label=f"optode {c + 1}")[0] for c in range(4)]
        ax_opt.set_xlim(-30, 0); ax_opt.set_ylim(-5, 5); ax_opt.set_xlabel("seconds"); ax_opt.set_ylabel("% change")
        ax_opt.set_title("Athena fNIRS: near-infrared light through the forehead, % change from the last 30 s "
                         "(blood volume and oxygen; slow, so hold your breath for 20 s or do mental arithmetic)")
        ax_opt.legend(loc="upper left", ncol=4, frameon=False, fontsize=8)
        for s in ("top", "right"): ax_opt.spines[s].set_visible(False)
        opt_note = ax_opt.text(0.99, 0.95, "", transform=ax_opt.transAxes, ha="right", va="top", fontsize=9, color="#555")

    ax_ctl.set_title("Sound (click, or press the key)")
    ax_ctl.set_facecolor("#f3f3f3")
    for s in ax_ctl.spines.values(): s.set_visible(False)
    radio = RadioButtons(ax_ctl, SOUND_LABELS, active=SOUND_MODES.index(getattr(son, "mode", None)))
    for lbl in radio.labels: lbl.set_fontsize(10)

    status = fig.text(0.01, 0.005, "", fontsize=9, color="#555")
    t0 = time.monotonic()
    state = {"son": son}
    pose = HeadPose()

    def select_mode(mode):
        if mode is None:
            if state["son"] is not None:
                state["son"].close(); state["son"] = None
        elif state["son"] is None:
            state["son"] = Sonifier(mode)
        else:
            state["son"].set_mode(mode)

    def on_radio(label):
        select_mode(SOUND_MODES[SOUND_LABELS.index(label)])

    def on_key(event):
        if event.key in ("0", "1", "2", "3"):
            radio.set_active(int(event.key))     # triggers on_radio

    radio.on_clicked(on_radio)
    fig.canvas.mpl_connect("key_press_event", on_key)

    def tick():
        source.pump(0.05)
        x = filtered_raw()
        if x.shape[1] >= FS:
            seg = x[:, -FS * RAW_SECONDS:]
            tt = t_raw[-seg.shape[1]:]
            sd = eeg.last(1).std(axis=1)
            for c in range(4):
                lines[c].set_data(tt, np.clip(seg[c], -60, 60) + offsets[c])
                bad = sd[c] > BAD_CONTACT_UV or sd[c] < 0.5
                labels[c].set_color("#c33" if bad else "#000")
                labels[c].set_text(f"{CH[c]}  (check contact)" if bad else CH[c])
        bp = band_powers()
        if np.all(np.isfinite(bp)):
            for b, v in zip(bars, bp):
                b.set_height(v)
            now = time.monotonic() - t0
            hist_t.append(now); hist_a.append(bp[2])
            while hist_t and hist_t[0] < now - HISTORY_SECONDS:
                hist_t.pop(0); hist_a.pop(0)
            hist_line.set_data(np.array(hist_t) - now, hist_a)
            if state["son"] is not None:
                state["son"].set_alpha(bp[2], min(hist_a), max(hist_a))
        a = acc.last(RAW_SECONDS)
        if a.shape[1] > 2:
            ta = t_imu[-a.shape[1]:]
            for c in range(3):
                imu_lines[c].set_data(ta, a[c])
            tilt, speed = pose.read()
            tilt_text.set_text(f"tilt {tilt:4.0f}°   turning {speed:5.0f}°/s")
            if state["son"] is not None:
                state["son"].set_head(tilt, speed)
        if ax_opt is not None:
            o = optics.last(30)
            if o.shape[1] > FS_OPT * 2:
                base = np.median(o, axis=1, keepdims=True)
                pct = 100 * (o - base) / np.where(base == 0, 1, base)
                to = t_opt[-o.shape[1]:]
                for c in range(4):
                    opt_lines[c].set_data(to, pct[c])
                opt_note.set_text(f"{optics.count / FS_OPT:5.1f} s of optics at {FS_OPT} Hz")
            elif opt_note is not None:
                opt_note.set_text("waiting for optics data" + (" (synthetic)" if fake else ""))
        src = "synthetic signal" if fake else ("Muse S Athena" if athena else "Muse")
        status.set_text(f"{src}   EEG {eeg.count / FS:6.1f} s   motion {acc.count / FS_IMU:6.1f} s   {time.strftime('%H:%M:%S')}")
        fig.canvas.draw_idle()

    tick.state = state
    return fig, tick


def run_window(source, fake, son, athena=False):
    import matplotlib.pyplot as plt
    fig, tick = build_window(source, fake, son, athena)
    timer = fig.canvas.new_timer(interval=60)
    timer.add_callback(tick)
    timer.start()
    plt.show()
    if tick.state["son"] is not None:
        tick.state["son"].close()


def save_frame(source, fake, path, athena=False):
    """Render one frame without a display, after pumping for 12 s."""
    import matplotlib
    matplotlib.use("Agg")
    for _ in range(120):
        source.pump(0.1)
    fig, tick = build_window(source, fake, None, athena)
    for _ in range(3):
        tick()
    fig.savefig(path, dpi=110)
    print("saved", path)


def run_test(source, seconds, son):
    print(f"Capturing {seconds:.0f} s...")
    end = time.monotonic() + seconds
    pose = HeadPose()
    while time.monotonic() < end:
        source.pump(0.1)
        if son is not None:
            bp = band_powers()
            if np.isfinite(bp[2]):
                son.set_alpha(bp[2], 0.0, 20.0)
            son.set_head(*pose.read())
    x = eeg.last(seconds)
    print(f"EEG samples per channel: {x.shape[1]} (expected about {int(FS * seconds)})")
    for c in range(4):
        print(f"  {CH[c]:5} mean {x[c].mean():8.1f} µV   sd {x[c].std():7.1f} µV")
    bp = band_powers()
    print("band power (dB): " + ", ".join(f"{b[0]} {v:.1f}" for b, v in zip(BANDS, bp)))
    a = acc.last(seconds); g = gyro.last(seconds)
    print(f"accelerometer samples: {a.shape[1]} (expected about {int(FS_IMU * seconds)}); mean g per axis "
          + ", ".join(f"{v:.2f}" for v in a.mean(axis=1)) if a.shape[1] else "accelerometer: no samples")
    print(f"gyroscope samples: {g.shape[1]}; mean |rotation| {np.linalg.norm(g, axis=0).mean():.1f} °/s" if g.shape[1] else "gyroscope: no samples")
    if optics.count:
        print(f"optics (Athena) samples: {optics.last(seconds).shape[1]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--address", help="Bluetooth address of the Muse (otherwise scan)")
    ap.add_argument("--model", choices=["auto", "legacy", "athena"], default="auto", help="force the protocol instead of probing")
    ap.add_argument("--sound", choices=["pitch", "am", "head"], help="start with this sonification on")
    ap.add_argument("--fake", action="store_true", help="synthetic signals instead of a headband")
    ap.add_argument("--fake-athena", action="store_true", help="synthetic signals including the Athena's fNIRS channels")
    ap.add_argument("--test", type=float, metavar="SECONDS", help="capture this long, print statistics, no window")
    ap.add_argument("--save", metavar="PNG", help="render one frame to a file instead of opening a window")
    a = ap.parse_args()

    muse, athena = None, False
    if a.fake or a.fake_athena:
        a.fake = True
        athena = a.fake_athena
        source = FakeMuse(athena=athena)
    else:
        muse, athena = connect_muse(a.address, a.model)
        source = muse.adapter      # pump() runs the Bluetooth event loop; the callbacks fire inside it
    son = Sonifier(a.sound) if a.sound else None

    try:
        if a.test:
            run_test(source, a.test, son)
        elif a.save:
            save_frame(source, a.fake, a.save, athena)
        else:
            run_window(source, a.fake, son, athena)
    except KeyboardInterrupt:
        pass
    finally:
        if son is not None:
            son.close()
        if muse is not None:
            try:
                muse.stop(); muse.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
