"""Signals: ring buffers, the headband connection, and a synthetic headband for testing."""
import sys
import time

import numpy as np

FS = 256                     # EEG sampling rate, Hz
FS_IMU = 52                  # accelerometer and gyroscope rate, Hz
FS_OPT = 64                  # Athena near-infrared (optics) rate, Hz
CH = ["TP9", "AF7", "AF8", "TP10"]   # left ear, left forehead, right forehead, right ear
BANDS = [("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30)]
HISTORY_SECONDS = 60


class Ring:
    """A ring buffer of the last `seconds` of a multichannel signal, with an estimate of when each sample arrived."""

    def __init__(self, channels, fs, seconds):
        self.fs = fs
        self.n = fs * seconds
        self.buf = np.zeros((channels, self.n))
        self.i = 0
        self.count = 0           # total samples ever written (absolute index of the next sample)
        self.t_last = None       # monotonic time of the last push

    def push(self, block):            # block: (channels, k)
        k = block.shape[1]
        idx = (self.i + np.arange(k)) % self.n
        self.buf[:, idx] = block
        self.i = (self.i + k) % self.n
        self.count += k
        self.t_last = time.monotonic()

    def last(self, seconds):
        m = int(min(self.fs * seconds, self.count, self.n))
        idx = (self.i - m + np.arange(m)) % self.n
        return self.buf[:, idx]

    def at(self, ch, abs_idx):
        """Samples of one channel at absolute indices (held at the newest sample if asked for the future)."""
        abs_idx = np.minimum(abs_idx, self.count - 1)
        return self.buf[ch, abs_idx % self.n]

    def slice(self, start, stop):
        """Samples at absolute indices [start, stop), all channels; None if not (or no longer) available."""
        if start < 0 or stop > self.count or stop - start > self.n or start < self.count - self.n:
            return None
        idx = np.arange(start, stop) % self.n
        return self.buf[:, idx]

    def time_of(self, abs_idx):
        """Estimated arrival time (monotonic seconds) of a sample, from the time of the last push."""
        if self.t_last is None:
            return None
        return self.t_last - (self.count - abs_idx) / self.fs

    def index_at(self, t):
        """Absolute sample index that was arriving at monotonic time t."""
        return int(round(self.count - (self.t_last - t) * self.fs))


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


# ----------------------------------------------------------------------------- synthetic headband
class FakeMuse:
    """Synthetic signals aligned with the wall clock: EEG noise with alpha on for 10 s and off for 10 s, a blink every
    7 s, a jaw clench every 11 s, a nod every 9 s, a head that tilts slowly; optionally the Athena's optics. It also
    answers an audio engine: an evoked response after each click it is told about, and a 40 Hz ripple while the
    steady-state tone plays, so that the averaging demos can be tested without a head."""

    def __init__(self, athena=False):
        self.t = 0.0
        self.start = time.monotonic()
        self.athena = athena
        self.rng = np.random.default_rng(0)
        self.engine = None       # set by the entry point when sound is available

    def pump(self, seconds):
        k = int(seconds * FS)
        t = self.t + np.arange(k) / FS
        alpha_gain = 15 * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * t / 20)))
        blink = 120 * np.exp(-((t % 7 - 3.5) ** 2) / (2 * 0.08 ** 2))
        clench = 60 * ((t % 11) > 5.0) * ((t % 11) < 5.4) * np.sin(2 * np.pi * 27 * t)
        block = np.empty((4, k))
        for c in range(4):
            drift = np.cumsum(self.rng.normal(0, 1, k)); drift -= np.linspace(drift[0], drift[-1], k)
            block[c] = 8 * self.rng.normal(0, 1, k) + 0.3 * drift + alpha_gain * np.sin(2 * np.pi * 10 * t + c) + clench
            if c in (1, 2):
                block[c] += blink
        if self.engine is not None:
            wall = self.start + t
            for ct in self.engine.recent_clicks(wall[0] - 0.6, wall[-1]):
                lag = wall - ct
                block[1:3] += -8 * np.exp(-((lag - 0.10) ** 2) / (2 * 0.012 ** 2)) + 6 * np.exp(-((lag - 0.19) ** 2) / (2 * 0.02 ** 2))
            if self.engine.assr_on:
                block += 1.0 * np.sin(2 * np.pi * 40 * t)
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
            slow = 1 + 0.02 * np.sin(2 * np.pi * to / 25) + 0.004 * np.sin(2 * np.pi * 1.1 * to)
            o = np.vstack([slow * (40000 + 3000 * c) for c in range(4)]) + 30 * self.rng.normal(0, 1, (4, ko))
            on_optics(o, to)
        self.t += k / FS
        # keep the synthetic clock on the wall clock
        ahead = (self.start + self.t) - time.monotonic()
        if ahead > 0:
            time.sleep(min(ahead, seconds))


# ----------------------------------------------------------------------------- the real headband
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
