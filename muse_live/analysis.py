"""Analysis: filters, band powers, head pose, detectors for blinks, clenches and nods, and online averaging."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, spectrogram as sp_spectrogram

from .signals import FS, FS_IMU, BANDS, eeg, acc, gyro

SOS_WIDE = butter(4, [1, 40], btype="band", fs=FS, output="sos")
SOS_SLOW = butter(2, [0.5, 6], btype="band", fs=FS, output="sos")
SOS_FAST = butter(4, [20, 45], btype="band", fs=FS, output="sos")
BAD_CONTACT_UV = 150


def filtered_raw(seconds=4):
    x = eeg.last(seconds)
    if x.shape[1] < FS:
        return x
    x = x - x.mean(axis=1, keepdims=True)
    return sosfiltfilt(SOS_WIDE, x, axis=1)


def band_powers(seconds=2, per_channel=False):
    """Band power in dB from the last `seconds`; mean over channels unless per_channel."""
    x = eeg.last(seconds)
    if x.shape[1] < FS:
        shape = (len(BANDS), 4) if per_channel else (len(BANDS),)
        return np.full(shape, np.nan)
    x = x - x.mean(axis=1, keepdims=True)
    f, p = welch(x, fs=FS, nperseg=FS, axis=1)
    out = []
    for _, lo, hi in BANDS:
        sel = (f >= lo) & (f < hi)
        out.append(10 * np.log10(p[:, sel].mean(axis=1) + 1e-12))
    out = np.array(out)
    return out if per_channel else out.mean(axis=1)


def contact_quality():
    """Standard deviation of the last second per channel; large means a poor contact."""
    x = eeg.last(1)
    if x.shape[1] < FS // 2:
        return np.zeros(4)
    return x.std(axis=1)


def best_channel():
    """The channel with the smallest spread in the last second, which is the best contact."""
    sd = contact_quality()
    return int(np.argmin(sd)) if sd.any() else 1


def eeg_spectrogram(ch, seconds=20):
    """(f, t, dB) of one channel over the last `seconds`, 0 to 40 Hz, 1 s windows stepped by 0.25 s."""
    x = eeg.last(seconds)
    if x.shape[1] < 2 * FS:
        return None
    y = x[ch] - x[ch].mean()
    f, t, S = sp_spectrogram(y, fs=FS, nperseg=FS, noverlap=FS - FS // 4)
    sel = f <= 40
    return f[sel], t - t[-1], 10 * np.log10(S[sel] + 1e-12)


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


class FaceDetector:
    """Blinks (a large slow deflection on the forehead pair) and jaw clenches (a burst of 20 to 45 Hz on all channels).
    update() returns a list of (kind, abs_sample_index) for new events."""

    def __init__(self):
        self.last_blink = -10 ** 9
        self.last_clench = -10 ** 9
        self.blink_thr = 60.0
        self.clench_base = None

    def update(self):
        events = []
        if eeg.count < 3 * FS:
            return events
        x = eeg.last(3)
        start = eeg.count - x.shape[1]
        fast = sosfiltfilt(SOS_FAST, x - x.mean(axis=1, keepdims=True), axis=1)
        rms_all = np.sqrt((fast ** 2).mean(axis=0))
        base = np.median(rms_all)
        self.clench_base = base if self.clench_base is None else 0.95 * self.clench_base + 0.05 * base
        now = np.sqrt((fast[:, -int(0.15 * FS):] ** 2).mean())
        recent_fast = np.sqrt((fast[:, -int(0.6 * FS):] ** 2).mean())
        clenching = now > max(4 * self.clench_base, 12.0)
        if clenching:
            idx = eeg.count - int(0.075 * FS)
            if idx - self.last_clench > int(0.5 * FS):
                self.last_clench = idx
                events.append(("clench", idx))
        if recent_fast > max(2.5 * self.clench_base, 8.0) or now > max(2.0 * self.clench_base, 8.0):
            return events                  # the edges of a muscle burst leak into the slow band; no blink call now
        fore = x[1:3].mean(axis=0)
        fore = fore - fore.mean()
        slow = sosfiltfilt(SOS_SLOW, fore)
        lo, hi = int(0.45 * FS), int(0.1 * FS)   # look 0.1 to 0.45 s back, so a burst starting now has shown itself
        recent = slice(-lo, -hi)
        mad = np.median(np.abs(slow - np.median(slow))) * 1.4826 + 1e-6
        thr = max(self.blink_thr, 5 * mad)
        seg = np.abs(slow[recent])
        if seg.max() > thr:
            idx = start + x.shape[1] - lo + int(np.argmax(seg))
            if idx - self.last_blink > int(0.4 * FS):
                self.last_blink = idx
                events.append(("blink", idx))
        return events


class NodDetector:
    """Nods from the gyroscope: a rotation-speed peak above a threshold, with a refractory period.
    update() returns new nod times (monotonic seconds); tempo() the rate from the last few nods."""

    def __init__(self, threshold=80.0):
        self.thr = threshold
        self.last_idx = -10 ** 9
        self.times = []

    def update(self):
        new = []
        if gyro.count < FS_IMU:
            return new
        g = gyro.last(0.6)
        start = gyro.count - g.shape[1]
        mag = np.linalg.norm(g, axis=0)
        j = int(np.argmax(mag))
        if mag[j] > self.thr and j < len(mag) - 3 and mag[j] >= mag[j + 1] and mag[j] >= mag[j - 1 if j else 0]:
            idx = start + j
            if idx - self.last_idx > int(0.35 * FS_IMU):
                self.last_idx = idx
                t = gyro.time_of(idx)
                self.times.append(t)
                self.times = self.times[-32:]
                new.append(t)
        return new

    def period(self):
        if len(self.times) < 3:
            return None
        d = np.diff(self.times[-6:])
        d = d[(d > 0.25) & (d < 2.5)]
        return float(np.median(d)) if len(d) else None


class EpochAverager:
    """Average the EEG around events given as monotonic times. mean() returns (lags_s, (4, L) mean in µV, n)."""

    def __init__(self, pre=0.1, post=0.5):
        self.pre, self.post = pre, post
        self.L = int((pre + post) * FS)
        self.sum = np.zeros((4, self.L))
        self.n = 0
        self.pending = []

    def add(self, t):
        self.pending.append(t)

    def update(self):
        if eeg.t_last is None:
            return
        keep = []
        for t in self.pending:
            end_idx = eeg.index_at(t + self.post)
            if end_idx > eeg.count:
                keep.append(t); continue
            start = end_idx - self.L
            seg = eeg.slice(start, end_idx)
            if seg is None:
                continue
            seg = seg - seg[:, :int(self.pre * FS)].mean(axis=1, keepdims=True)
            if np.abs(seg).max() > 150:        # reject epochs with a blink or a big artefact
                continue
            self.sum += seg
            self.n += 1
        self.pending = keep

    def mean(self):
        lags = np.arange(self.L) / FS - self.pre
        return lags, (self.sum / self.n if self.n else np.zeros_like(self.sum)), self.n

    def reset(self):
        self.sum[:] = 0; self.n = 0; self.pending = []


class SpectrumAverager:
    """Average power spectrum of 2 s windows, for the 40 Hz steady-state response."""

    def __init__(self):
        self.f = None
        self.sum = None
        self.n = 0
        self.last_count = 0

    def update(self):
        if eeg.count < 2 * FS or eeg.count - self.last_count < FS:
            return
        self.last_count = eeg.count
        x = eeg.last(2)
        x = x - x.mean(axis=1, keepdims=True)
        if np.abs(x).max() > 200:
            return
        f, p = welch(x, fs=FS, nperseg=2 * FS, axis=1)
        p = p.mean(axis=0)
        if self.sum is None:
            self.f, self.sum = f, np.zeros_like(p)
        self.sum += p
        self.n += 1

    def result(self):
        """(f, dB spectrum 20 to 60 Hz, n, SNR at 40 Hz in dB against the neighbours) or None."""
        if not self.n:
            return None
        p = self.sum / self.n
        sel = (self.f >= 20) & (self.f <= 60)
        at = p[np.argmin(np.abs(self.f - 40))]
        nb = p[((self.f >= 34) & (self.f <= 38)) | ((self.f >= 42) & (self.f <= 46))].mean()
        return self.f[sel], 10 * np.log10(p[sel] + 1e-12), self.n, 10 * np.log10(at / (nb + 1e-12))

    def reset(self):
        self.sum = None; self.n = 0
