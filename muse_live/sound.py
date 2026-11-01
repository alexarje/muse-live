"""Sound: one audio engine with several mappings from the signals to sound, plus clicks and a steady-state tone."""
import threading
import time

import numpy as np

from .signals import FS, eeg
from .analysis import best_channel

AUDIO_FS = 44100
SOUND_MODES = [None, "pitch", "am", "head", "theremin", "brainmix", "pan", "drum"]
SOUND_LABELS = ["0  off",
                "1  pitch follows alpha",
                "2  tremolo of the raw EEG",
                "3  head tilt and motion",
                "4  head theremin (a scale)",
                "5  brain mix (one loop per band)",
                "6  left–right alpha pans a tone",
                "7  face drums (blink, clench)"]
PENTATONIC = 220.0 * 2 ** (np.array([0, 2, 4, 7, 9, 12, 14, 16, 19, 21, 24]) / 12)


def _kick():
    t = np.arange(int(0.35 * AUDIO_FS)) / AUDIO_FS
    f = 45 + 110 * np.exp(-t / 0.04)
    return 0.9 * np.sin(2 * np.pi * np.cumsum(f) / AUDIO_FS) * np.exp(-t / 0.12)


def _snare():
    t = np.arange(int(0.25 * AUDIO_FS)) / AUDIO_FS
    rng = np.random.default_rng(1)
    return 0.5 * rng.normal(0, 1, t.size) * np.exp(-t / 0.06) + 0.4 * np.sin(2 * np.pi * 180 * t) * np.exp(-t / 0.08)


def _click():
    t = np.arange(int(0.003 * AUDIO_FS)) / AUDIO_FS
    return 0.7 * np.sin(2 * np.pi * 2000 * t) * np.hanning(t.size)


class AudioEngine:
    """Everything is rendered in one PortAudio callback. The window thread only sets targets."""

    def __init__(self):
        import sounddevice as sd
        self.mode = None
        self.freq = 440.0; self.target = 440.0
        self.amp = 0.0; self.amp_target = 0.0
        self.pan = 0.5
        self.phase = 0.0
        self.pos = 0                      # absolute output sample counter
        self.eeg_pos = None; self.dc = 0.0; self.level = 20.0
        self.bands = np.zeros(4)          # 0..1 gains for the brain mix
        self.shots = []                   # active one-shots: [array, offset]
        self.samples = {"kick": _kick(), "snare": _snare(), "click": _click()}
        self.click_mode = None            # None, "metronome" or "erp"
        self.click_period = 1.0
        self.next_click = None
        self.click_log = []               # monotonic times at which clicks sound
        self.new_clicks = []
        self.assr_on = False
        self.lock = threading.Lock()
        self.rng = np.random.default_rng(2)
        self.stream = sd.OutputStream(samplerate=AUDIO_FS, channels=2, blocksize=512, callback=self._callback)
        self.stream.start()

    # ------------------------------------------------------------ control, from the window thread
    def set_mode(self, mode):
        self.mode = mode
        self.eeg_pos = None
        if mode is None:
            self.amp_target = 0.0

    def set_alpha(self, alpha_db, lo, hi):
        if self.mode != "pitch":
            return
        if hi - lo < 3:
            mid = 0.5 * (lo + hi); lo, hi = mid - 1.5, mid + 1.5
        x = float(np.clip((alpha_db - lo) / (hi - lo), 0, 1))
        self.target = 220.0 * 2 ** (2 * x); self.amp_target = 0.2

    def set_head(self, tilt_deg, speed_dps):
        if self.mode == "head":
            x = float(np.clip(tilt_deg / 60.0, 0, 1))
            self.target = 220.0 * 2 ** (2 * x)
            self.amp_target = 0.04 + 0.3 * float(np.clip(speed_dps / 150.0, 0, 1))
        elif self.mode == "theremin":
            k = int(np.clip(round(tilt_deg / 60.0 * (len(PENTATONIC) - 1)), 0, len(PENTATONIC) - 1))
            self.target = float(PENTATONIC[k])
            self.amp_target = 0.08 + 0.25 * float(np.clip(speed_dps / 120.0, 0, 1))

    def set_pan(self, x):
        if self.mode == "pan":
            self.pan = float(np.clip(x, 0, 1)); self.target = 330.0; self.amp_target = 0.2

    def set_bands(self, gains):
        self.bands = np.clip(np.nan_to_num(np.asarray(gains, dtype=float)), 0, 1)

    def trigger(self, kind):
        if kind in self.samples:
            with self.lock:
                self.shots.append([self.samples[kind], 0])

    def set_metronome(self, period):
        """Clicks every `period` seconds, or off with None."""
        if period is None:
            self.click_mode = None; self.next_click = None
        else:
            self.click_period = float(period)
            if self.click_mode != "metronome":
                self.click_mode = "metronome"; self.next_click = None

    def set_erp(self, on):
        self.click_mode = "erp" if on else None
        self.next_click = None

    def set_assr(self, on):
        self.assr_on = bool(on)

    def pop_new_clicks(self):
        with self.lock:
            out, self.new_clicks = self.new_clicks, []
        return out

    def recent_clicks(self, t0, t1):
        with self.lock:
            return [t for t in self.click_log[-64:] if t0 <= t <= t1]

    def close(self):
        try:
            self.stream.stop(); self.stream.close()
        except Exception:
            pass

    # ------------------------------------------------------------ rendering, in the audio thread
    def _callback(self, out, frames, time_info, status):
        out.fill(0.0)
        n = np.arange(frames)
        t = (self.pos + n) / AUDIO_FS
        mono = np.zeros(frames)
        left = right = 1.0
        mode = self.mode
        if mode in ("pitch", "head", "theremin", "pan"):
            self.freq += (self.target - self.freq) * 0.15
            self.amp += (self.amp_target - self.amp) * 0.2
            ph = self.phase + 2 * np.pi * self.freq * (n + 1) / AUDIO_FS
            mono += self.amp * np.sin(ph)
            self.phase = ph[-1] % (2 * np.pi)
            if mode == "pan":
                left, right = np.cos(self.pan * np.pi / 2), np.sin(self.pan * np.pi / 2)
        elif mode == "am" and eeg.count > FS // 4:
            step = FS / AUDIO_FS
            if self.eeg_pos is None or eeg.count - self.eeg_pos > FS * 0.5 or self.eeg_pos > eeg.count:
                self.eeg_pos = eeg.count - FS * 0.1
            idx = self.eeg_pos + n * step
            i0 = np.floor(idx).astype(int); frac = idx - i0
            ch = best_channel()
            e = eeg.at(ch, i0) * (1 - frac) + eeg.at(ch, i0 + 1) * frac
            self.eeg_pos = idx[-1] + step
            self.dc += (e.mean() - self.dc) * 0.05
            e = e - self.dc
            self.level += (np.abs(e).mean() + 1e-6 - self.level) * 0.02
            env = np.clip(e / (4 * self.level), -1, 1)
            ph = self.phase + 2 * np.pi * 200.0 * (n + 1) / AUDIO_FS
            mono += 0.3 * (0.5 + 0.5 * env) * np.sin(ph)
            self.phase = ph[-1] % (2 * np.pi)
        elif mode == "brainmix":
            step_len = 0.125                                   # a sixteenth at 120 BPM
            s = self.pos + n
            step = np.floor(s / (step_len * AUDIO_FS)).astype(int)
            ph_in = (s % int(step_len * AUDIO_FS)) / AUDIO_FS
            attack = np.minimum(1.0, ph_in / 0.003)
            g = self.bands
            bass_env = np.where(step % 4 == 0, np.exp(-ph_in / 0.15), 0.0) * attack
            mono += 0.5 * g[0] * np.sin(2 * np.pi * 55 * t) * bass_env
            pad = (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 277.18 * t) + np.sin(2 * np.pi * 329.63 * t)) / 3
            mono += 0.12 * g[1] * pad * (0.8 + 0.2 * np.sin(2 * np.pi * 0.5 * t))
            arp_f = PENTATONIC[(step % 5) + 3]
            mono += 0.2 * g[2] * np.sin(2 * np.pi * arp_f * t) * np.exp(-ph_in / 0.08) * attack
            hat = self.rng.normal(0, 1, frames) * np.exp(-ph_in / 0.02) * np.where(step % 2 == 0, 1.0, 0.5)
            mono += 0.12 * g[3] * hat
        # one-shots (drums, clicks)
        with self.lock:
            keep = []
            for shot in self.shots:
                arr, off = shot
                k = min(frames, arr.size - off)
                if k > 0:
                    mono[:k] += arr[off:off + k]
                    shot[1] = off + k
                    if shot[1] < arr.size:
                        keep.append(shot)
            self.shots = keep
            # scheduled clicks
            if self.click_mode is not None:
                if self.next_click is None:
                    self.next_click = self.pos + int(0.5 * AUDIO_FS)
                while self.next_click < self.pos + frames:
                    k = self.next_click - self.pos
                    if k >= 0:
                        arr = self.samples["click"]
                        m = min(arr.size, frames - k)
                        mono[k:k + m] += arr[:m]
                        if m < arr.size:
                            self.shots.append([arr, m])
                        try:
                            lead = time_info.outputBufferDacTime - self.stream.time
                        except Exception:
                            lead = 0.0
                        when = time.monotonic() + lead + k / AUDIO_FS
                        self.click_log.append(when); self.new_clicks.append(when)
                        self.click_log = self.click_log[-256:]
                    period = self.click_period if self.click_mode == "metronome" else 1.0 + self.rng.uniform(-0.2, 0.2)
                    self.next_click += int(period * AUDIO_FS)
        if self.assr_on:
            # a 40 Hz click train: broadband, so it drives the response harder than a modulated tone
            period = AUDIO_FS / 40.0
            phase = (self.pos + n) % period
            mono += 0.5 * np.where(phase < 0.0015 * AUDIO_FS, np.sin(2 * np.pi * 2000 * t), 0.0)
        mono = np.clip(mono, -1, 1)
        out[:, 0] = mono * left
        out[:, 1] = mono * right
        self.pos += frames
