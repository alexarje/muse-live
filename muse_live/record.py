"""Recording to CSV: EEG, motion, optics and band powers, with markers."""
import os
import time

import numpy as np

from .signals import FS, FS_IMU, FS_OPT, CH, BANDS, eeg, acc, gyro, optics


class Recorder:
    def __init__(self, root="recordings"):
        self.root = root
        self.dir = None
        self.files = {}
        self.written = {}
        self.t0 = None
        self.markers = 0

    @property
    def active(self):
        return self.dir is not None

    def start(self):
        self.dir = os.path.join(self.root, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.dir, exist_ok=True)
        self.t0 = time.monotonic()
        heads = {"eeg": "t," + ",".join(CH), "acc": "t,x,y,z", "gyro": "t,x,y,z", "optics": "t,o1,o2,o3,o4",
                 "bands": "t," + ",".join(b[0] for b in BANDS), "markers": "t,label"}
        for name, head in heads.items():
            f = open(os.path.join(self.dir, name + ".csv"), "w"); f.write(head + "\n"); self.files[name] = f
        for name, ring in (("eeg", eeg), ("acc", acc), ("gyro", gyro), ("optics", optics)):
            self.written[name] = ring.count
        self.markers = 0

    def stop(self):
        for f in self.files.values():
            f.close()
        self.files = {}; self.dir = None

    def mark(self, label="mark"):
        if self.active:
            self.files["markers"].write(f"{time.monotonic() - self.t0:.3f},{label}\n"); self.files["markers"].flush()
            self.markers += 1

    def log_bands(self, bp):
        if self.active and np.all(np.isfinite(bp)):
            self.files["bands"].write(f"{time.monotonic() - self.t0:.3f}," + ",".join(f"{v:.2f}" for v in bp) + "\n")

    def update(self):
        if not self.active:
            return
        for name, ring in (("eeg", eeg), ("acc", acc), ("gyro", gyro), ("optics", optics)):
            start, stop = self.written[name], ring.count
            if stop <= start:
                continue
            seg = ring.slice(start, stop)
            if seg is None:                       # fell behind the ring; skip the gap
                self.written[name] = stop; continue
            t_stop = ring.time_of(stop) - self.t0
            ts = t_stop - (stop - np.arange(start, stop)) / ring.fs
            lines = [f"{ts[j]:.4f}," + ",".join(f"{v:.3f}" for v in seg[:, j]) for j in range(seg.shape[1])]
            self.files[name].write("\n".join(lines) + "\n")
            self.written[name] = stop
        for f in self.files.values():
            f.flush()
