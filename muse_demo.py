#!/usr/bin/env python
"""Muse live: EEG, head motion and sound from a Muse headband, in one window, for teaching.

Works with the Muse 2016, Muse 2 and Muse S (one protocol) and the Muse S Athena (another): the script probes the
headband after connecting and picks the right decoder. It talks to the device directly over Bluetooth LE (bleak),
no browser, no LSL.

    .venv/bin/python muse_demo.py                        # scan, connect to the first Muse, open the window
    .venv/bin/python muse_demo.py --address AA:BB:CC:DD:EE:FF
    .venv/bin/python muse_demo.py --view zoo             # start in a view: overview, spectrogram, zoo, nod, erp, assr
    .venv/bin/python muse_demo.py --sound brainmix       # start with a sound mode on
    .venv/bin/python muse_demo.py --fake                 # synthetic signals, for testing without a headband
    .venv/bin/python muse_demo.py --fake-athena          # the same with the Athena's optics
    .venv/bin/python muse_demo.py --test 8               # connect, capture 8 s, print statistics, no window
    .venv/bin/python muse_demo.py --fake --view erp --save erp.png   # render one frame of a view to a file

Keys in the window: v next view, 0–7 sound, r start or stop recording, space marker, m metronome (nod view),
e clicks (evoked response view), a tone (40 Hz view), c reset the averages.

The headband must be switched on and must not be paired in the system Bluetooth settings, nor connected to the
Muse phone app.
"""
import argparse
import time

import numpy as np

from muse_live.signals import FS, FS_IMU, CH, BANDS, eeg, acc, gyro, optics, FakeMuse, connect_muse
from muse_live.analysis import band_powers
from muse_live.sound import SOUND_MODES
from muse_live.views import VIEWS, Window


def run_test(source, seconds):
    print(f"Capturing {seconds:.0f} s...")
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        source.pump(0.1)
    x = eeg.last(seconds)
    print(f"EEG samples per channel: {x.shape[1]} (expected about {int(FS * seconds)})")
    for c in range(4):
        print(f"  {CH[c]:5} mean {x[c].mean():8.1f} µV   sd {x[c].std():7.1f} µV")
    bp = band_powers()
    print("band power (dB): " + ", ".join(f"{b[0]} {v:.1f}" for b, v in zip(BANDS, bp)))
    a = acc.last(seconds); g = gyro.last(seconds)
    if a.shape[1]:
        print(f"accelerometer samples: {a.shape[1]} (expected about {int(FS_IMU * seconds)}); mean g per axis "
              + ", ".join(f"{v:.2f}" for v in a.mean(axis=1)))
    else:
        print("accelerometer: no samples")
    if g.shape[1]:
        print(f"gyroscope samples: {g.shape[1]}; mean |rotation| {np.linalg.norm(g, axis=0).mean():.1f} °/s")
    if optics.count:
        print(f"optics (Athena) samples: {optics.last(seconds).shape[1]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--address", help="Bluetooth address of the Muse (otherwise scan)")
    ap.add_argument("--model", choices=["auto", "legacy", "athena"], default="auto", help="force the protocol instead of probing")
    ap.add_argument("--view", choices=VIEWS, default="overview", help="the view to start in")
    ap.add_argument("--sound", choices=[m for m in SOUND_MODES if m], help="start with this sonification on")
    ap.add_argument("--no-sound", action="store_true", help="do not open an audio stream at all")
    ap.add_argument("--mains", type=float, default=50.0, help="mains frequency to notch out, 50 or 60 Hz")
    ap.add_argument("--fake", action="store_true", help="synthetic signals instead of a headband")
    ap.add_argument("--fake-athena", action="store_true", help="synthetic signals including the Athena's fNIRS channels")
    ap.add_argument("--test", type=float, metavar="SECONDS", help="capture this long, print statistics, no window")
    ap.add_argument("--save", metavar="PNG", help="render one frame of the chosen view to a file instead of opening a window")
    a = ap.parse_args()
    import muse_live.analysis as analysis
    analysis.MAINS_HZ = a.mains

    muse, athena = None, False
    if a.fake or a.fake_athena:
        a.fake = True; athena = a.fake_athena
        source = FakeMuse(athena=athena)
    else:
        muse, athena = connect_muse(a.address, a.model)
        source = muse.adapter      # pump() runs the Bluetooth event loop; the callbacks fire inside it

    engine = None
    if not a.no_sound and not a.test:
        try:
            from muse_live.sound import AudioEngine
            engine = AudioEngine()
            if a.fake:
                source.engine = engine
        except Exception as e:
            print(f"Sound is off ({e}).")
            if "PortAudio" in str(e):
                print("The PortAudio library is missing. On Debian and Ubuntu: sudo apt install libportaudio2, then start again.")
    recorder = None
    try:
        if a.test:
            run_test(source, a.test)
        else:
            from muse_live.record import Recorder
            recorder = Recorder()
            if a.save:
                import matplotlib; matplotlib.use("Agg")
                w = Window(source, a.fake, athena, engine, recorder, view=a.view, sound=a.sound)
                if a.view == "erp" and engine is not None:
                    engine.set_erp(True)
                if a.view == "assr" and engine is not None:
                    engine.set_assr(True)
                if a.view == "nod":
                    w.metronome_on = True
                for _ in range(int(16 / 0.06)):
                    w.tick()
                w.fig.savefig(a.save, dpi=110)
                print("saved", a.save)
            else:
                Window(source, a.fake, athena, engine, recorder, view=a.view, sound=a.sound).run()
    except KeyboardInterrupt:
        pass
    finally:
        if recorder is not None and recorder.active:
            recorder.stop()
        if engine is not None:
            engine.close()
        if muse is not None:
            try:
                muse.stop(); muse.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
