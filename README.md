# Muse live

A single Python script that connects to a Muse headband over Bluetooth LE and shows, in one window, the four raw EEG traces, the power in the delta, theta, alpha and beta bands, alpha power over the last minute, and the head's motion from the built-in accelerometer. Three sonifications can be switched on from buttons in the window. Made for teaching: the point of the demo is the eyes-open, eyes-closed difference, where alpha rises within a second or two of closing the eyes, and the artefacts that blinks, jaw tension and head movements leave in the signal.

It talks to the headband directly through [bleak](https://github.com/hbldh/bleak) and the Muse protocol implementations in [muselsl](https://github.com/alexandrebarachant/muse-lsl). No browser, no Web Bluetooth, no LSL stream to set up.

![The window on a synthetic signal](docs/window.png)

## Which headbands

One script serves all of them: after connecting it probes the device and picks the decoder.

- Muse 2016 (MU-02): verified, 256 EEG samples per second per channel and 52 per second from the accelerometer and gyroscope.
- Muse 2 and Muse S (first and second generation): same protocol in muselsl, so they should work; not tried.
- Muse S Athena (2025): verified, 256 EEG samples per second per channel, 52 from the motion sensors and 64 from the near-infrared optodes, which get their own panel. The script probes the headband itself before subscribing, since muselsl's own auto mode subscribes to the old characteristics first and fails on an Athena; `--model athena` or `--model legacy` skips the probe.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python muse_demo.py
```

The script scans for 12 s, connects to the first Muse it finds, and opens the window. To skip the scan, give the address:

```bash
.venv/bin/python muse_demo.py --address AA:BB:CC:DD:EE:FF
```

Other modes:

```bash
.venv/bin/python muse_demo.py --sound am        # start with a sonification on (pitch, am or head)
.venv/bin/python muse_demo.py --test 8          # capture 8 s, print sample counts and statistics, no window
.venv/bin/python muse_demo.py --fake            # synthetic signals, no headband
.venv/bin/python muse_demo.py --fake --sound head --test 5   # hear a sonification without a headband
.venv/bin/python muse_demo.py --fake --save frame.png
```

## The window

- Top: the four EEG channels, TP9 and TP10 behind the ears, AF7 and AF8 on the forehead, band-pass filtered 1 to 40 Hz. A red channel label means a poor contact: wet the skin, press the band, or move hair away.
- Middle left: band power in dB over the last 2 s, mean of the four channels, alpha highlighted.
- Middle right: alpha power over the last minute. Ask the wearer to close their eyes for 15 s, then open them, and point at the step.
- Bottom left: the accelerometer's three axes in g. Tilting the head shifts the lines; a nod makes a bump. The corner shows the tilt away from the starting orientation and the rotation speed from the gyroscope.
- Bottom right: the sound buttons. The keys 0, 1, 2 and 3 do the same.

Sound mappings:

1. Pitch follows alpha: a sine from 220 Hz at the lowest alpha of the last minute to 880 Hz at the highest.
2. Tremolo of the raw EEG: a 200 Hz tone whose loudness follows the raw EEG of the best-contact channel, read in real time, so the alpha rhythm is a 10 Hz tremolo and a blink a thump.
3. Head tilt and motion: pitch follows how far the head is tilted from where it started (0 to 60 degrees over 220 to 880 Hz), loudness follows how fast it turns.

## Before a demonstration

1. Charge the headband. Switch it on; the light blinks while it waits for a connection. The 2016 model switches itself off after a few minutes without one.
2. The Muse must not be paired in the computer's Bluetooth settings. A system pairing holds the connection and nothing else can reach the device, which is also why browser tools such as EEGEdu show nothing in that state. On Linux: `bluetoothctl remove <address>`.
3. Close the Muse app on any phone nearby; it grabs the headband as soon as it sees it.
4. Only one program can hold the headband. The first connection after it wakes can take 20 s.
5. Run `--test 8` once. Expect about 2048 EEG samples per channel and 416 from the accelerometer. Standard deviations above a few hundred microvolts mean a poor contact, or that the band is lying on the table.

## Notes

- The pin `bitstring<5` in requirements.txt matters: muselsl's packet decoder uses a call that bitstring 5 removed, and without the pin the headband connects but no samples ever arrive.
- Sound uses [sounddevice](https://python-sounddevice.readthedocs.io/), which needs the PortAudio library (`libportaudio2` on Debian and Ubuntu).
- Plotting uses matplotlib's Tk backend; a machine without `python3-tk` can still use `--test` and `--save`.
- The script does not record. For recording, muselsl's own `muselsl stream` and `muselsl record` do that.

## Licence

MIT. Written by Alexander Refsum Jensenius with Claude, for the course MUS2640 Sensing Sound and Music at the University of Oslo.
