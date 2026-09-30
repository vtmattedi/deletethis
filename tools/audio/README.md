# INMP441 audio pipeline

Host-side tools for working out what the microphone actually hears
before any detector logic goes into the firmware.

```
INMP441 -> ESP32 -> USB serial -> capture.py -> WAV -> analyze.py
                                \
                                 -> visualize.py (live)
```

## Setup

```
pip install -r tools/audio/requirements.txt
```

The firmware must be the one in `src/main.cpp`, and the serial link
runs at **921600 baud**, not 115200. 16 kHz x 4 bytes is 64 kB/s, which
does not fit in 115200 baud. If your USB-serial adapter cannot hold
921600, lower `SERIAL_BAUD` in `src/main.cpp` and pass the same value
to `--baud` — but below about 700000 baud the stream will drop frames.

## Firmware modes

The ESP32 starts in TEXT mode and takes single-character commands:

| Command | Mode | Output |
| --- | --- | --- |
| `t` | TEXT | human-readable RMS / band / peak diagnostics |
| `c` | CAPTURE | binary PCM stream, nothing else on the wire |
| `s` | IDLE | stops both |
| `?` | — | one status line |

`capture.py` and `visualize.py` send `c` themselves, so you never have
to do this by hand. TEXT mode is there for a serial monitor.

Only one program can hold the port. Close the PlatformIO serial
monitor before capturing.

## Recording

```
python tools/audio/capture.py --list-ports
python tools/audio/capture.py COM13 --label off --seconds 30
```

Writes `recordings/2026-09-30_091500_off.wav`: mono, 16 kHz, 32-bit
PCM. The 24-bit microphone samples are stored shifted up into the
32-bit container, so nothing is lost and `sample / 2**31` is a correct
normalised amplitude.

Watch the closing summary. Lost or dropped frames mean the serial link
could not keep up, and that recording is not trustworthy.

## Live view

```
python tools/audio/visualize.py COM13
python tools/audio/visualize.py COM13 --nfft 4096 --fmax 4000
```

Waveform, spectrum, RMS/dBFS, dominant peaks, and a scrolling
spectrogram. The spectrogram is the panel worth watching: mains hum, a
steady tone, a fan spinning up and a compressor cutting in are all
obvious there and nearly invisible in a single FFT dump.

## Collecting a dataset

Keep the microphone **fixed in its final installation position** for
every recording. Moving it between states invalidates the comparison.

Aim for 30–60 s per recording, at least three per state:

```
off_01.wav          fan_01.wav          compressor_01.wav
off_02.wav          fan_02.wav          compressor_02.wav
off_03.wav          fan_03.wav          compressor_03.wav
```

Then capture the interference cases, which are what a naive threshold
will get wrong: `speech_01.wav`, `tv_01.wav`, `door_01.wav`,
`walking_01.wav`.

The state is read from the file name. Timestamps and trailing index
numbers are stripped, so both `off_01.wav` and the timestamped name
`capture.py` generates are labelled `off`.

## Analysis

```
python tools/audio/analyze.py                       # everything
python tools/audio/analyze.py --plots --csv features.csv
python tools/audio/analyze.py --nfft 4096           # 1024 / 2048 / 4096
```

Defaults match the firmware: 16 kHz, 1024-sample window, Hamming, 50%
overlap. `--nfft` changes the window size without reflashing, which is
the point of doing this on the host first.

Prints three tables:

- **Per file** — a sanity check. Two recordings of the same state that
  disagree mean something moved or something else was running.
- **Per state** — the comparison the whole exercise is for.
- **Feature separation** — the gap between state medians divided by
  the spread within each state. A band with a large gap and a small
  spread is a band worth reimplementing on the ESP32; a large gap with
  an equally large spread is not.

`--plots` writes, into `plots/`: the average spectrum per state
overlaid, band energy over time per recording, and a spectrogram per
recording.

## Live classification

```
python tools/audio/classify_live.py COM9
python tools/audio/classify_live.py COM9 --diagnostics --log live.csv

python tools/audio/classify_live.py COM9 \
    --compressor-threshold -48 \
    --fan-mid-threshold -63 \
    --fan-high-threshold -68 \
    --hold-seconds 2
```

Reads the same stream with no firmware changes and reports
OFF / FAN / COMPRESSOR:

```
STATE=FAN  30-80=-58.2  500-1k=-57.1  1k-2k=-62.8  RMS=-41.3  candidate=FAN  stable=3.4s
```

`candidate` is what the rules say right now; `STATE` is what has held
long enough to be published. State changes are logged on their own
line with a timestamp.

The rules are hierarchical: 30–80 Hz above the compressor threshold
wins outright, otherwise 500–1k and/or 1k–2k above their thresholds
means FAN, otherwise OFF. `--fan-require both` makes the second stage
stricter.

Two independent guards stop a transient becoming a state. A rolling
median over `--median-seconds` removes single-window spikes, and
`--hold-seconds` requires the candidate to persist before it is
published. A 150 ms door slam does not reach the published state even
though it is far above every threshold.

Features are computed on exactly the same terms as `analyze.py` —
verified bit-exact, not merely by eye — so a threshold read off an
analyze.py table means the same thing here.

**The default thresholds are guesses**, not measurements. They are
there so the tool runs before you have data. Record the states, run
`analyze.py`, and replace them with values taken from the per-state
table and the separation ranking.

## Deciding what goes back into the firmware

Read the separation table and the per-state spectra before writing any
rules. Then keep it rule-based to start with — RMS, two or three band
energies, a spectral ratio, and a few seconds of temporal persistence
so a door slam cannot flip the state. Porting three numbers you have
evidence for beats porting a model you cannot debug over serial.

Only then is it worth asking whether `arduinoFFT` should become
`esp-dsp`. Until the features are settled, it does not matter.

## Files

| File | Role |
| --- | --- |
| `acstream.py` | the wire protocol, shared by capture and visualize |
| `capture.py` | serial -> WAV |
| `visualize.py` | serial -> live plots |
| `analyze.py` | WAV -> feature tables and plots |
| `classify_live.py` | serial -> live OFF / FAN / COMPRESSOR |
| `recordings/` | captured WAVs (git-ignored except `.gitkeep`) |
| `plots/` | analyzer output |
