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

`sounddevice` is only needed for `visualize.py --play`; everything else
works without it, and the viewer says so instead of failing if PortAudio
is missing.

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

The live classifier runs here too and its verdict is shown above the
readout, colour-coded, with the candidate and how long it has held.
`--no-classify` turns it off; the same threshold flags as
`classify_live.py` are accepted, so a rule can be tried against the
real air conditioner while watching the spectrum that produced it.

```
python tools/audio/visualize.py COM9 --play
python tools/audio/visualize.py COM9 --play --gain-db 30
python tools/audio/visualize.py --audio-test
python tools/audio/visualize.py --list-audio-devices
```

Playback is **off unless you pass `--play`**. The readout always says
which it is — `audio off`, `audio ON` with buffer statistics, or
`audio FAILED` with the reason — so a window that is silent is never
ambiguous.

`--audio-test` plays a 440 Hz tone through the same playback path and
needs no ESP32. It is written at about the level the microphone
reaches and amplified by the same `--gain-db`, so it answers the
question silence cannot: if you hear the tone, playback works and any
later silence is the signal, not the path.

`--play` sends the audio to the speakers, which is often the fastest
way to tell what a band is actually picking up. The microphone runs
tens of dB below full scale, so playback is amplified (`--gain-db`,
24 dB by default) — that is a listening aid, not a calibrated monitor,
and the readout counts clipped samples. **Use headphones** if the
microphone can hear the speakers, or it will feed back.

Playback never blocks the serial reader. Samples go into a ring buffer
that PortAudio's own callback thread drains, so a sound card that
falls behind cannot stall the reader and start losing frames. The
ESP32's clock and the sound card's are independent and will drift: if
the card runs slow the oldest audio is dropped, if it runs fast the
callback emits silence. The readout reports both, because a large
count means what you are hearing is not continuous.

Audio and the classifier are both fed from the reader thread, not from
the plot. The plot queue drops frames when a redraw falls behind,
which would put holes in the audio and corrupt the classifier's
rolling median and hold timing.

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
python tools/audio/analyze.py --plots --csv
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

`--plots` writes, into `results/plots/`: the average spectrum per state
overlaid, band energy over time per recording, and a spectrogram per
recording.

## Live classification

```
python tools/audio/classify_live.py COM9
python tools/audio/classify_live.py COM9 --diagnostics --log

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

If the serial link loses frames, the window that would have spanned
the gap is discarded rather than built out of audio either side of it.
Such a window reads tens of dB high across the spectrum and would be
misclassified; the feature history is cleared with it, and the gap is
logged.

## Offline replay

Before changing a threshold, test it against the recordings:

```
python tools/audio/classify_live.py --replay
python tools/audio/classify_live.py --replay --fan-high-threshold -66
python tools/audio/classify_live.py --replay --replay-csv
```

This runs the *same* `FeatureExtractor`, `Smoother` and `classify()`
over the labelled WAV files and reports per-file accuracy, a confusion
matrix and a per-condition breakdown. A rule change can be judged in a
second instead of by standing in front of the air conditioner.

Current defaults, against the 18 recordings:

```
Condition             files  accuracy   predicted
COMPRESSOR / clean        3      100%  COMPRESSOR
COMPRESSOR / talking      3      100%  COMPRESSOR
FAN / clean               3      100%         FAN
FAN / talking             3      100%         FAN
OFF / clean               3      100%         OFF
OFF / talking             3       96%    FAN, OFF

Overall: 99.3% of 15656 decided windows; 17/18 files perfect
```

### What the thresholds are worth

**COMPRESSOR is solid.** 30–80 Hz sits at −38 dB while the compressor
runs and −57..−59 dB in every other condition, so −48 is ~10 dB clear
of both sides. It is correct on all six compressor recordings,
including the ones with talking.

**FAN is tight, and speech is the reason.** The fan's own 1k–2k level
is about −63.7 dB, while speech with the AC off peaks near −58 dB.
The usable window for `--fan-high-threshold` is only −66 to −64; −65
is its centre, and above −63.5 fan detection collapses entirely
because the threshold passes the fan's own level. Both fan bands must
be over threshold — `--fan-require either` cannot beat 93%.

The one remaining error is one OFF-with-talking recording read as FAN
for part of its length. Level alone cannot always separate a fan from
a voice; both put real energy in the same bands. If that matters,
the next feature to try is stationarity rather than another
threshold: over a 2 s window the rolling standard deviation of 500–1k
is ~1.4 dB for a fan and ~9.8 dB for speech, which separates the
failing case far better than any level does. Spectral tilt
(1500–4000 minus 200–1200) does **not** work — it is at chance.

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
| `visualize.py` | serial -> live plots, classifier and playback |
| `analyze.py` | WAV -> feature tables and plots |
| `classify_live.py` | serial -> live OFF / FAN / COMPRESSOR |
| `recordings/` | captured WAVs, the inputs (git-ignored except `.gitkeep`) |
| `results/` | everything generated: `features.csv`, `plots/`, `live.csv`, `replay.csv` |
