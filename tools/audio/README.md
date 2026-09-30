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

`fastapi` and `uvicorn` are only needed for `backend/`; the
command-line tools work without them.

The firmware must be the one in `src/main.cpp`. Copy the credentials
template before the first build, or the compile stops with a message
saying so:

```
cp include/creds.example.h include/creds.h     # then fill it in
```

### The two transports have separate jobs

```
TCP     binary audio, and nothing else
Serial  commands, status and logs, and nothing else
```

Nothing has to arbitrate who owns Serial, because audio never goes
there. A log line cannot land in the middle of a PCM frame, so the
firmware is free to report Wi-Fi changes, client connects and I2S
overruns whenever they happen.

**Audio comes over TCP.** The ESP32 prints its IP at boot:

```
python tools/audio/capture.py 192.168.1.50 --label off --seconds 30
python tools/audio/visualize.py 192.168.1.50:3333
python tools/audio/classify_live.py 192.168.1.50
```

A name with a dot, or anything with an explicit `:port`, is a network
target; `COM9` and `/dev/ttyUSB0` are not. Streaming starts as soon as
the socket is accepted — there is no handshake and no command to send.
The default port is 3333, and **one client at a time**: a second
connection is refused rather than queued.

**Serial is for watching the device.** At 921600 baud:

| Command | Effect |
| --- | --- |
| `t` | TEXT mode: periodic RMS / band / peak diagnostics |
| `s` | IDLE: stop the diagnostics |
| `?` | status: sample rate, IP, port, client, I2S overruns |
| `c` | prints `# capture over serial disabled; use TCP` |

Serial also logs Wi-Fi up/down, client connect and disconnect, and
I2S overruns as they occur.

Streaming audio over serial still exists as a fallback: set
`ALLOW_SERIAL_CAPTURE` to 1 in `src/main.cpp`, reflash, and the tools
accept `COM9` again. It is off by default so the invariant above
holds. 16 kHz x 4 bytes is 64 kB/s, which
does not fit in 115200 baud. If your USB-serial adapter cannot hold
921600, lower `SERIAL_BAUD` in `src/main.cpp` and pass the same value
to `--baud` — but below about 700000 baud the stream will drop frames.

## Recording

```
python tools/audio/capture.py 192.168.1.50 --label off --seconds 30
```

The serial monitor can stay open while you record, because the two
transports no longer compete: audio goes over TCP and the monitor only
carries text.

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

The classifier is fed from the reader thread, not from the plot. The
plot queue drops frames when a redraw falls behind, which would
corrupt the classifier's rolling median and hold timing; it needs
every frame, in order.

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

## Backend and web UI

```
python tools/audio/backend/app.py --target 192.168.1.50:3333
```

```
Backend starting
ESP32: 192.168.1.50:3333
stream: connected to 192.168.1.50:3333 (16000 Hz, 512 samples/frame)
Web: http://127.0.0.1:8000
```

A long-running local service that is **the only TCP client** the ESP32
has. It owns reception, feature extraction, classification, runtime
config and event recording. The browser displays and configures; it
never classifies and never talks to the ESP32.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/status` | stream health, state, features, config |
| `GET /api/config` | current thresholds and timings |
| `PATCH /api/config` | change them live |
| `GET /api/events` | recorded transitions, newest first |
| `GET /api/events/{id}` | one event's full metadata |
| `GET /api/events/{id}/audio` | that event's WAV |
| `GET /api/events/{id}/timeline` | feature timeline recomputed from the WAV |
| `GET /api/history` | compact state history |
| `WS /ws/live` | snapshots, 5 Hz by default |

The classifier runs at the full window rate (~31/s); only the UI
updates are throttled. Raw PCM never leaves the backend.

`PATCH /api/config` applies at once without touching the TCP
connection. The published state is kept — moving a threshold is not an
observation — but the rolling median and the candidate are cleared, so
the hold has to be earned again under the new rules.

### Transition events

A **published** state change writes two files to `results/events/`:

```
2026-09-30_130533_FAN_to_COMPRESSOR.wav     raw evidence
2026-09-30_130533_FAN_to_COMPRESSOR.json    decision summary
```

15 s before and 15 s after, unmodified PCM in the same 32-bit format
`capture.py` writes, so `analyze.py` and `classify_live.py --replay`
read them directly. The JSON holds the features at the transition, the
decision-window statistics, how long the candidate held, and the exact
config in force at the time. Overlapping captures are independent: a
second transition during the first's tail starts its own event.

The classifier's *first* verdict after startup is not a transition and
is not recorded — it is the classifier settling, not a change.

Note that a transition publishes `holdSeconds` after the audio really
changed, so that much of the pre-roll is already the new state. At the
defaults (15 s pre-roll against a 2.5 s decision) there is plenty of
genuine "before"; raising `holdSeconds` towards `eventPreSeconds` eats
into it.

### History

One row per second in SQLite at `results/audio.db`: state, candidate,
hold, RMS, the four bands, and stream health. About 86k rows a day,
which SQLite does not notice. The ~31 classifier windows per second
are deliberately *not* stored — they only matter around a transition,
and that is what the event WAVs are for.

```
GET /api/history?seconds=900
GET /api/history?from=2026-09-30T13:00:00&to=2026-09-30T14:00:00
```

`from` and `to` accept a unix timestamp or an ISO-8601 string. The
response is columnar, because the only consumer is a chart. The UI
plots RMS and the three rule bands with a state strip underneath.

### Event timeline

`GET /api/events/{id}/timeline` recomputes the feature timeline from
the event's WAV, using the classifier config recorded with it. Times
are relative to the transition, so negative is before it.

Deriving it on demand keeps the storage honest: the WAV stays the only
copy of the evidence and the JSON stays a decision summary, with no
third place for the two to disagree. It also means an old event can be
re-examined under different thresholds without having recorded
anything extra.

One caveat the UI states on the chart: the replay starts cold, so the
published state near the left edge is still warming up and will not
match what was live at the time — live had history from before the
pre-roll. The band levels and the candidate are exact; only the held
state lags.

### Not built yet

The replay API, `/ws/audio` browser listening, and the replay UI are
later phases. `classify_live.py --replay` covers replay from the
command line in the meantime.

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
| `visualize.py` | live plots and classifier |
| `analyze.py` | WAV -> feature tables and plots |
| `classify_live.py` | serial -> live OFF / FAN / COMPRESSOR |
| `recordings/` | captured WAVs, the inputs (git-ignored except `.gitkeep`) |
| `backend/` | the local service: stream, classifier, config, events, API |
| `web/` | the browser UI it serves |
| `results/` | everything generated: `events/`, `plots/`, `features.csv`, … |
