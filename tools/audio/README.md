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
command-line tools work without them. `compare_models.py` additionally
needs scikit-learn (`pip install -r tools/audio/requirements-analysis.txt`);
the backend and the Docker image deliberately do not.

### Docker Compose

From the repository root, stop any native backend process first, then run:

```sh
docker compose up --build -d
docker compose logs -f audio-backend
```

Open <http://localhost:8000>. The Compose defaults target the ESP32 at
`10.10.3.17:3333`. To change the stream address, published web port, or
timezone, copy `.env.example` to `.env` and edit its values before starting.

Compose bind-mounts the existing host directory
`tools/audio/results` at the identical location inside the image. The current
`audio.db` (including its WAL files), `config.json`, event JSON metadata, and
event WAV files therefore remain in place and survive image rebuilds and
container removal. `docker compose down` does not remove them.

Do not run the native backend and Compose service together. The ESP32 permits
one audio client, and the history database should have only one application
writer. Stop the container cleanly with:

```sh
docker compose down
```

The firmware must be the one in `src/main.cpp`. Copy the credentials
template before the first build, or the compile stops with a message
saying so:

```
cp include/creds.example.h include/creds.h     # then fill it in
```

### The two transports have separate jobs

```
TCP     binary audio, and nothing else
Serial  status and connection state, and nothing else
```

The device captures and forwards. It does not analyse: there is no FFT
on the ESP32 and no audio diagnostics on Serial. Every band, threshold
and rule lives in the PC tools, where it can be changed without a
reflash, and `arduinoFFT` is no longer a dependency.

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
| `?` | status: sample rate, frame size, uptime, IP, port, client, I2S overruns |
| `c` | prints `# capture over serial disabled; use TCP` |

Serial also logs Wi-Fi up/down, client connect and disconnect, and
I2S overruns as they occur. It says nothing else: there is no
periodic output to scroll past while you are watching for a
connection problem.

Streaming audio over serial still exists as a fallback: set
`ALLOW_SERIAL_CAPTURE` to 1 in `src/main.cpp`, reflash, and the tools
accept `COM9` again. It is off by default so the invariant above
holds; it re-adds the `c` and `s` commands. 16 kHz x 4 bytes is
64 kB/s, which does not fit in 115200 baud. If your USB-serial adapter cannot hold
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
python tools/audio/analyze.py --fan-separation
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

Both paths import `features.py`. In addition to RMS and the original
rule bands, it produces 80–200, 200–500, 2–4 kHz, the two wider bands,
dominant peak, spectral centroid/flatness/crest/flux, seven relative
spectral-shape differences, and rolling 2 s stability values. These are
diagnostics only: classification still reads exactly 30–80, 500–1k and
1k–2k with the existing thresholds, median and hold.

`analyze.py --fan-separation` ranks every shared feature for FAN versus
OFF/noise. It reports medians, P10/P90, directional ROC AUC and overlap,
while keeping each interference label visible. Each observation is one
recording median—not an adjacent FFT window—so long recordings cannot
dominate and there is no window-level train/test leakage. Future model
experiments must likewise split by recording, preferably by session or
day.

For the next dataset pass, record separate sessions for OFF and FAN
with TV, printer idle, printer moving, talking, music, and miscellaneous
loud noises. Prefer several independent files per condition over one
long file; names such as `off_printer_moving_01.wav` preserve the
interference label in CSV and separation reports.

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

## Classifier versions

```
v1   one state: 30-80 Hz high -> COMPRESSOR; else fan bands high -> FAN; else OFF
v2   three independent observations: fan_detected, compressor_detected, beep
```

v1 is frozen. It is the baseline, and the rule the events recorded
before v2 were classified by; a regression test replays today's
Smoother against a verbatim copy of the old one and requires identical
output, window for window.

### What v2 reports, and what it refuses to

| Observation | Type | Meaning |
| --- | --- | --- |
| `fan_detected` | bool | the fan spectrum is present and holding still |
| `compressor_detected` | bool | sustained energy in 30-80 Hz |
| `beep` | event | a short pure tone near 4.12 kHz just happened |

**Watson does not say whether the air conditioner is on, and does not
say a command was accepted.** It does not know that a beep acknowledges
a command or that a compressor starting means a request succeeded. The
controller sends the commands and keeps the history, so it is the one
place those facts can be combined. v1's single state could not say "fan
and compressor both running" and made the compressor hide the fan;
`fan=true, compressor=true` and `fan=true, compressor=false` are both
ordinary v2 answers.

```
features -> median (one shared history)
              |-> fan detector        -> hold timer -> fan_detected
              '-> compressor detector -> hold timer -> compressor_detected
raw windows -> beep detector (state machine)         -> BeepEvent
```

Each detector is a module of its own under `classifier/detectors/`
(`fan.py`, `compressor.py`, `beep.py`). They are written once on numpy
booleans, so the live service (scalars) and the threshold search
(arrays) run the same code. Fan and compressor share the median history
but not a hold timer, so neither can delay or cancel the other.
`AcousticObservations.legacy_state()` can still fold the two booleans
into OFF / FAN / COMPRESSOR; it exists for the history chart and for
comparison with v1, and is not a v2 output.

**Fan.** Energy in 500-1k and 1k-2k is not enough: speech, a printer or
a television put power there too. What a fan does that they do not is
hold still, so the fan also needs `1k-2k_std <= fanStabilityThreshold`
(the standard deviation of that band over the last two seconds), with at
least `fanStabilityMinSeconds` of history behind it. This is not
`holdSeconds`: the hold stops a short-lived candidate being *published*;
stationarity says what the signal *is*.

**Compressor.** 30-80 Hz at or above `compressorThreshold`. Nothing
about the fan enters.

**Beep.** The unit's tone is a pure tone near 4.12 kHz lasting about
150 ms. The detector measures, per analysis window, the power in
4050-4180 Hz against its neighbours (3850-4000 and 4230-4380 Hz), as
*contrast* in dB, and runs a small state machine on that:

* a run starts and continues while contrast stays above the **edge**
  value (10 dB, generous) and the peak sits in the frequency window;
* when it ends it is accepted only if its best contrast reached the
  **peak** value (15 dB), it lasted 60-300 ms, its pitch did not wander
  more than 50 Hz, and it is at least -80 dBFS;
* one gap window is tolerated, and an 80 ms refractory time stops the
  decaying tail from becoming a second event;
* duration is measured centre to centre of the first and last window,
  because a window that overlaps the tone at all would otherwise add up
  to a window length (150 ms reads 192 ms; a 30 ms blip reads 64 ms and
  would pass a 60 ms minimum).

Rejections are counted by reason (`weak`, `too_short`, `too_long`,
`unstable_pitch`) so a missed beep can be explained. Contrast, not
absolute level, is what separates a beep from a loud room.

**What the recorded beeps say** (`check_beeps.py`, `evaluate_events.py`):
19 beeps in 65 minutes of event audio, and none in 14.5 minutes of
audio with no commands in it. Every real beep had contrast of at least
15.8 dB and lasted at least 64 ms; the look-alikes were single 32 ms
windows at 12-13.5 dB, and one loud wandering ring, which is why the
pitch-spread check exists. **The margin is thin: about 2 dB between the
weakest real beep and the strongest look-alike.** The thresholds were
set from 19 beeps in one room; treat them as provisional and re-run
`check_beeps.py` on the first collected data.

Beeps in the old reviewed events, by what the reviewer said happened
(beeps seen in the OFF-to-COMPRESSOR and COMPRESSOR-to-OFF events lead
the published change by about 2.5 s, which is the 2 s hold plus half a
second):

| reviewed as | events | with a beep |
| --- | --- | --- |
| OFF → COMPRESSOR | 5 | 4 |
| COMPRESSOR → OFF | 4 | 2 |
| OFF → FAN | 2 | 2 |
| FAN → OFF | 2 | 1 |
| FAN → COMPRESSOR | 27 | 1 |
| OFF → OFF | 21 | 2 |
| COMPRESSOR → FAN | 21 | 0 |

Most of those recordings were made without marking when a command was
sent, so this is a count of beeps near transitions, not a measure of how
often a command beeps. The `commandContext` marker below exists to make
that measurable.

### Where things are stored

Each version keeps its events, history and saved settings in its own
folder, and the two never share one:

| | v1 | v2 |
| --- | --- | --- |
| events | `results/events/` | `results/v2/events/` |
| history | `results/audio.db` | `results/v2/audio.db` |
| settings | `results/config.json` | `results/v2/config.json` |
| evaluation, plots | | `results/v2/evaluation/`, `results/v2/plots/` |

The v1 events are the labelled evidence v2 is judged against; a v2 run
that added its own to that folder would change the baseline it is being
compared with. So `AppConfig` refuses the mix (v2 pointed at a v1 path,
or v1 at a v2 path). Switching version is a restart, never a setting,
and a saved settings file is never reinterpreted across versions — one
with no version key predates versions and is v1. The v1 events are
never modified, migrated or renamed.

The backend's `--classifier` defaults to **v2**, so with Docker Compose
the container writes to `results/v2/`; add `--classifier`, `v1` to its
`command` to keep growing the v1 dataset. `classify_live.py` stays v1.

### v2 events

An event is written when a published observation changes or a beep is
found: 15 s before and 15 s after, raw PCM, with the decision that
caused it.

| `eventType` | file suffix | written when |
| --- | --- | --- |
| `fan` | `FAN_ON`, `FAN_OFF` | `fan_detected` changed |
| `compressor` | `COMPRESSOR_ON`, `COMPRESSOR_OFF` | `compressor_detected` changed |
| `beep` | `BEEP` | a beep was found |
| `manual` | `MANUAL` | "Record event now" |

Fan and compressor changes are independent: a compressor starting is a
`COMPRESSOR_ON` event and says nothing about the fan. The first value
published after a start or an audio gap is a baseline, not a change.
Every event has `observations` (both values at that moment), the
config in force, and a `beep` object (duration, pitch, contrast, level)
for beeps.

**Review.** v1 reviews said what the combined state really was before
and after. A v2 review is about one observation:

| event | fields |
| --- | --- |
| `fan`, `compressor` | `correct`; `actualValue` (required when wrong, what it really was afterwards) |
| `beep` | `correct` (`actualBeep` is derived: wrong means it was not a beep) |
| any | `interference`, `notes` |

Contradictions (`correct: true` with a different `actualValue`) are
refused. The same endpoints accept both shapes, chosen by the event.

**Command markers.** `POST /api/commands` `{command, expectedBeep,
note}` notes that a command was just sent (the UI has a "Command marker"
card). An event within 10 s of one carries it as `commandContext`
(`command`, `sentAt`, `expectedBeep`, `secondsFromEvent`), and
`PATCH`/`DELETE /api/events/{id}/command-context` set or remove it by
hand. It is data about the world and nothing reads it back: it does not
change what is detected.

### v2 settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `compressorThreshold` | `-38` dB | 30-80 Hz level that counts as a compressor |
| `fanMidThreshold`, `fanHighThreshold`, `fanRequire` | `-62`, `-65`, `both` | fan energy |
| `fanStabilityFeature` | `1k-2k_std` | which temporal feature must stay small |
| `fanStabilityThreshold` | `6.0` dB | largest value still read as a steady fan |
| `fanStabilityMinSeconds` | `1.0` | history needed before the gate trusts it |
| `beepMinContrastDb` / `beepEdgeContrastDb` | `15` / `10` | peak and edge contrast |
| `beepMinMs` / `beepMaxMs` | `60` / `300` | accepted duration |
| `beepMinHz` / `beepMaxHz` | `4050` / `4180` | accepted pitch |
| `beepMaxPeakSpreadHz` | `50` | how far the pitch may wander |
| `beepMinLevelDb` | `-80` | quietest accepted tone |

### Evaluating against reviewed events

```
python tools/audio/evaluate_events.py tools/audio/results/events
python tools/audio/check_beeps.py tools/audio/results/events
```

Replays each reviewed event's WAV through the same extractor and
detectors the backend uses and scores each observation separately.
**Ground truth is `review.actualFrom` / `review.actualTo` and nothing
else.** Unreviewed and `UNKNOWN` events are counted but never scored.
Output goes to `results/v2/evaluation/`.

**Temporary compatibility truth.** The old labels name one state, so to
score two yes/no questions with them the evaluator assumes the
compressor only runs while the fan turns:

```
OFF = (fan no, compressor no)   FAN = (fan yes, compressor no)   COMPRESSOR = (fan yes, compressor yes)
```

It is a bridge so the old reviews stay usable, written into
`summary.json`, and is not the v2 review schema. It is also why
"COMPRESSOR" windows now count against the fan detector when the fan is
not detected, which the old scoring (the compressor shadowed the fan)
never exposed. Once v2 reviews exist, score those instead.

A window is scored only where its label can be trusted: the first 5 s
and 4 s either side of a real transition are skipped. Anything chosen
from the data is validated **by group** — events within 120 s of each
other — never by window.

| File | What it holds |
| --- | --- |
| `summary.json` | parameters, dataset counts, per-observation metrics, search, beeps |
| `confusion_v1.csv`, `confusion_v2.csv` | window-level 2×3 confusion per observation |
| `per_event.csv` | every event, v1 and v2 side by side |
| `per_interference.csv` | correct share per truth value per interference tag |
| `feature_separation.csv` | which features separate yes from no, per observation |
| `candidate_thresholds.csv` | every setting the search tried |

The search uses a vectorised path rather than the smoother's Python
loop; it reimplements only the rolling median and the hold, and checks
them against the real smoother window for window on every run.

### What the evidence says (88 reviewed events, 2026-10-01)

53,014 scored windows in 63 groups; published, window level, v1 as run
live (compressor -38 dB) against v2 defaults:

| | v1 | v2 |
| --- | --- | --- |
| fan: reported but not running | 3010 | **1544** (-49%) |
| fan: running but not reported | 336 | 1265 |
| fan recall / specificity | 98.9% / 84.8% | 96.2% / 92.2% |
| fan balanced accuracy | 0.918 | **0.942** |
| compressor recall / specificity | 95.5% / 98.3% | 95.5% / 98.3% |
| compressor balanced accuracy | 0.969 | 0.969 |

The gate trades real fan windows for false ones: it removes about half
the false fans and loses about 2.7 points of fan recall. The compressor
is unchanged because its detector is. Its threshold sits in the middle
of a -42..-34 dB plateau.

The fan gate threshold came from the data and is far looser than a first
guess. Below about 2 dB recall collapses (a real fan moves a little);
the plateau within 0.005 of the best is 3.8-6.4 dB (the search
recommends 5.1, the shipped value is 6.0 and sits inside it).
Choosing it from 62 groups and testing on the held-out one gives the
same 0.942, so the figure is not an artefact of tuning.

Things to know before trusting it:

* **Thresholds belong to an installation.** v1's code default of -48 dB
  for the compressor was tuned on the first recordings; in the newer
  sessions the OFF level is near -49 dB, so -48 would call a third of
  OFF windows COMPRESSOR. v2 defaults to -38. Re-run the evaluation
  when the microphone moves.
* **A stationarity gate costs FAN-under-noise.** Speech or television
  over a running fan raises its band variation. In the reviewed events
  the one television event with a running fan (872 windows) is lost
  entirely (0%), while printer events keep 95%; the fan with
  interference cases are too few (about 4,700 windows, three tags) to
  say more. If the unit must be detected while people talk over it,
  expect to need another feature.
* **The residual false fan is concentrated.** Events tagged "other"
  account for most of what remains (78% correct); that is a steady
  non-AC sound sitting in both fan bands, which a variance test cannot
  remove.
* The labelled set is small (88 events, 15 OFF with interference).
  Differences of a point or two are noise.

### Do models do better?

```
python tools/audio/compare_models.py tools/audio/results/events
```

Fits logistic regression, shallow decision trees and a small random
forest, one binary model per observation, and scores them next to v1
and the v2 detector — same inputs, same publication hold, folds split by
group. Analysis only: nothing in the backend imports it or scikit-learn
(`pip install -r requirements-analysis.txt`). A model has to beat the
detector by 0.02 balanced accuracy to be worth discussing, and if a
shallow tree comes close, read its splits and fold them into the
detector rather than deploying a model. Run it again once v2 data
exists.

### Before collecting the v2 dataset

1. Start the backend (default v2). Confirm three cards in the page and
   that `results/v2/events/` is where files appear.
2. Press **Mark command sent** when you send a command by remote, with
   "a beep is expected" set honestly, so beep rate and delay can be
   measured later.
3. Review each event as it comes: fan and compressor changes as correct
   or wrong (say what it really was when wrong), beeps as real or not.
   Tag interference.
4. Collect OFF periods with no command at all (the false-beep rate comes
   from these), and some with the television and talking over a running
   fan.
5. Do not retune thresholds mid-collection; evaluate, then change.

## Backend and web UI

```
python tools/audio/backend/app.py --target 192.168.1.50:3333
```

```
Backend starting
ESP32: 192.168.1.50:3333
Classifier: v2   results: tools/audio/results/v2
stream: connected to 192.168.1.50:3333 (16000 Hz, 512 samples/frame)
Web: http://127.0.0.1:8000
```

`--classifier v1|v2` chooses the classifier (default **v2**) and, with
it, where everything is stored: see [Classifier versions](#classifier-versions).

A long-running local service that is **the only TCP client** the ESP32
has. It owns reception, feature extraction, classification, runtime
config and event recording. The browser displays and configures; it
never classifies and never talks to the ESP32.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/status` | stream health, state, features, config |
| `GET /api/config` | current thresholds and timings |
| `PATCH /api/config` | change them live |
| `GET /api/events?page=1&pageSize=20&search=fan` | paged/searchable saved events, newest first |
| `POST /api/events/manual` | start a manual pre/post-roll capture |
| `DELETE /api/events` | delete a bundle supplied as `{ "ids": [...] }` |
| `GET /api/events/{id}` | one event's full metadata |
| `GET /api/events/{id}/audio` | that event's WAV |
| `GET /api/events/{id}/timeline` | feature timeline recomputed from the WAV |
| `PATCH /api/events/{id}/review` | save human labels in the event JSON |
| `DELETE /api/events/{id}` | delete the event JSON and WAV |
| `GET /api/history` | compact state history |
| `WS /ws/live` | snapshots, 5 Hz by default |
| `WS /ws/audio` | on-demand raw PCM for low-latency browser listening |

The classifier runs at the full window rate (~31/s); only the UI
updates are throttled. Raw PCM never leaves the backend.

The Stream card's **Listen live** control opens `/ws/audio` only while
monitoring is enabled. Each connection has a short bounded queue that
drops its oldest frame if the browser falls behind, preventing delayed
audio from accumulating. The browser resamples the 16 kHz stream via
Web Audio and applies the locally saved playback-gain preference; the
stream, event WAVs and classifier input remain unchanged. Use
headphones to prevent speaker-to-microphone feedback.

`PATCH /api/config` atomically saves the complete validated settings to
the running version's `config.json` (`results/config.json` for v1,
`results/v2/config.json` for v2) and applies them without touching the
TCP connection. The backend loads that file on its next start. The
published state is kept — moving a threshold is not an observation —
but the rolling median and candidate are cleared, so the hold has to be
earned again under the new rules.

### Transition events

Under v1, a **published** state change writes two files to `results/events/` (v2 events are described under *v2 events* above and go to `results/v2/events/`):

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
is not recorded — it is the classifier settling, not a change. The
same applies after any discontinuity: a TCP dropout, a reconnect or an
I2S gap clears the pre-roll, so the next published state is a fresh
baseline rather than the far side of a transition.

That matters because the alternative is a lie. If the air conditioner
changes while the stream is down, an event written across the gap
would claim to show FAN → COMPRESSOR while containing no audio joining
the two — only "last seen before" and "first settled after". Those
events are suppressed; the connection events in the history are what
records that something was missed.

Note that a transition publishes `holdSeconds` after the audio really
changed, so that much of the pre-roll is already the new state. At the
defaults (15 s pre-roll against a 2.5 s decision) there is plenty of
genuine "before"; raising `holdSeconds` towards `eventPreSeconds` eats
into it.

The browser's **Record event now** button starts the same pre/post-roll
capture without requiring a transition. These records have
`"source": "manual"` and use the current state for both `from` and
`to`. Every new event starts with `review.status = "unreviewed"`.
Reviewing it stores the human `actualFrom` / `actualTo`, correctness,
interference tags and notes atomically in that JSON; it never changes
the original classifier result or WAV. Older JSON files are migrated
to an unreviewed review block when indexed.

### History

One row per second in SQLite at `results/audio.db`: state, candidate,
hold, RMS, the original compact bands, 500–1k/1–2k rolling standard
deviation, spectral flux/flatness, and the frame counters. About 86k
rows a day, which SQLite does not notice. The ~31 classifier windows
per second are deliberately *not* stored — they only matter around a
transition, and that is what the event WAVs are for.

```
GET /api/history?seconds=900
GET /api/history?from=2026-09-30T13:00:00&to=2026-09-30T14:00:00
GET /api/history?seconds=86400&points=500
```

`from` and `to` accept a unix timestamp or an ISO-8601 string. The
response is columnar, because the only consumer is a chart.

**Long ranges are bucketed, not truncated.** Up to `points` (2000 by
default) rows come back as-is; beyond that the whole range is divided
into that many buckets, numeric columns averaged and state taken from
the last row in each. A `LIMIT` would have returned the first few
hours of a day and called it a day — which is worse than useless,
because it looks like data. The response says `downsampled` and
`bucketSeconds` so the UI can label it.

There is no `stream_connected` column. Rows are only written while
audio is arriving, so it was always 1 and an outage showed up as a
hole in the timestamps. Connect and disconnect are recorded properly
in a `connection` table instead, returned alongside the history and
shaded on the chart — including the event immediately *before* the
range, without which the browser cannot know how the range opened.

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

The firmware no longer links `arduinoFFT` at all, so porting features
back means adding an FFT again — `esp-dsp` is the one to reach for at
that point, not the Arduino library that was there before.

## Files

| File | Role |
| --- | --- |
| `acstream.py` | the wire protocol, shared by capture and visualize |
| `capture.py` | serial -> WAV |
| `visualize.py` | live plots and classifier |
| `analyze.py` | WAV -> feature tables and plots |
| `classify_live.py` | serial -> live OFF / FAN / COMPRESSOR (v1 only) |
| `features.py` | the one DSP implementation every other tool imports |
| `classifier/` | the rules (`v1.py`, `v2.py`) and the shared Smoother (`common.py`) |
| `evaluation.py`, `evaluate_events.py` | replay reviewed events and score both classifiers |
| `compare_models.py` | analysis only: do learned models beat the rule? |
| `recordings/` | captured WAVs, the inputs (git-ignored except `.gitkeep`) |
| `backend/` | the local service: stream, classifier, config, events, API |
| `web/` | the browser UI it serves |
| `results/` | everything generated: v1 `events/`, `audio.db`, `config.json`, … |
| `results/v2/` | the same for v2, plus `evaluation/` and `plots/` |
