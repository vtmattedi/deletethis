# Watson firmware

Watson is an **acoustic observation device**: ESP32 DevKit V1 + INMP441. It
listens and reports three independent observations over NightMareNetwork. It
does not infer whether the air conditioner is on, or whether a command worked;
the controller combines these with what it sent.

```text
acoustic:fan_detected         ManagedSensor<bool>   retained state
acoustic:compressor_detected  ManagedSensor<bool>   retained state
acoustic:beep                 ManagedEvent<String>  transient, never retained
```

The two states are independent (both may be true) and are published only when
the held detector output changes. Nothing has a value until its first hold
elapses. A beep is one Event per accepted tone, with a JSON payload:

```json
{"event_id":4,"timestamp_ms":50162,"peak_hz":4117.2,"duration_ms":192,"contrast_db":20.4,"level_db":-62.8}
```

`event_id` counts accepted beeps this boot; `timestamp_ms` is uptime when the
tone ended. NightMareNetwork's Event codecs are the built-in ones (applications
do not define their own), so the semantic `BeepPayload` struct travels as JSON
in a `ManagedEvent<String>`.

## Build, flash, monitor

```text
pio run -t upload --upload-port COM18      # USB
pio device monitor -p COM18                # 921600 baud
pio run -e esp32dev-notcp                  # same firmware, no raw PCM server
```

`include/creds.h` (gitignored; template `creds.example.h`) holds the Wi-Fi and
broker credentials NightMareNetwork reads. Platform: pioarduino 55.03.39
(Arduino 3.3.9 / ESP-IDF 5.5). Partitions: `src/OtaDual.csv` (two 1.9 MB OTA
slots; the image is ~1.4 MB). USB flashing always works; OTA is additional.

Serial is the NightMare console (`CONFIG LIST`, `> acoustic:fan_detected`, ...)
plus two Watson commands. It never carries audio.

```text
WATSON STATS      acquisition, analysis, detector, TCP and heap counters
WATSON SELFTEST   golden windows through this device's FFT vs the float64 reference
```

## Pipeline

```text
INMP441 --I2S DMA--> acquisition task (core 0, prio 12)
                          |--> analysis queue (8 x 512 samples) --> analysis task (core 1, prio 4)
                          |                                            WatsonCore: window -> features
                          |                                            -> median -> fan / compressor / beep
                          |                                                  |
                          `--> TCP queue (only while a client is connected)    `--> atomics + a small queue
                                   --> stream task (core 1, prio 3)                      |
                                                                          loop(): tickNightMareESP + pump()
                                                                          = the only NightMare callers
```

* **Acquisition never waits.** It reads 512-sample blocks (32 ms = one hop = one
  TCP frame) and pushes them to each consumer's queue. A full queue *refuses*
  the block, counts it, and the next block that gets in carries the loss. I2S
  DMA overflows are counted from the driver's overflow callback. There is no
  heap allocation, logging or network call in that path.
* **Analysis** reads a 2048-sample window in place across the four oldest queued
  blocks and releases one block per window (75 % overlap), so audio is never
  copied or held twice. It takes ~3.5 ms per window (11 % of a core).
  Features come from arduinoFFT (single precision); the real 2048-point
  transform is done as a packed 1024-point complex one and unpacked, which
  halves the scratch memory.
* **A hole in the audio** (blocks refused, or an I2S overrun) drops everything
  being built across it: sample window, stationarity history, medians, hold
  candidates and any beep run. Published fan/compressor values stay; each
  re-earns its hold. A beep spanning the hole is discarded, and no old beep is
  replayed.
* **NightMare** is only touched from `loop()`; the analysis task hands over
  state through atomics and beeps through a queue.

### Memory is the constraint

Wi-Fi and the TLS session to the broker need most of the ~210 KB of
byte-addressable internal RAM. What the audio path costs, and how it was kept
small:

* queue slots and the Hamming table live in the ESP32's *instruction-RAM heap*
  (~37 KB that normal allocations cannot use). They are integer words only: the
  FPU's float load/store cannot address that memory, so the FFT scratch stays in
  normal RAM and the window is read as integer bits. `AudioBlock` has only
  32-bit fields for the same reason.
* the TCP debug server allocates its queue, frame buffer and task only while a
  client is connected, refuses clients unless 24 KB of RAM would remain, and
  drops a client if free RAM falls below 9 KB.
* task stacks are sized from measured high-water marks (`WATSON STATS`).
* steady state after the broker connects: ~35 KB free (largest block ~24 KB).

## Detectors

Ported from the PC v2 detectors, with the same feature math (tools/audio/
features.py) at 2048 Hamming / hop 512 (7.8125 Hz/bin, 32 ms cadence).

| | rule |
|---|---|
| fan | median(500-1k) >= -62 **and** median(1k-2k) >= -65 **and** median(1k-2k std) <= 6.0 dB **and** >= 1.0 s of history. Own 2 s hold. |
| compressor | median(**52-65 Hz**) >= **-43.0 dB**. Optional sideband guard: `primary - lower(30-45) >= m1` and `primary - upper(65-80) >= m2`, **disabled by default**. Own 2 s hold. |
| beep | per-window raw features (never median-smoothed): peak 4050-4180 Hz, contrast >= 10 dB to extend a run, run's best contrast >= 15 dB, level >= -80 dBFS, 60-300 ms, peak spread <= 50 Hz, <= 40 ms gaps, 80 ms refractory. One Event per accepted run; rejections counted as weak / too_short / too_long / unstable_pitch. |

Fan and compressor share the median stage and nothing else; neither can
suppress or imply the other.

### Why 52-65 Hz and -43 dB

The old PC feature was the whole 30-80 Hz band at -38 dB. Narrowing the band
changes the power it integrates, so the threshold was re-derived over all
labelled audio with the firmware's own feature math
([validation/results/compressor_derivation.md](../validation/results/compressor_derivation.md)):
balanced accuracy 0.976 (recall 96.2 %, false positives 1.0 %); the plateau of
near-optimal thresholds is -50 ... -41 dB and a leave-one-group-out search over
87 time-groups picks -44 ... -42. The sideband deltas for calibrating the guard
are in the same file (true compressor: primary-lower p5 15.7 dB, primary-upper
p5 17.2 dB; false positives are much smaller).

### Why the beep detector uses a 64 ms footprint

The PC detector models a tone's extent with one "window" length, calibrated for
a 64 ms window. A 128 ms Hamming window only flags about as much extra length
(its outer quarters carry almost no weight), so the firmware keeps the PC's
64 ms footprint (`kBeepFootprintMs`). With it, all 69 beeps the PC finds in the
21 reviewed beep events are reproduced with no extras; with the true 128 ms,
only 44 are
([beep_calibration.md](../validation/results/beep_calibration.md)).

## Configs

All persistent, validated before they reach a detector (min < max, edge contrast
<= peak contrast, bands below Nyquist, finite numbers, bounded windows), and
applied between windows without restarting I2S. A change resets only what it
affects: thresholds nothing; the median window the median rings and both hold
candidates; the history length the fan's stationarity; the compressor band the
compressor's ring; beep settings the beep run. Published values are never reset
by a settings change. Because each write is checked against the *current* other
values, move a band by changing its max first when raising it, its min first
when lowering it.

```text
acoustic:median_ms 500    acoustic:hold_ms 2000    acoustic:history_ms 2000
acoustic:compressor:band_min_hz 52   :band_max_hz 65   :threshold_db -43
acoustic:compressor:sidebands:enable false
acoustic:compressor:sidebands:thresholds "0,0"      # "lower_margin_db,upper_margin_db"
acoustic:fan:mid_threshold_db -62   :high_threshold_db -65   :require_both true
acoustic:fan:stability_threshold_db 6.0   :stability_min_ms 1000
acoustic:beep:min_hz 4050 :max_hz 4180 :left_min_hz 3850 :left_max_hz 4000
              :right_min_hz 4230 :right_max_hz 4380 :edge_contrast_db 10
              :min_contrast_db 15 :min_level_db -80 :min_duration_ms 60
              :max_duration_ms 300 :max_peak_spread_hz 50 :max_gap_ms 40
              :refractory_ms 80
```

The sideband margins `0,0` are deliberately neutral and have no effect while
`sidebands:enable` is false. Before enabling the guard, replace them with
margins validated on the dataset. The measured compressor fundamental is
~58.6 Hz (7.5 bins); lower sideband 30-45, lower shoulder 45-52 (diagnostic
only), primary 52-65, upper sideband 65-80.

NightMare's Config codecs have no pair type, so `sidebands:thresholds` is a
validated `Config<String>` in the canonical `lower,upper` form (no whitespace).

Structural parameters (sample rate, FFT size, hop, DMA geometry) are compile-time
constants in `src/audio/AudioConfig.h`.

## Raw PCM debug server (`ENABLE_TCP`)

Debug and calibration only. Port 3333, one client, 16 kHz mono, 32-bit PCM
(24 valid bits, right-aligned), with exactly the `ACD1` stream header and
`ACDF` frames that `tools/audio/acstream.py` reads; the PC tools connect
unchanged. The server never reads from the client. It consumes from its own
queue on its own task; if the client or link cannot keep up, blocks are dropped
(reported in the next frame's `dropped`) and acquisition never notices.

Build it out with `-DENABLE_TCP=0` (`pio run -e esp32dev-notcp`).

## Microphone presence (hardware flag)

`fan_detected` and `compressor_detected` declare NightMare's hardware policy
with `REPORT_HW_CONNECTION`, so their manifest entries carry
`hardware.connected`. GPIO33 (INMP441 SD) has the internal pulldown enabled, so
an unplugged or unpowered microphone reads as constant zero instead of floating
noise. A block whose 24-bit samples span <= 8 LSB is "flat" (a live INMP441 is
hundreds of LSB even in a silent room); 32 flat blocks in a row (1.02 s) mark the
hardware disconnected, 16 live blocks in a row (0.5 s) mark it back.

While disconnected the sensors keep their last value, `hardware.connected` is
false, and nothing the detectors conclude from silence is published. On both
transitions the analysis drops its history and forgets the published values, so
the first value after the microphone returns is earned fresh (hold included).
`WATSON SIMFLAT ON|OFF` makes the acquisition see a dead microphone, to exercise
this without unplugging anything; `WATSON STATS` has a `microphone:` line.
Host tests: `replay --flatline-selftest` (debounce boundaries, stuck-high line)
and `replay --flatline <pcm>` (87 real recordings: 0 disconnects).

## Connection: ESP-NOW preferred

`NM_NETWORK_ESPNOW 1`. ESP-NOW (to the NightMare gateway) is the preferred
connection and the remote TLS broker is the failover, used when no gateway
answers within `nightmare:connection:failover_secs` (60 s). The preference is
applied once on first boot (marker Config `acoustic:connection:espnow_preferred_applied`),
so a later `CONFIG SET nightmare:connection:preferred_connection` is respected.
`creds.h` needs `NM_ESPNOW_PSK`, the gateway's network key (16-64 bytes); the
shipped value is a placeholder and must be replaced.

NightMare runs the Wi-Fi station only while an MQTT profile is selected, so
**while ESP-NOW is connected there is no IP link**: SNTP, OTA and the raw PCM
debug server are unavailable until the device fails over (or you run
`NETWORK SET MQTT`). ESP-NOW also adds tasks (client, rx log, worker), which the
~35 KB heap budget below has to absorb: untested on hardware.

## Validation

```text
validation/host/build.sh      # builds the firmware core + real arduinoFFT on the host (Docker gcc)
python validation/parity.py   # firmware core vs float64 reference over the labelled audio
python validation/replay_eval.py
python validation/derive_compressor.py
python validation/calibrate_beep.py
python validation/make_golden.py   # regenerates src/audio/GoldenVectors.h
```

Results (committed in `validation/results/`):

* **Feature parity**, 112 files: band levels agree with the float64 reference to
  ~0.001 dB, the fan std to 0.0001 dB, beep peak/contrast/level to 0.004 Hz/dB,
  with **zero** fan/compressor candidate mismatches and identical published
  change times and beeps.
* **On-device**: `WATSON SELFTEST` runs four recorded windows through the
  ESP32's own FFT; worst difference 0.018 dB (tolerance 0.05 dB).
* **Labelled replay** (117 files) vs the PC v2 detectors: fan recall 0.968 vs
  0.967, false positives 6.6 % vs 5.8 %; compressor recall 0.961 vs 0.769 at
  1.2 % false positives for both ([replay_eval.md](../validation/results/replay_eval.md)).
* **Gaps**: a 3- or 40-block hole leaves a steady fan's published value
  untouched (one discontinuity, no change events); a beep spanning a hole is
  discarded; a hole far from a beep does not affect it.
* **Hardware beep test**: five 4118 Hz tones played at the microphone produced
  exactly five Events (ids 1-5, peak 4117-4118 Hz, ~1.34 s apart, not retained).
* **Acquisition under load** (idle, a streaming TCP client, Wi-Fi reconnects,
  Config writes, serial monitor): **0 I2S overruns after start-up**; the worst
  case is a ~85 ms late block while Config values are written to flash.

## Known limits

* **Start-up**: NightMare's start-up does flash work that stalls every task for
  ~200 ms at about 1.7 s (5 I2S overruns). I2S starts after it, but the PHY
  calibration stall still lands in that window; the `steady:` line in `WATSON
  STATS` counts from 15 s.
* **OTA**: works (one upload completed on a clean link); other uploads from the development PC aborted mid-transfer (60-80 %) over this
  room's Wi-Fi link even with the audio stack off. The espota host protocol is
  stop-and-wait with a 10 s timeout and the link showed 10-20 % ping loss at
  times. Flash writes during an update stall the instruction cache, so audio is
  lost in bursts that no scheduling can prevent on this chip; an update
  therefore pauses acquisition (maintenance mode) and resumes with a reported
  gap if it fails. USB flashing is unaffected. Re-test OTA on a cleaner link.
* **Debug stream over a weak link**: three or four clean connections are typical
  on this link, after which retransmissions can stall a session. It recovers
  fully and never affects acquisition.
* **Internal RAM** leaves ~35 KB free with the broker connected. A TLS
  handshake needs most of that transiently, so features that grow heap use
  (more Resources, big Config sets, long-lived TCP clients) should be added
  with that in mind.
