"""Live view of the ESP32/INMP441 stream.

    python tools/audio/visualize.py COM13

Shows, updating in real time:

  * the waveform of the last second or so
  * the current FFT spectrum in dBFS
  * RMS / dBFS and the dominant peaks above 100 Hz
  * a scrolling spectrogram

The spectrogram is the panel to watch. A single FFT dump tells you
very little, but a few seconds of scrolling history makes mains hum,
a steady tone, a fan spinning up and a compressor kicking in visually
obvious.

Serial reading runs on its own thread, so a slow redraw stalls the
plot rather than the stream.

``--classify`` (on by default) also runs the live OFF / FAN /
COMPRESSOR classifier from classify_live.py and shows its verdict, and
``--play`` sends the audio to the speakers so you can hear what the
microphone hears while watching the spectrogram.

Both are fed from the reader thread, not from the plot. The plot queue
drops frames when a redraw falls behind, which would put holes in the
audio and corrupt the classifier's rolling median and hold timing;
those two need every frame, in order. Playback itself never blocks the
reader -- see Player.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks

from classify_live import (
    COMPRESSOR,
    DEFAULT_COMPRESSOR_THRESHOLD,
    DEFAULT_FAN_HIGH_THRESHOLD,
    DEFAULT_FAN_MID_THRESHOLD,
    DEFAULT_FAN_REQUIRE,
    DEFAULT_HOLD_SECONDS,
    DEFAULT_MEDIAN_SECONDS,
    FAN,
    OFF,
    Decision,
    FeatureExtractor,
    Smoother,
    Thresholds,
)

# Playback is optional: the viewer is useful without a sound card, and
# sounddevice needs PortAudio, which is not always present.
try:
    import sounddevice
except (ImportError, OSError) as _sounddevice_error:  # pragma: no cover
    sounddevice = None
    SOUNDDEVICE_ERROR = str(_sounddevice_error)
else:
    SOUNDDEVICE_ERROR = ""

from acstream import (
    DEFAULT_BAUD,
    FULL_SCALE,
    AudioStream,
    ProtocolError,
    dbfs,
    list_ports,
)

# Anything quieter than this is drawn as the floor colour.
DEFAULT_DB_FLOOR = -100.0
DEFAULT_DB_CEILING = -20.0

# The microphone floor is far below full scale, so a fixed +-1.0
# waveform axis would draw every real signal as a flat line.
WAVEFORM_MIN_SCALE = 1.0 / (1 << 10)

# Peaks below this are mains hum and handling noise, not signal.
PEAK_MIN_HZ = 100.0

REDRAW_INTERVAL_MS = 60

# The INMP441 runs tens of dB below full scale, so raw playback is
# inaudible. This is a listening aid, not a calibrated monitor.
DEFAULT_GAIN_DB = 24.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-time waveform / FFT / spectrogram viewer.",
    )

    parser.add_argument("port", nargs="?", help="Serial port, e.g. COM13")

    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"Serial baud rate (default: {DEFAULT_BAUD})",
    )

    parser.add_argument(
        "--nfft",
        type=int,
        default=1024,
        help="FFT size, matching the firmware default of 1024",
    )

    parser.add_argument(
        "--waveform-seconds",
        type=float,
        default=0.5,
        help="Waveform window length (default: 0.5)",
    )

    parser.add_argument(
        "--history-seconds",
        type=float,
        default=15.0,
        help="Spectrogram history length (default: 15)",
    )

    parser.add_argument(
        "--fmax",
        type=float,
        default=None,
        help="Highest frequency to plot (default: Nyquist)",
    )

    parser.add_argument(
        "--db-floor",
        type=float,
        default=DEFAULT_DB_FLOOR,
        help=f"Spectrogram colour floor (default: {DEFAULT_DB_FLOOR})",
    )

    parser.add_argument(
        "--db-ceiling",
        type=float,
        default=DEFAULT_DB_CEILING,
        help=f"Spectrogram colour ceiling (default: {DEFAULT_DB_CEILING})",
    )

    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List available serial ports and exit",
    )

    audio = parser.add_argument_group("playback")

    audio.add_argument(
        "--play",
        action="store_true",
        help=(
            "Play the incoming audio on the speakers. Use headphones "
            "if the microphone can hear them, or it will feed back"
        ),
    )

    audio.add_argument(
        "--gain-db",
        type=float,
        default=DEFAULT_GAIN_DB,
        help=(
            "Playback gain. The microphone runs far below full scale, "
            f"so some is needed to hear anything (default: "
            f"{DEFAULT_GAIN_DB:g})"
        ),
    )

    audio.add_argument(
        "--audio-device",
        default=None,
        help="Output device index or name (default: system default)",
    )

    audio.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="List output devices and exit",
    )

    audio.add_argument(
        "--audio-test",
        action="store_true",
        help=(
            "Play a test tone through the playback path and exit. "
            "Needs no ESP32: use it to tell a broken speaker path "
            "from a quiet room"
        ),
    )

    rules = parser.add_argument_group("classifier")

    rules.add_argument(
        "--no-classify",
        dest="classify",
        action="store_false",
        help="Do not run the live classifier",
    )

    rules.add_argument(
        "--compressor-threshold",
        type=float,
        default=DEFAULT_COMPRESSOR_THRESHOLD,
        help=f"(default: {DEFAULT_COMPRESSOR_THRESHOLD:g})",
    )

    rules.add_argument(
        "--fan-mid-threshold",
        type=float,
        default=DEFAULT_FAN_MID_THRESHOLD,
        help=f"(default: {DEFAULT_FAN_MID_THRESHOLD:g})",
    )

    rules.add_argument(
        "--fan-high-threshold",
        type=float,
        default=DEFAULT_FAN_HIGH_THRESHOLD,
        help=f"(default: {DEFAULT_FAN_HIGH_THRESHOLD:g})",
    )

    rules.add_argument(
        "--fan-require",
        choices=["either", "both"],
        default=DEFAULT_FAN_REQUIRE,
        help=f"(default: {DEFAULT_FAN_REQUIRE})",
    )

    rules.add_argument(
        "--hold-seconds",
        type=float,
        default=DEFAULT_HOLD_SECONDS,
        help=(
            "How long a candidate must persist before the state "
            f"changes (default: {DEFAULT_HOLD_SECONDS:g})"
        ),
    )

    return parser


def audio_test(args: argparse.Namespace) -> int:
    """Play a tone through the real playback path, and say what happened.

    This exists because silence is ambiguous. The microphone runs tens
    of dB below full scale, so "I hear nothing" can mean the speaker
    path is broken or it can mean the room is quiet. A tone through the
    same Player, with the same device and gain, separates the two: if
    you hear this, playback works and any later silence is the signal.
    """
    seconds = 1.5
    rate = 16000

    player = Player(
        rate,
        device=args.audio_device,
        gain_db=args.gain_db,
    )

    if not player.start():
        print(f"Playback FAILED: {player.error}", file=sys.stderr)
        print(
            "\nTry --list-audio-devices and pass one with "
            "--audio-device.",
            file=sys.stderr,
        )
        return 1

    device = (
        player.stream.device if player.stream is not None else "default"
    )

    print(
        f"Playing a 440 Hz tone for {seconds:g}s on device {device} "
        f"at {rate} Hz."
    )
    print(
        f"The tone is written at the same level the microphone "
        f"typically reaches, then amplified by {args.gain_db:g} dB, "
        "so it is about as loud as the air conditioner will be."
    )

    if not player.wait_until_running():
        print(
            "The device never asked for audio. The stream opened but "
            "is not running.",
            file=sys.stderr,
        )
        player.stop()
        return 1

    # Ignore whatever the cold start cost; measure the steady state.
    player.reset_counters()

    # -40 dBFS, roughly what the INMP441 produces in a quiet room.
    amplitude = 10.0 ** (-40.0 / 20.0)

    chunk = 512
    total = int(seconds * rate)
    phase = 0

    try:
        # Paced against an absolute deadline, as the ESP32 paces itself.
        # Sleeping chunk/rate each time instead would accumulate the
        # Windows timer's ~15 ms granularity into a drift of its own,
        # and this test would report drops that the real stream does
        # not have.
        started = time.monotonic()

        while phase < total:
            index = np.arange(phase, min(phase + chunk, total))

            player.push(
                amplitude * np.sin(2 * np.pi * 440.0 * index / rate)
            )

            phase += chunk

            delay = started + phase / rate - time.monotonic()

            if delay > 0:
                time.sleep(delay)

        # Let the tail play out, then snapshot before the empty ring
        # starts counting silence that is just the end of the tone.
        drain_by = time.monotonic() + 1.0

        while (
            player.buffered_samples() > 0
            and time.monotonic() < drain_by
        ):
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        status = player.status()
        player.stop()

    print()
    print(status)
    print()

    if player.underruns > rate * 0.1:
        print(
            "The device asked for more audio than was supplied, which "
            "means the tone was broken up. That is a buffering problem, "
            "not a routing one."
        )
    else:
        print(
            "If you heard a steady tone, playback works: run the viewer "
            "with --play and any silence is the signal, not the path."
        )
        print(
            "If you heard nothing, check the output device "
            "(--list-audio-devices) and the system volume, or raise "
            "--gain-db."
        )

    return 0


def list_audio_devices() -> int:
    if sounddevice is None:
        print(
            f"sounddevice unavailable: {SOUNDDEVICE_ERROR}",
            file=sys.stderr,
        )
        return 1

    for index, device in enumerate(sounddevice.query_devices()):
        if device["max_output_channels"] < 1:
            continue

        print(
            f"{index:3d}  {device['name']}  "
            f"({device['max_output_channels']} ch, "
            f"{device['default_samplerate']:.0f} Hz)"
        )

    return 0


STATE_COLOURS = {
    OFF: "#3b7dd8",
    FAN: "#1f9d55",
    COMPRESSOR: "#d2691e",
}


class Player:
    """Speaker playback that can never stall the serial reader.

    ``OutputStream.write()`` blocks whenever the device falls behind,
    which would back up into the serial reader and start losing frames
    -- the plot and the classifier would suffer to keep the speaker
    fed. So nothing here ever waits on the sound card:

    * the reader thread only copies samples into a ring buffer;
    * PortAudio's own callback thread drains it.

    The two clocks are independent -- the ESP32's 16 kHz and the sound
    card's -- so they drift. The ring absorbs that: if the card runs
    slow the oldest audio is dropped, if it runs fast the callback
    emits silence. Both are counted and shown, because a large count
    means the sound you are hearing is not continuous.
    """

    def __init__(
        self,
        sample_rate: int,
        device: str | int | None = None,
        gain_db: float = 0.0,
        buffer_seconds: float = 2.4,
    ) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self.gain = 10.0 ** (gain_db / 20.0)

        self.capacity = max(1024, int(buffer_seconds * sample_rate))
        self.buffer = np.zeros(self.capacity, dtype=np.float32)

        self.lock = threading.Lock()
        self.read_pos = 0
        self.count = 0

        self.dropped = 0      # samples overwritten, device too slow
        self.underruns = 0    # samples of silence, device too fast
        self.clipped = 0      # samples past full scale, gain too high

        # Set the first time the device asks for audio. Opening a
        # stream returns before the device is actually pulling, and on
        # a cold start that can take a few hundred ms -- long enough to
        # overflow the ring and look like a fault that is not there.
        self.running = threading.Event()

        self.stream: "sounddevice.OutputStream | None" = None
        self.error: str = ""

    # ------------------------------------------------------- lifecycle

    def start(self) -> bool:
        if sounddevice is None:
            self.error = (
                f"sounddevice unavailable ({SOUNDDEVICE_ERROR})"
            )
            return False

        try:
            self.stream = sounddevice.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.device,
                callback=self._callback,
                # Let PortAudio pick its own block size; a fixed one
                # only adds latency here.
                blocksize=0,
            )
            self.stream.start()
        except Exception as error:      # device busy, bad rate, ...
            self.error = str(error)
            self.stream = None
            return False

        return True

    def stop(self) -> None:
        if self.stream is None:
            return

        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        finally:
            self.stream = None

    # ---------------------------------------------------------- audio

    def push(self, samples: np.ndarray) -> None:
        """Called from the reader thread. Never blocks."""
        if self.stream is None:
            return

        block = np.asarray(samples, dtype=np.float32) * np.float32(
            self.gain
        )

        clipped = int(np.count_nonzero(np.abs(block) > 1.0))

        if clipped:
            np.clip(block, -1.0, 1.0, out=block)

        size = block.size

        if size > self.capacity:
            block = block[-self.capacity :]
            size = self.capacity

        with self.lock:
            self.clipped += clipped

            overflow = self.count + size - self.capacity

            if overflow > 0:
                # The card is behind. Drop the oldest audio rather than
                # wait: staying live matters more than completeness.
                self.read_pos = (self.read_pos + overflow) % self.capacity
                self.count -= overflow
                self.dropped += overflow

            start = (self.read_pos + self.count) % self.capacity
            first = min(size, self.capacity - start)

            self.buffer[start : start + first] = block[:first]

            if size > first:
                self.buffer[: size - first] = block[first:]

            self.count += size

    def _callback(self, outdata, frames, _time, _status) -> None:
        """Runs on PortAudio's thread. Must not block or allocate much."""
        self.running.set()

        with self.lock:
            available = min(frames, self.count)

            if available:
                first = min(available, self.capacity - self.read_pos)

                outdata[:first, 0] = self.buffer[
                    self.read_pos : self.read_pos + first
                ]

                if available > first:
                    outdata[first:available, 0] = self.buffer[
                        : available - first
                    ]

                self.read_pos = (
                    self.read_pos + available
                ) % self.capacity
                self.count -= available

            if available < frames:
                outdata[available:, 0] = 0.0
                self.underruns += frames - available

    def wait_until_running(self, timeout: float = 2.0) -> bool:
        """Block until the device starts pulling audio.

        Only for start-up and diagnostics, never on the reader path.
        """
        return self.running.wait(timeout)

    def buffered_samples(self) -> int:
        with self.lock:
            return self.count

    def reset_counters(self) -> None:
        with self.lock:
            self.dropped = 0
            self.underruns = 0
            self.clipped = 0

    def status(self) -> str:
        with self.lock:
            buffered = self.count / self.sample_rate
            dropped = self.dropped
            underruns = self.underruns
            clipped = self.clipped

        return (
            f"audio   ON  {buffered * 1000:4.0f} ms buffered   "
            f"late {dropped / self.sample_rate:.1f}s   "
            f"silence {underruns / self.sample_rate:.1f}s   "
            f"clipped {clipped}"
        )


class StreamReader(threading.Thread):
    """Pulls frames off the serial port into a queue."""

    def __init__(
        self,
        stream: AudioStream,
        player: Player | None = None,
        extractor: FeatureExtractor | None = None,
        smoother: Smoother | None = None,
    ) -> None:
        super().__init__(daemon=True)

        self.stream = stream
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)
        self.stop_event = threading.Event()
        self.error: Exception | None = None

        # Playback and classification are driven from here rather than
        # from the plot. The plot queue deliberately drops frames when
        # a redraw falls behind, which would put holes in the audio and
        # break the classifier's rolling median and hold timing. Both
        # of those need every frame, in order.
        self.player = player
        self.extractor = extractor
        self.smoother = smoother

        self.decision: Decision | None = None
        self.decision_lock = threading.Lock()

        self.missing = stream.lost_frames + stream.dropped_frames
        self.gaps = 0

    def run(self) -> None:
        try:
            for frame in self.stream.frames():
                if self.stop_event.is_set():
                    return

                self._handle_gap()

                if self.player is not None:
                    self.player.push(frame.samples / FULL_SCALE)

                self._classify(frame.samples)

                try:
                    self.queue.put_nowait(frame.samples)
                except queue.Full:
                    # The plot is behind. Dropping the oldest audio is
                    # the right trade for a viewer: stay live.
                    try:
                        self.queue.get_nowait()
                        self.queue.put_nowait(frame.samples)
                    except queue.Empty:
                        pass
        except Exception as error:  # surfaced on the plot thread
            self.error = error

    def _handle_gap(self) -> None:
        """Discard state built across a hole in the stream."""
        now_missing = (
            self.stream.lost_frames + self.stream.dropped_frames
        )

        if now_missing <= self.missing:
            return

        lost = now_missing - self.missing
        self.missing = now_missing
        self.gaps += 1

        info = self.stream.info

        if self.extractor is not None and info is not None:
            self.extractor.reset(
                skip_samples=lost * info.frame_samples
            )

        if self.smoother is not None:
            self.smoother.reset()

    def _classify(self, samples: np.ndarray) -> None:
        if self.extractor is None or self.smoother is None:
            return

        decision = None

        for features in self.extractor.push(samples):
            decision = self.smoother.update(features)

        if decision is not None:
            with self.decision_lock:
                self.decision = decision

    def latest_decision(self) -> Decision | None:
        with self.decision_lock:
            return self.decision

    def stop(self) -> None:
        self.stop_event.set()


class Viewer:
    def __init__(self, args: argparse.Namespace, stream: AudioStream) -> None:
        info = stream.info
        assert info is not None

        self.args = args
        self.stream = stream
        self.sample_rate = info.sample_rate
        self.nfft = args.nfft

        self.fmax = args.fmax or self.sample_rate / 2.0

        self.window = np.hamming(self.nfft)
        # Coherent gain, so a tone reads at its true amplitude.
        self.window_gain = self.window.sum() / self.nfft

        self.freqs = np.fft.rfftfreq(self.nfft, 1.0 / self.sample_rate)
        self.fmax_bin = int(np.searchsorted(self.freqs, self.fmax)) + 1
        self.fmax_bin = min(self.fmax_bin, len(self.freqs))

        self.waveform_length = max(
            self.nfft,
            int(args.waveform_seconds * self.sample_rate),
        )

        self.ring = np.zeros(self.waveform_length, dtype=np.float64)

        columns = max(
            2,
            int(
                args.history_seconds
                * self.sample_rate
                / info.frame_samples
            ),
        )

        self.spectrogram = np.full(
            (self.fmax_bin, columns),
            args.db_floor,
            dtype=np.float64,
        )

        self.history_seconds = (
            columns * info.frame_samples / self.sample_rate
        )

        self.player: Player | None = None

        # Shown in the readout whatever happens. Playback that silently
        # is not running looks exactly like playback that is running
        # with nothing to hear, and there is no way to tell from the
        # window which one you have.
        self.audio_status = "audio   off          (--play to enable)"

        if args.play:
            self.player = Player(
                self.sample_rate,
                device=args.audio_device,
                gain_db=args.gain_db,
            )

            if self.player.start():
                device = (
                    self.player.stream.device
                    if self.player.stream is not None
                    else "default"
                )

                print(
                    f"Playback on device {device} at "
                    f"{self.sample_rate} Hz, {args.gain_db:g} dB gain."
                )
            else:
                self.audio_status = (
                    f"audio   FAILED       {self.player.error}"
                )

                print(
                    f"warning: playback disabled: {self.player.error}",
                    file=sys.stderr,
                )

                self.player = None

        extractor = None
        smoother = None

        if args.classify:
            extractor = FeatureExtractor(
                self.sample_rate,
                self.nfft,
                0.5,
            )

            smoother = Smoother(
                Thresholds(
                    compressor=args.compressor_threshold,
                    fan_mid=args.fan_mid_threshold,
                    fan_high=args.fan_high_threshold,
                    fan_require_both=args.fan_require == "both",
                ),
                window_rate=self.sample_rate / extractor.hop,
                median_seconds=DEFAULT_MEDIAN_SECONDS,
                hold_seconds=args.hold_seconds,
            )

        self.reader = StreamReader(
            stream,
            player=self.player,
            extractor=extractor,
            smoother=smoother,
        )

        self._build_figure()

    # ---------------------------------------------------------- figure

    def _build_figure(self) -> None:
        self.figure = plt.figure(figsize=(12, 8))
        self.figure.canvas.manager.set_window_title(
            f"INMP441 live view - {self.args.port}"
        )

        grid = GridSpec(
            3,
            2,
            figure=self.figure,
            height_ratios=[1, 1, 1.4],
            hspace=0.45,
            wspace=0.2,
        )

        # --- waveform -------------------------------------------------
        self.wave_axis = self.figure.add_subplot(grid[0, 0])

        wave_time = (
            np.arange(self.waveform_length) / self.sample_rate
        )

        (self.wave_line,) = self.wave_axis.plot(
            wave_time,
            self.ring,
            linewidth=0.7,
        )

        self.wave_axis.set_xlim(0, wave_time[-1])
        self.wave_axis.set_ylim(-WAVEFORM_MIN_SCALE, WAVEFORM_MIN_SCALE)
        self.wave_axis.set_title("Waveform")
        self.wave_axis.set_xlabel("s")
        self.wave_axis.set_ylabel("amplitude")
        self.wave_axis.grid(alpha=0.3)

        # --- spectrum -------------------------------------------------
        self.spectrum_axis = self.figure.add_subplot(grid[0, 1])

        (self.spectrum_line,) = self.spectrum_axis.plot(
            self.freqs[: self.fmax_bin],
            np.full(self.fmax_bin, self.args.db_floor),
            linewidth=0.8,
        )

        (self.peak_markers,) = self.spectrum_axis.plot(
            [], [], "v", markersize=6,
        )

        self.spectrum_axis.set_xlim(0, self.fmax)
        self.spectrum_axis.set_ylim(
            self.args.db_floor, self.args.db_ceiling + 20
        )
        self.spectrum_axis.set_title(f"Spectrum (N={self.nfft}, Hamming)")
        self.spectrum_axis.set_xlabel("Hz")
        self.spectrum_axis.set_ylabel("dBFS")
        self.spectrum_axis.grid(alpha=0.3)

        # --- readout --------------------------------------------------
        self.text_axis = self.figure.add_subplot(grid[1, :])
        self.text_axis.axis("off")

        # Stacked, not side by side: the readout lines are wide enough
        # that a second column would overlap them.
        self.state_text = self.text_axis.text(
            0.0,
            1.0,
            "",
            family="monospace",
            fontsize=13,
            fontweight="bold",
            va="top",
        )

        self.readout = self.text_axis.text(
            0.0,
            0.62,
            "",
            family="monospace",
            fontsize=9.5,
            va="top",
        )

        # --- spectrogram ----------------------------------------------
        self.spectrogram_axis = self.figure.add_subplot(grid[2, :])

        self.spectrogram_image = self.spectrogram_axis.imshow(
            self.spectrogram,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            vmin=self.args.db_floor,
            vmax=self.args.db_ceiling,
            extent=(-self.history_seconds, 0.0, 0.0, self.fmax),
        )

        self.spectrogram_axis.set_title("Spectrogram")
        self.spectrogram_axis.set_xlabel("s (now at right)")
        self.spectrogram_axis.set_ylabel("Hz")

        self.figure.colorbar(
            self.spectrogram_image,
            ax=self.spectrogram_axis,
            label="dBFS",
            pad=0.01,
        )

    # ----------------------------------------------------------- update

    def _spectrum_db(self, block: np.ndarray) -> np.ndarray:
        windowed = block * self.window

        spectrum = np.fft.rfft(windowed)

        magnitude = np.abs(spectrum) / (self.nfft * self.window_gain)

        # Single-sided: everything but DC and Nyquist doubles.
        magnitude[1:-1] *= 2.0

        return 20.0 * np.log10(np.maximum(magnitude, 1e-12))

    def _dominant_peaks(
        self, spectrum_db: np.ndarray, count: int = 3
    ) -> list[tuple[float, float]]:
        first_bin = int(np.searchsorted(self.freqs, PEAK_MIN_HZ))

        region = spectrum_db[first_bin : self.fmax_bin]

        if region.size == 0:
            return []

        indices, _ = find_peaks(region, prominence=6.0)

        if indices.size == 0:
            indices = np.array([int(np.argmax(region))])

        order = np.argsort(region[indices])[::-1][:count]

        return [
            (
                float(self.freqs[first_bin + indices[i]]),
                float(region[indices[i]]),
            )
            for i in order
        ]

    def update(self, _frame: int):
        if self.reader.error is not None:
            plt.close(self.figure)
            return ()

        pushed = 0

        while True:
            try:
                samples = self.reader.queue.get_nowait()
            except queue.Empty:
                break

            self._push(samples)
            pushed += 1

            # Never redraw more than a screenful of backlog at once.
            if pushed >= self.spectrogram.shape[1]:
                break

        if pushed == 0:
            return ()

        block = self.ring[-self.nfft :].copy()
        block -= block.mean()

        spectrum_db = self._spectrum_db(block)

        rms_counts = float(np.sqrt(np.mean(block * block))) * FULL_SCALE
        peak_counts = float(np.max(np.abs(block))) * FULL_SCALE

        peaks = self._dominant_peaks(spectrum_db)

        self.wave_line.set_ydata(self.ring)
        self._rescale_waveform()

        self.spectrum_line.set_ydata(spectrum_db[: self.fmax_bin])

        if peaks:
            self.peak_markers.set_data(
                [peak[0] for peak in peaks],
                [peak[1] + 3.0 for peak in peaks],
            )
        else:
            self.peak_markers.set_data([], [])

        self.spectrogram_image.set_data(self.spectrogram)

        peak_text = "   ".join(
            f"{frequency:7.1f} Hz {level:6.1f} dB" for frequency, level in peaks
        ) or "none"

        lines = [
            f"RMS  {rms_counts:10.0f} counts   {dbfs(rms_counts):7.2f} dBFS",
            f"Peak {peak_counts:10.0f} counts   {dbfs(peak_counts):7.2f} dBFS",
            f"Dominant >{PEAK_MIN_HZ:.0f} Hz:  {peak_text}",
            f"Frames lost {self.stream.lost_frames}   "
            f"dropped on device {self.stream.dropped_frames}   "
            f"resyncs {self.stream.resyncs}   "
            f"gaps {self.reader.gaps}",
        ]

        lines.append(
            self.player.status()
            if self.player is not None
            else self.audio_status
        )

        decision = (
            self.reader.latest_decision() if self.args.classify else None
        )

        if decision is not None:
            lines.append(
                f"Bands   30-80 {decision.smoothed['30-80']:7.1f}   "
                f"500-1k {decision.smoothed['500-1k']:7.1f}   "
                f"1k-2k {decision.smoothed['1k-2k']:7.1f}"
            )

        self.readout.set_text("\n".join(lines))

        self._update_state(decision)

        return ()

    def _update_state(self, decision: Decision | None) -> None:
        if not self.args.classify:
            return

        if decision is None:
            self.state_text.set_text("STATE  --    waiting for audio")
            self.state_text.set_color("#888888")
            return

        state = decision.state or "--"

        self.state_text.set_text(
            f"STATE  {state:<11}"
            f"candidate {decision.candidate:<11}"
            f"stable {decision.stable_seconds:5.1f} s"
        )

        self.state_text.set_color(
            STATE_COLOURS.get(state, "#888888")
        )

    def _rescale_waveform(self) -> None:
        """Grow at once to fit a transient, shrink back slowly.

        Snapping the axis to every block would make the trace jitter
        and hide how loud one state is relative to another.
        """
        needed = max(
            float(np.max(np.abs(self.ring))) * 1.2,
            WAVEFORM_MIN_SCALE,
        )

        current = self.wave_axis.get_ylim()[1]

        if needed > current:
            scale = needed
        elif needed < current * 0.5:
            scale = current * 0.9
        else:
            return

        self.wave_axis.set_ylim(-scale, scale)

    def _push(self, samples: np.ndarray) -> None:
        block = samples.astype(np.float64) / FULL_SCALE

        count = min(len(block), len(self.ring))

        self.ring = np.roll(self.ring, -count)
        self.ring[-count:] = block[-count:]

        column = self._spectrum_db(
            self._latest_window()
        )[: self.fmax_bin]

        self.spectrogram = np.roll(self.spectrogram, -1, axis=1)
        self.spectrogram[:, -1] = column

    def _latest_window(self) -> np.ndarray:
        block = self.ring[-self.nfft :].copy()

        return block - block.mean()

    # ------------------------------------------------------------- run

    def run(self) -> int:
        from matplotlib.animation import FuncAnimation

        self.reader.start()

        # Held so the animation is not garbage collected.
        self.animation = FuncAnimation(
            self.figure,
            self.update,
            interval=REDRAW_INTERVAL_MS,
            blit=False,
            cache_frame_data=False,
        )

        try:
            plt.show()
        finally:
            self.reader.stop()

            if self.player is not None:
                self.player.stop()

        if self.reader.error is not None:
            print(f"error: {self.reader.error}", file=sys.stderr)
            return 1

        return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # sounddevice takes either an index or a name substring.
    if isinstance(args.audio_device, str) and args.audio_device.isdigit():
        args.audio_device = int(args.audio_device)

    if args.list_audio_devices:
        return list_audio_devices()

    if args.audio_test:
        return audio_test(args)

    if args.list_ports:
        ports = list_ports()
        print("\n".join(ports) if ports else "No serial ports found.")
        return 0

    if not args.port:
        parser.error("a serial port is required (or use --list-ports)")

    if args.nfft < 64 or args.nfft & (args.nfft - 1):
        parser.error("--nfft must be a power of two and at least 64")

    try:
        stream = AudioStream(args.port, args.baud)
        stream.open()
    except ProtocolError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"error: could not open {args.port}: {error}", file=sys.stderr)
        return 1

    assert stream.info is not None
    print(f"Stream: {stream.info.describe()}")
    print("Close the plot window to stop.")

    try:
        return Viewer(args, stream).run()
    finally:
        stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
