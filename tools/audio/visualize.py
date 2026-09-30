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
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks

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

    return parser


class StreamReader(threading.Thread):
    """Pulls frames off the serial port into a queue."""

    def __init__(self, stream: AudioStream) -> None:
        super().__init__(daemon=True)

        self.stream = stream
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)
        self.stop_event = threading.Event()
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            for frame in self.stream.frames():
                if self.stop_event.is_set():
                    return

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

        self.reader = StreamReader(stream)

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

        self.readout = self.text_axis.text(
            0.0,
            0.5,
            "",
            family="monospace",
            fontsize=11,
            va="center",
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

        self.readout.set_text(
            f"RMS  {rms_counts:10.0f} counts   {dbfs(rms_counts):7.2f} dBFS\n"
            f"Peak {peak_counts:10.0f} counts   {dbfs(peak_counts):7.2f} dBFS\n"
            f"Dominant >{PEAK_MIN_HZ:.0f} Hz:  {peak_text}\n"
            f"Frames lost {self.stream.lost_frames}   "
            f"dropped on device {self.stream.dropped_frames}   "
            f"resyncs {self.stream.resyncs}"
        )

        return ()

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

        if self.reader.error is not None:
            print(f"error: {self.reader.error}", file=sys.stderr)
            return 1

        return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

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
