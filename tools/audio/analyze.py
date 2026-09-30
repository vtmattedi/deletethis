"""Offline analysis of recorded INMP441 audio.

    python tools/audio/analyze.py                       # everything
    python tools/audio/analyze.py recordings/off_01.wav recordings/fan_01.wav
    python tools/audio/analyze.py --nfft 4096 --plots

Computes the same features over every recording, grouped by state, so
that OFF / FAN / COMPRESSOR can be compared on identical terms.

The defaults deliberately match the firmware in ``src/main.cpp``:
16 kHz, 1024-sample FFT, Hamming window. ``--nfft`` overrides the
window size so 2048 and 4096 can be tried without reflashing.

The state of each recording is taken from its file name: timestamps
and trailing index numbers are stripped, so both ``off_01.wav`` and
``2026-09-30_091500_off.wav`` are labelled ``off``.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram

HERE = Path(__file__).resolve().parent
DEFAULT_RECORDINGS = HERE / "recordings"
DEFAULT_PLOTDIR = HERE / "plots"

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_NFFT = 1024

# Bands to characterise. The 1100-1200 entry is deliberately narrow:
# the firmware's TEXT mode kept reporting a peak near 1156 Hz, and the
# point of this analysis is to find out whether that tone is actually
# tied to the air conditioner running or is just always there.
BANDS: list[tuple[str, float, float]] = [
    ("30-80", 30.0, 80.0),
    ("80-200", 80.0, 200.0),
    ("200-500", 200.0, 500.0),
    ("500-1k", 500.0, 1000.0),
    ("1k-2k", 1000.0, 2000.0),
    ("2k-4k", 2000.0, 4000.0),
    ("1100-1200", 1100.0, 1200.0),
]

# Peaks below this are mains hum and handling noise.
PEAK_MIN_HZ = 100.0
PEAK_MAX_HZ = 4000.0

DB_FLOOR = -120.0

# Recording names: strip timestamps and trailing indices to get a state.
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_PATTERN = re.compile(r"^\d+$")

STATE_ORDER = ["off", "fan", "compressor"]


def infer_state(path: Path) -> str:
    tokens = [
        token
        for token in path.stem.split("_")
        if token
        and not DATE_PATTERN.match(token)
        and not TIME_PATTERN.match(token)
    ]

    return "_".join(tokens).lower() if tokens else path.stem.lower()


def to_float(samples: np.ndarray) -> np.ndarray:
    """Normalise any WAV integer format to [-1, 1) floats."""
    if samples.ndim > 1:
        samples = samples[:, 0]

    if np.issubdtype(samples.dtype, np.floating):
        return samples.astype(np.float64)

    info = np.iinfo(samples.dtype)

    if info.min == 0:
        # 8-bit WAV is unsigned.
        midpoint = (info.max + 1) / 2.0
        return (samples.astype(np.float64) - midpoint) / midpoint

    return samples.astype(np.float64) / float(-info.min)


def to_db(power: np.ndarray | float) -> np.ndarray | float:
    """Mean-square power (full scale = 1.0) -> dBFS."""
    amplitude = np.sqrt(np.maximum(power, 0.0))

    return np.maximum(
        20.0 * np.log10(np.maximum(amplitude, 1e-12)),
        DB_FLOOR,
    )


def frame_signal(
    signal: np.ndarray,
    nperseg: int,
    noverlap: int,
) -> np.ndarray:
    """Split into overlapping windows, aligned with ``spectrogram``.

    Same segmentation scipy uses, so window *i* here is window *i*
    there and the time axis is shared.
    """
    step = nperseg - noverlap

    count = 1 + (signal.size - nperseg) // step

    offsets = step * np.arange(count)[:, None]

    return signal[offsets + np.arange(nperseg)[None, :]]


@dataclass
class Analysis:
    path: Path
    state: str
    sample_rate: int
    duration: float

    times: np.ndarray            # (n_windows,)
    freqs: np.ndarray            # (n_bins,)
    power: np.ndarray            # (n_bins, n_windows) mean-square per bin

    # True RMS of each raw window, in dBFS. Measured in the time
    # domain, NOT summed out of `power`: see analyse_file.
    rms_db: np.ndarray           # (n_windows,)

    # Windowed spectral power per band, in dBFS. A different quantity
    # from rms_db, on a different convention, so the two are kept
    # apart rather than one being derived from the other.
    band_db: dict[str, np.ndarray]

    peak_hz: np.ndarray          # (n_windows,)

    @property
    def mean_spectrum_db(self) -> np.ndarray:
        return to_db(self.power.mean(axis=1))


def analyse_file(
    path: Path,
    nfft: int,
    overlap: float,
) -> Analysis:
    sample_rate, raw = wavfile.read(path)

    signal = to_float(raw)

    if signal.size < nfft:
        raise ValueError(
            f"{path.name}: only {signal.size} samples, "
            f"which is shorter than the {nfft}-sample window"
        )

    signal = signal - signal.mean()

    noverlap = int(nfft * overlap)

    freqs, times, power = spectrogram(
        signal,
        fs=sample_rate,
        window="hamming",
        nperseg=nfft,
        noverlap=noverlap,
        detrend="constant",
        scaling="spectrum",
        mode="psd",
    )

    # ------------------------------------------------------------------
    # Overall level: true RMS, straight from the time-domain windows.
    #
    # Summing `power` over frequency would be convenient but it is not
    # the RMS of the signal. scaling="spectrum" normalises by the
    # window's coherent gain so that a pure tone reads its true
    # amplitude; broadband noise under that same normalisation reads
    # high by the window's equivalent noise bandwidth, about 1.33 dB
    # for Hamming. That is the right convention for comparing bands and
    # for spotting a tone, and the wrong one for "how loud is it" --
    # and the error depends on how tonal the signal is, so it does not
    # even cancel when comparing OFF against COMPRESSOR.
    #
    # So the two are computed separately and neither is derived from
    # the other: rms_db is exact, band_db is spectral.
    # ------------------------------------------------------------------
    windows = frame_signal(signal, nfft, noverlap)

    # Match the spectrogram's detrend="constant" per window.
    windows = windows - windows.mean(axis=1, keepdims=True)

    rms_db = to_db(np.mean(windows * windows, axis=1))

    if rms_db.size != times.size:
        raise ValueError(
            f"{path.name}: framing mismatch, "
            f"{rms_db.size} windows vs {times.size} spectrogram columns"
        )

    band_db: dict[str, np.ndarray] = {}

    for name, low, high in BANDS:
        mask = (freqs >= low) & (freqs < high)

        if not mask.any():
            band_db[name] = np.full(times.shape, DB_FLOOR)
            continue

        band_db[name] = to_db(power[mask, :].sum(axis=0))

    peak_mask = (freqs >= PEAK_MIN_HZ) & (freqs <= PEAK_MAX_HZ)
    peak_freqs = freqs[peak_mask]

    if peak_freqs.size:
        peak_hz = peak_freqs[np.argmax(power[peak_mask, :], axis=0)]
    else:
        peak_hz = np.zeros(times.shape)

    return Analysis(
        path=path,
        state=infer_state(path),
        sample_rate=sample_rate,
        duration=signal.size / sample_rate,
        times=times,
        freqs=freqs,
        power=power,
        rms_db=rms_db,
        band_db=band_db,
        peak_hz=peak_hz,
    )


# ----------------------------------------------------------- reporting


def format_table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [len(header) for header in headers]

    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render(cells: list[str]) -> str:
        return "  ".join(
            cell.ljust(widths[index]) if index == 0 else cell.rjust(widths[index])
            for index, cell in enumerate(cells)
        )

    lines = [render(headers), "  ".join("-" * width for width in widths)]
    lines.extend(render(row) for row in rows)

    return "\n".join(lines)


def state_sort_key(state: str) -> tuple[int, str]:
    if state in STATE_ORDER:
        return (STATE_ORDER.index(state), state)

    return (len(STATE_ORDER), state)


def group_by_state(
    analyses: list[Analysis],
) -> dict[str, list[Analysis]]:
    grouped: dict[str, list[Analysis]] = {}

    for analysis in analyses:
        grouped.setdefault(analysis.state, []).append(analysis)

    return dict(
        sorted(grouped.items(), key=lambda item: state_sort_key(item[0]))
    )


def per_file_table(analyses: list[Analysis]) -> str:
    headers = ["File", "State", "s"] + ["RMS"] + [name for name, _, _ in BANDS] + ["peak Hz"]

    rows = []

    for analysis in analyses:
        row = [
            analysis.path.name,
            analysis.state,
            f"{analysis.duration:.1f}",
            f"{np.median(analysis.rms_db):.1f}",
        ]

        row += [
            f"{np.median(analysis.band_db[name]):.1f}"
            for name, _, _ in BANDS
        ]

        row.append(f"{np.median(analysis.peak_hz):.0f}")

        rows.append(row)

    return format_table(rows, headers)


def comparison_table(grouped: dict[str, list[Analysis]]) -> str:
    headers = (
        ["State", "files", "s"]
        + ["RMS"]
        + [name for name, _, _ in BANDS]
        + ["peak Hz"]
    )

    rows = []

    for state, analyses in grouped.items():
        rms = np.concatenate([a.rms_db for a in analyses])
        peak = np.concatenate([a.peak_hz for a in analyses])
        duration = sum(a.duration for a in analyses)

        row = [
            state,
            str(len(analyses)),
            f"{duration:.0f}",
            f"{np.median(rms):.1f}",
        ]

        for name, _, _ in BANDS:
            values = np.concatenate([a.band_db[name] for a in analyses])
            row.append(f"{np.median(values):.1f}")

        row.append(f"{np.median(peak):.0f}")

        rows.append(row)

    return format_table(rows, headers)


def separation_table(grouped: dict[str, list[Analysis]]) -> str:
    """How far apart are the states, in units of within-state spread?

    A feature only earns a place in the firmware classifier if the gap
    between states is large compared with the noise inside each state.
    This is the crude version of that check: the spread between state
    medians, divided by the typical within-state interquartile range.
    """
    states = list(grouped)

    if len(states) < 2:
        return "(need at least two states to compare)"

    features = ["RMS"] + [name for name, _, _ in BANDS]

    rows = []

    for feature in features:
        medians = []
        spreads = []

        for analyses in grouped.values():
            if feature == "RMS":
                values = np.concatenate([a.rms_db for a in analyses])
            else:
                values = np.concatenate([a.band_db[feature] for a in analyses])

            medians.append(float(np.median(values)))
            spreads.append(
                float(
                    np.percentile(values, 75) - np.percentile(values, 25)
                )
            )

        span = max(medians) - min(medians)
        noise = float(np.mean(spreads))

        score = span / noise if noise > 1e-6 else float("inf")

        rows.append(
            [
                feature,
                f"{span:.1f}",
                f"{noise:.1f}",
                f"{score:.2f}",
            ]
        )

    rows.sort(key=lambda row: float(row[3]), reverse=True)

    return format_table(
        rows,
        ["Feature", "span dB", "spread dB", "span/spread"],
    )


def write_csv(path: Path, analyses: list[Analysis]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)

        writer.writerow(
            ["file", "state", "t", "rms_db"]
            + [name for name, _, _ in BANDS]
            + ["peak_hz"]
        )

        for analysis in analyses:
            for index, time in enumerate(analysis.times):
                writer.writerow(
                    [
                        analysis.path.name,
                        analysis.state,
                        f"{time:.3f}",
                        f"{analysis.rms_db[index]:.2f}",
                    ]
                    + [
                        f"{analysis.band_db[name][index]:.2f}"
                        for name, _, _ in BANDS
                    ]
                    + [f"{analysis.peak_hz[index]:.1f}"]
                )


# --------------------------------------------------------------- plots


def make_plots(
    grouped: dict[str, list[Analysis]],
    plotdir: Path,
    fmax: float,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plotdir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    # --- average spectrum per state ----------------------------------
    figure, axis = plt.subplots(figsize=(11, 6))

    for state, analyses in grouped.items():
        freqs = analyses[0].freqs

        spectra = [a.power.mean(axis=1) for a in analyses]
        mean_power = np.mean(spectra, axis=0)

        axis.plot(freqs, to_db(mean_power), linewidth=1.0, label=state)

    for _, low, high in BANDS:
        axis.axvspan(low, high, color="grey", alpha=0.05)

    axis.set_xlim(0, fmax)
    axis.set_xlabel("Hz")
    axis.set_ylabel("dBFS")
    axis.set_title("Average spectrum per state")
    axis.grid(alpha=0.3)
    axis.legend()

    path = plotdir / "spectrum_by_state.png"
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    written.append(path)

    # --- band energy over time, and spectrogram, per file ------------
    for state, analyses in grouped.items():
        for analysis in analyses:
            written.append(_plot_bands_over_time(plt, analysis, plotdir))
            written.append(_plot_spectrogram(plt, analysis, plotdir, fmax))

    return written


def _plot_bands_over_time(plt, analysis: Analysis, plotdir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(11, 5))

    axis.plot(
        analysis.times,
        analysis.rms_db,
        linewidth=1.4,
        color="black",
        label="RMS",
    )

    for name, _, _ in BANDS:
        axis.plot(
            analysis.times,
            analysis.band_db[name],
            linewidth=0.8,
            label=name,
        )

    axis.set_xlabel("s")
    axis.set_ylabel("dBFS")
    axis.set_title(f"Band energy over time - {analysis.path.name}")
    axis.grid(alpha=0.3)
    axis.legend(ncol=4, fontsize=8)

    path = plotdir / f"bands_{analysis.path.stem}.png"
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)

    return path


def _plot_spectrogram(plt, analysis: Analysis, plotdir: Path, fmax: float) -> Path:
    figure, axis = plt.subplots(figsize=(11, 5))

    mesh = axis.pcolormesh(
        analysis.times,
        analysis.freqs,
        to_db(analysis.power),
        shading="auto",
        cmap="magma",
        vmin=DB_FLOOR,
        vmax=-20.0,
    )

    axis.set_ylim(0, fmax)
    axis.set_xlabel("s")
    axis.set_ylabel("Hz")
    axis.set_title(f"Spectrogram - {analysis.path.name}")

    figure.colorbar(mesh, ax=axis, label="dBFS", pad=0.01)

    path = plotdir / f"spectrogram_{analysis.path.stem}.png"
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)

    return path


# ----------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare recorded AC states with fixed DSP features.",
    )

    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="WAV files (default: every WAV in tools/audio/recordings)",
    )

    parser.add_argument(
        "--nfft",
        type=int,
        default=DEFAULT_NFFT,
        help=f"FFT/window size (default: {DEFAULT_NFFT}; try 2048, 4096)",
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Window overlap as a fraction (default: 0.5)",
    )

    parser.add_argument(
        "--fmax",
        type=float,
        default=None,
        help="Highest frequency to plot (default: Nyquist)",
    )

    parser.add_argument(
        "--plots",
        action="store_true",
        help="Write comparison plots as PNG files",
    )

    parser.add_argument(
        "--plotdir",
        type=Path,
        default=DEFAULT_PLOTDIR,
        help="Where to write plots (default: tools/audio/plots)",
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Also write per-window features to this CSV file",
    )

    return parser


def collect_files(args: argparse.Namespace) -> list[Path]:
    if args.files:
        paths: list[Path] = []

        for entry in args.files:
            if entry.is_dir():
                paths.extend(sorted(entry.glob("*.wav")))
            else:
                paths.append(entry)

        return paths

    return sorted(DEFAULT_RECORDINGS.glob("*.wav"))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.nfft < 64 or args.nfft & (args.nfft - 1):
        parser.error("--nfft must be a power of two and at least 64")

    if not 0.0 <= args.overlap < 1.0:
        parser.error("--overlap must be in [0, 1)")

    paths = collect_files(args)

    if not paths:
        print(
            f"No recordings found in {DEFAULT_RECORDINGS}.\n"
            "Record some first, e.g.\n"
            "  python tools/audio/capture.py COM13 --label off --seconds 30",
            file=sys.stderr,
        )
        return 1

    analyses: list[Analysis] = []

    for path in paths:
        if not path.exists():
            print(f"warning: {path} does not exist, skipping", file=sys.stderr)
            continue

        try:
            analyses.append(analyse_file(path, args.nfft, args.overlap))
        except (ValueError, OSError) as error:
            print(f"warning: {error}", file=sys.stderr)

    if not analyses:
        print("error: nothing could be analysed", file=sys.stderr)
        return 1

    sample_rate = analyses[0].sample_rate
    fmax = args.fmax or sample_rate / 2.0

    mismatched = {a.sample_rate for a in analyses}

    if len(mismatched) > 1:
        print(
            f"warning: mixed sample rates {sorted(mismatched)}; "
            "band edges are still in Hz, but the comparison is uneven",
            file=sys.stderr,
        )

    resolution = sample_rate / args.nfft

    print()
    print(
        f"{len(analyses)} recording(s)   "
        f"{sample_rate} Hz   "
        f"N={args.nfft} ({resolution:.2f} Hz/bin, "
        f"{args.nfft / sample_rate * 1000:.0f} ms/window)   "
        f"Hamming   overlap {args.overlap:.0%}"
    )

    print()
    print("Per file (median over windows, dBFS)")
    print("RMS is true time-domain level; bands are spectral power.")
    print()
    print(per_file_table(analyses))

    grouped = group_by_state(analyses)

    print()
    print("Per state (median over all windows of all files, dBFS)")
    print("RMS is true time-domain level; bands are spectral power.")
    print()
    print(comparison_table(grouped))

    print()
    print("Feature separation (higher is more useful for a classifier)")
    print()
    print(separation_table(grouped))

    if args.csv:
        write_csv(args.csv, analyses)
        print()
        print(f"Wrote per-window features to {args.csv}")

    if args.plots:
        written = make_plots(grouped, args.plotdir, fmax)
        print()
        print(f"Wrote {len(written)} plot(s) to {args.plotdir}")

    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
