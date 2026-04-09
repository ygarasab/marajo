"""
Helpers for heatmaps built from peaks.csv (freq_hz, amp_psd).

Used by scripts that compare PSD peaks across days and/or camera rotations.
"""

from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict

import numpy as np

PEAKS_FILENAME = "peaks.csv"
_REQUIRED_PEAK_COLS = frozenset({"freq_hz", "amp_psd"})


def _rotation_sort_key(name: str) -> int:
    try:
        return int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def _validate_peaks_header(fieldnames: list[str] | None, csv_path: str) -> None:
    if not fieldnames or not _REQUIRED_PEAK_COLS.issubset(fieldnames):
        missing = sorted(_REQUIRED_PEAK_COLS - set(fieldnames or []))
        raise ValueError(f"Missing columns {missing} in: {csv_path}")


def read_peaks_csv_into_bins(csv_path: str, round_decimals: int) -> dict[float, list[float]]:
    """Parse one peaks.csv into rounded_freq -> list of amp_psd."""
    bins: dict[float, list[float]] = defaultdict(list)
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        _validate_peaks_header(reader.fieldnames, csv_path)
        for row in reader:
            freq_str = row.get("freq_hz")
            amp_str = row.get("amp_psd")
            if not freq_str or not amp_str:
                continue
            raw_freq = float(freq_str)
            amp_psd = float(amp_str)
            rounded_freq = round(raw_freq, round_decimals)
            bins[rounded_freq].append(amp_psd)
    return dict(bins)


def list_day_dirs(base_path: str) -> list[str]:
    """Sorted immediate child directory names under base_path (e.g. date folders)."""
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base path does not exist: {base_path}")

    day_dirs = []
    for name in os.listdir(base_path):
        day_path = os.path.join(base_path, name)
        if os.path.isdir(day_path):
            day_dirs.append(name)

    day_dirs.sort()
    return day_dirs


def discover_rotations(base_path: str, fps: str) -> list[str]:
    """
    Rotation folder names under base_path/<day>/<fps>/ that contain peaks.csv
    for at least one day.
    """
    seen: set[str] = set()
    for day in list_day_dirs(base_path):
        rot_parent = os.path.join(base_path, day, fps)
        if not os.path.isdir(rot_parent):
            continue
        for name in os.listdir(rot_parent):
            if not name.startswith("rotation_"):
                continue
            peaks_path = os.path.join(rot_parent, name, PEAKS_FILENAME)
            if os.path.isfile(peaks_path):
                seen.add(name)

    return sorted(seen, key=_rotation_sort_key)


def discover_rotations_for_day(base_path: str, day: str, fps: str) -> list[str]:
    """Rotation folders under base_path/<day>/<fps>/ that contain peaks.csv."""
    rot_parent = os.path.join(base_path, day, fps)
    if not os.path.isdir(rot_parent):
        return []

    names: list[str] = []
    for name in os.listdir(rot_parent):
        if not name.startswith("rotation_"):
            continue
        peaks_path = os.path.join(rot_parent, name, PEAKS_FILENAME)
        if os.path.isfile(peaks_path):
            names.append(name)

    return sorted(names, key=_rotation_sort_key)


def fps_output_dir(base_path: str, fps: str) -> str:
    """Directory for heatmaps aggregated over days at fixed fps: base_path/fps/."""
    return os.path.join(base_path, fps)


def load_day_bins(
    base_path: str,
    fps: str,
    rotation_folder: str,
    round_decimals: int,
) -> tuple[Counter[float], dict[str, dict[float, list[float]]], list[str]]:
    """
    One fixed rotation across all days: columns = day, rows = freq bins.

    Returns:
      freq_counter: total counts per rounded freq (all days)
      day_bins: day -> rounded_freq -> amp_psd values
      days: column order (days that had this rotation's peaks.csv)
    """
    freq_counter: Counter[float] = Counter()
    day_bins: dict[str, dict[float, list[float]]] = {}
    days_used: list[str] = []

    for day in list_day_dirs(base_path):
        fpath = os.path.join(base_path, day, fps, rotation_folder, PEAKS_FILENAME)
        if not os.path.exists(fpath):
            continue

        bins_for_day = read_peaks_csv_into_bins(fpath, round_decimals)
        for rf, vals in bins_for_day.items():
            freq_counter[rf] += len(vals)

        day_bins[day] = bins_for_day
        days_used.append(day)

    return freq_counter, day_bins, days_used


def load_day_bins_pooled_rotations(
    base_path: str,
    fps: str,
    round_decimals: int,
) -> tuple[
    Counter[float],
    dict[str, dict[float, list[float]]],
    list[str],
    dict[str, dict[float, int]],
    dict[str, int],
]:
    """
    All rotations pooled per day; columns = day (same layout as ``load_day_bins``).

    For each day, every ``rotation_*/peaks.csv`` under ``.../<day>/<fps>/`` is read and
    merged: same rounded ``freq_hz`` maps to one list of ``amp_psd`` values from all
    rotations that day.

    ``freq_counter`` counts every peak row across all days and all rotations (used for
    the global frequency ranking threshold).

    Additionally returns presence probability helpers:
      - ``presence_counts[day][freq]`` = number of rotations where that freq bin exists
      - ``rotations_per_day[day]`` = number of rotation folders that had a peaks.csv
    """
    freq_counter: Counter[float] = Counter()
    day_bins: dict[str, dict[float, list[float]]] = {}
    days_used: list[str] = []
    presence_counts: dict[str, dict[float, int]] = {}
    rotations_per_day: dict[str, int] = {}

    for day in list_day_dirs(base_path):
        rot_parent = os.path.join(base_path, day, fps)
        if not os.path.isdir(rot_parent):
            continue

        merged: dict[float, list[float]] = defaultdict(list)
        saw_any = False
        day_presence: dict[float, int] = defaultdict(int)
        day_rotation_count = 0

        for name in sorted(os.listdir(rot_parent), key=_rotation_sort_key):
            if not name.startswith("rotation_"):
                continue
            fpath = os.path.join(rot_parent, name, PEAKS_FILENAME)
            if not os.path.isfile(fpath):
                continue
            saw_any = True
            day_rotation_count += 1
            bins_rot = read_peaks_csv_into_bins(fpath, round_decimals)
            for rf, vals in bins_rot.items():
                merged[rf].extend(vals)
                freq_counter[rf] += len(vals)
                day_presence[rf] += 1

        if saw_any:
            day_bins[day] = dict(merged)
            days_used.append(day)
            presence_counts[day] = dict(day_presence)
            rotations_per_day[day] = day_rotation_count

    return freq_counter, day_bins, days_used, presence_counts, rotations_per_day


def load_rotation_bins_for_day(
    base_path: str,
    day: str,
    fps: str,
    round_decimals: int,
) -> tuple[Counter[float], dict[str, dict[float, list[float]]], list[str]]:
    """
    One fixed day, all rotations: columns = rotation_*, rows = freq bins.

    freq_counter counts rounded freq across every rotation's peaks.csv that day.
    """
    rotations = discover_rotations_for_day(base_path, day, fps)
    freq_counter: Counter[float] = Counter()
    rotation_bins: dict[str, dict[float, list[float]]] = {}

    for rot in rotations:
        fpath = os.path.join(base_path, day, fps, rot, PEAKS_FILENAME)
        bins_for_rot = read_peaks_csv_into_bins(fpath, round_decimals)
        for rf, vals in bins_for_rot.items():
            freq_counter[rf] += len(vals)
        rotation_bins[rot] = bins_for_rot

    return freq_counter, rotation_bins, rotations


def select_frequencies(freq_counter: Counter[float], min_appearances: int) -> list[float]:
    """Frequencies whose total count is strictly greater than min_appearances."""
    selected = [freq for freq, count in freq_counter.items() if count > min_appearances]
    selected.sort()
    return selected


def build_heatmap_matrix(
    freqs: list[float],
    column_keys: list[str],
    bins_by_column: dict[str, dict[float, list[float]]],
    *,
    agg: str = "mean",
    trim_fraction: float = 0.1,
) -> np.ndarray:
    """
    Matrix shape (len(freqs), len(column_keys)).

    Cell = mean or max amp_psd for that (column, rounded freq bin).
    """
    if agg not in ("mean", "max", "var", "trimmed_mean", "count"):
        raise ValueError(
            "agg must be one of 'mean', 'max', 'var', 'trimmed_mean', 'count', "
            f"got {agg!r}"
        )

    mat = np.full((len(freqs), len(column_keys)), np.nan, dtype=float)
    freq_index = {f: i for i, f in enumerate(freqs)}

    for j, col in enumerate(column_keys):
        col_bins = bins_by_column.get(col, {})
        for f_bin, values in col_bins.items():
            i = freq_index.get(f_bin)
            if i is None:
                continue
            if values:
                if agg == "mean":
                    mat[i, j] = float(np.mean(values))
                elif agg == "max":
                    mat[i, j] = float(np.max(values))
                elif agg == "var":
                    mat[i, j] = float(np.var(values))
                elif agg == "trimmed_mean":
                    v = sorted(values)
                    n = len(v)
                    k = int(n * trim_fraction)
                    if k <= 0 or 2 * k >= n:
                        mat[i, j] = float(np.mean(v))
                    else:
                        trimmed = v[k : n - k]
                        mat[i, j] = (
                            float(np.mean(trimmed)) if trimmed else float(np.mean(v))
                        )
                elif agg == "count":
                    mat[i, j] = float(len(values))

    return mat


def build_presence_probability_matrix(
    freqs: list[float],
    days: list[str],
    presence_counts: dict[str, dict[float, int]],
    rotations_per_day: dict[str, int],
) -> np.ndarray:
    """
    Presence probability matrix (freq x day).

    Cell value = presence_counts[day][freq] / rotations_per_day[day].
    """
    mat = np.full((len(freqs), len(days)), np.nan, dtype=float)
    freq_index = {f: i for i, f in enumerate(freqs)}

    for j, day in enumerate(days):
        total = rotations_per_day.get(day, 0)
        if total <= 0:
            continue
        day_presence = presence_counts.get(day, {})
        for f_bin, present_count in day_presence.items():
            i = freq_index.get(f_bin)
            if i is None:
                continue
            mat[i, j] = float(present_count) / float(total)

    return mat


def _is_multiple_of(freq: float, base: float, tol: float) -> bool:
    """True if freq is approximately a multiple of base (within tol)."""
    if base <= 0:
        return False
    remainder = freq % base
    return remainder <= tol or (base - remainder) <= tol


def _freq_accepted(
    freq: float,
    *,
    min_freq: float = 0.0,
    exclude_base: float = 0.0,
    exclude_tol: float = 0.05,
) -> bool:
    if freq < min_freq:
        return False
    if exclude_base > 0 and _is_multiple_of(freq, exclude_base, exclude_tol):
        return False
    return True


def _find_max_amp_freq(
    csv_path: str,
    *,
    min_freq: float = 0.0,
    exclude_base: float = 0.0,
    exclude_tol: float = 0.05,
) -> tuple[float, float] | None:
    """Return (freq_hz, amp_psd) of the peak with highest amp_psd in a peaks.csv."""
    best: tuple[float, float] | None = None
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        _validate_peaks_header(reader.fieldnames, csv_path)
        for row in reader:
            freq_str = row.get("freq_hz")
            amp_str = row.get("amp_psd")
            if not freq_str or not amp_str:
                continue
            freq = float(freq_str)
            if not _freq_accepted(
                freq,
                min_freq=min_freq,
                exclude_base=exclude_base,
                exclude_tol=exclude_tol,
            ):
                continue
            amp = float(amp_str)
            if best is None or amp > best[1]:
                best = (freq, amp)
    return best


def collect_peak_freqs_per_day(
    base_path: str,
    fps: str,
    rotation_folder: str,
    *,
    min_freq: float = 0.0,
    exclude_base: float = 0.0,
    exclude_tol: float = 0.05,
) -> tuple[list[str], list[float]]:
    """For a single rotation, get the freq with max amp_psd per day."""
    days_out: list[str] = []
    freqs_out: list[float] = []
    for day in list_day_dirs(base_path):
        fpath = os.path.join(base_path, day, fps, rotation_folder, PEAKS_FILENAME)
        if not os.path.isfile(fpath):
            continue
        result = _find_max_amp_freq(
            fpath,
            min_freq=min_freq,
            exclude_base=exclude_base,
            exclude_tol=exclude_tol,
        )
        if result is not None:
            days_out.append(day)
            freqs_out.append(result[0])
    return days_out, freqs_out


def collect_peak_freqs_per_day_pooled(
    base_path: str,
    fps: str,
    *,
    min_freq: float = 0.0,
    exclude_base: float = 0.0,
    exclude_tol: float = 0.05,
) -> tuple[list[str], list[float]]:
    """All rotations pooled: get the freq with max amp_psd per day."""
    days_out: list[str] = []
    freqs_out: list[float] = []
    for day in list_day_dirs(base_path):
        rot_parent = os.path.join(base_path, day, fps)
        if not os.path.isdir(rot_parent):
            continue
        best: tuple[float, float] | None = None
        for name in os.listdir(rot_parent):
            if not name.startswith("rotation_"):
                continue
            fpath = os.path.join(rot_parent, name, PEAKS_FILENAME)
            if not os.path.isfile(fpath):
                continue
            result = _find_max_amp_freq(
                fpath,
                min_freq=min_freq,
                exclude_base=exclude_base,
                exclude_tol=exclude_tol,
            )
            if result is not None and (best is None or result[1] > best[1]):
                best = result
        if best is not None:
            days_out.append(day)
            freqs_out.append(best[0])
    return days_out, freqs_out


def plot_peak_frequency_evolution(
    days: list[str],
    peak_freqs: list[float],
    out_path: str,
    *,
    title: str = "Evolução das frequências médias de maior intensidade ao longo dos dias",
    xlabel: str = "Dia da semana",
) -> None:
    """Plot the frequency with highest amp_psd per day, with cubic spline interpolation."""
    try:
        import matplotlib.pyplot as plt
        from scipy.interpolate import CubicSpline
    except Exception as e:
        raise RuntimeError(f"matplotlib and scipy are required: {e}") from e

    if len(days) < 2:
        print("Not enough data points for peak frequency evolution plot.")
        return

    x_arr = np.arange(1, len(days) + 1, dtype=float)
    y_arr = np.array(peak_freqs, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))

    cs = CubicSpline(x_arr, y_arr)
    x_smooth = np.linspace(x_arr.min(), x_arr.max(), 300)
    y_smooth = cs(x_smooth)

    ax.plot(x_smooth, y_smooth, "-", label="CP mean (interpolado)")
    ax.plot(x_arr, y_arr, "o", markersize=6, label="Pontos originais")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequência (Hz)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heatmap(
    freqs: list[float],
    column_labels: list[str],
    mat: np.ndarray,
    out_path: str,
    *,
    colorbar_label: str,
    title: str,
    xlabel: str,
) -> None:
    """Save a frequency (y) x columns (x) heatmap."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"matplotlib is required to plot heatmap: {e}") from e

    if len(freqs) == 0 or len(column_labels) == 0:
        print("No data to plot heatmap.")
        return

    ncols = len(column_labels)
    nrows = len(freqs)
    fig_w = max(8, min(0.55 * ncols, 20))
    fig_h = max(5, min(0.22 * nrows, 16))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat, aspect="auto", origin="lower", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, rotation=90)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("freq_hz")

    if ncols <= 30:
        x_ticks = np.arange(ncols)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(column_labels, rotation=90, fontsize=8)
    else:
        step = max(1, ncols // 25)
        x_ticks = np.arange(0, ncols, step)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([column_labels[k] for k in x_ticks], rotation=90, fontsize=8)

    if nrows <= 40:
        y_ticks = np.arange(nrows)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{f:g}" for f in freqs], fontsize=8)
    else:
        step = max(1, nrows // 25)
        y_ticks = np.arange(0, nrows, step)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{freqs[k]:g}" for k in y_ticks], fontsize=8)

    ax.set_title(title)
    fig.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
