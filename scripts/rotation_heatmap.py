import os
import csv
from collections import Counter, defaultdict

import numpy as np

DEFAULT_BASE_PATH = "out/processed/0.5/500"
DEFAULT_FPS_VALUES = ("60", "240")

MIN_APPEARANCES = 40
ROUND_DECIMALS = 6

PEAKS_FILENAME = "peaks.csv"


def list_day_dirs(base_path: str) -> list[str]:
    """Return sorted immediate child directory names under base_path (e.g. date folders)."""
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
    Find all rotation_* folders under base_path/<day>/<fps>/ that contain peaks.csv
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

    def rot_sort_key(name: str) -> int:
        try:
            return int(name.split("_", 1)[1])
        except (IndexError, ValueError):
            return 0

    return sorted(seen, key=rot_sort_key)


def fps_output_dir(base_path: str, fps: str) -> str:
    """Heatmaps for this (scale, frames, fps) setup go under base_path/fps/."""
    return os.path.join(base_path, fps)


def load_day_bins(
    base_path: str,
    fps: str,
    rotation_folder: str,
    round_decimals: int,
) -> tuple[Counter[float], dict[str, dict[float, list[float]]], list[str]]:
    """
    Read each day's peaks.csv once for a fixed rotation under .../<day>/<fps>/<rotation>/.

    Returns:
      freq_counter: counts of how many times each rounded freq appears across all days
      day_bins: day -> (rounded_freq -> list of amp_psd values that rounded into that bin)
      days: ordered list of day folder names actually processed
    """
    freq_counter: Counter[float] = Counter()
    day_bins: dict[str, dict[float, list[float]]] = {}
    days_used: list[str] = []

    for day in list_day_dirs(base_path):
        fpath = os.path.join(base_path, day, fps, rotation_folder, PEAKS_FILENAME)
        if not os.path.exists(fpath):
            continue

        bins_for_day: dict[float, list[float]] = defaultdict(list)

        with open(fpath, "r", newline="") as f:
            reader = csv.DictReader(f)
            required = {"freq_hz", "amp_psd"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                missing = sorted(required - set(reader.fieldnames or []))
                raise ValueError(f"Missing columns {missing} in: {fpath}")

            for row in reader:
                freq_str = row.get("freq_hz")
                amp_str = row.get("amp_psd")
                if not freq_str or not amp_str:
                    continue

                raw_freq = float(freq_str)
                amp_psd = float(amp_str)

                rounded_freq = round(raw_freq, round_decimals)
                bins_for_day[rounded_freq].append(amp_psd)
                freq_counter[rounded_freq] += 1

        day_bins[day] = dict(bins_for_day)
        days_used.append(day)

    return freq_counter, day_bins, days_used


def select_frequencies(freq_counter: Counter[float], min_appearances: int) -> list[float]:
    """Keep only frequencies whose total appearance count is > min_appearances."""
    selected = [freq for freq, count in freq_counter.items() if count > min_appearances]
    selected.sort()  # low -> high for the y-axis
    return selected


def build_heatmap_matrix(
    freqs: list[float],
    days: list[str],
    day_bins: dict[str, dict[float, list[float]]],
    *,
    agg: str = "mean",
) -> np.ndarray:
    """
    Build matrix with shape (len(freqs), len(days)).

    Cell value = aggregate of amp_psd values for that (day, freq_bin).
    agg: "mean" or "max".
    """
    if agg not in ("mean", "max"):
        raise ValueError(f"agg must be 'mean' or 'max', got {agg!r}")

    mat = np.full((len(freqs), len(days)), np.nan, dtype=float)
    freq_index = {f: i for i, f in enumerate(freqs)}

    for j, day in enumerate(days):
        bins_for_day = day_bins.get(day, {})
        for f_bin, values in bins_for_day.items():
            i = freq_index.get(f_bin)
            if i is None:
                continue  # not in selected y-axis frequencies
            if values:
                mat[i, j] = float(np.mean(values) if agg == "mean" else np.max(values))

    return mat


def plot_heatmap(
    freqs: list[float],
    days: list[str],
    mat: np.ndarray,
    out_path: str,
    *,
    colorbar_label: str,
    title: str,
) -> None:
    """Plot and save the (frequency x day) heatmap."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"matplotlib is required to plot heatmap: {e}") from e

    if len(freqs) == 0 or len(days) == 0:
        print("No data to plot heatmap.")
        return

    # Size scales gently with the matrix.
    fig_w = max(8, min(0.55 * len(days), 20))
    fig_h = max(5, min(0.22 * len(freqs), 16))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # origin='lower' so the first row (low freq) is at the bottom.
    im = ax.imshow(mat, aspect="auto", origin="lower", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, rotation=90)

    ax.set_xlabel("day")
    ax.set_ylabel("freq_hz")

    # X ticks: label sparsely if there are many columns.
    if len(days) <= 30:
        x_ticks = np.arange(len(days))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(days, rotation=90, fontsize=8)
    else:
        step = max(1, len(days) // 25)
        x_ticks = np.arange(0, len(days), step)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([days[k] for k in x_ticks], rotation=90, fontsize=8)

    # Y ticks: label sparsely if there are many rows.
    if len(freqs) <= 40:
        y_ticks = np.arange(len(freqs))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{f:g}" for f in freqs], fontsize=8)
    else:
        step = max(1, len(freqs) // 25)
        y_ticks = np.arange(0, len(freqs), step)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{freqs[k]:g}" for k in y_ticks], fontsize=8)

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    base_path = DEFAULT_BASE_PATH

    for fps in DEFAULT_FPS_VALUES:
        out_dir = fps_output_dir(base_path, fps)
        os.makedirs(out_dir, exist_ok=True)

        rotations = discover_rotations(base_path, fps)
        if not rotations:
            print(
                f"No rotations with {PEAKS_FILENAME} found under {base_path}/*/{fps}/rotation_*/"
            )
            continue

        for rotation_folder in rotations:
            freq_counter, day_bins, days = load_day_bins(
                base_path, fps, rotation_folder, ROUND_DECIMALS
            )
            selected_freqs = select_frequencies(freq_counter, MIN_APPEARANCES)

            if not selected_freqs:
                print(
                    f"[{fps} / {rotation_folder}] No frequencies with total appearances > "
                    f"{MIN_APPEARANCES}; skip."
                )
                continue

            base_title = (
                f"{rotation_folder} @ {fps} fps — freq heatmap "
                f"(freq count > {MIN_APPEARANCES} appearances; rounded to {ROUND_DECIMALS} decimals)"
            )

            stem = f"rotation_freq_heatmap_{rotation_folder}"
            path_mean = os.path.join(out_dir, f"{stem}_mean.png")
            path_max = os.path.join(out_dir, f"{stem}_max.png")

            mat_mean = build_heatmap_matrix(selected_freqs, days, day_bins, agg="mean")
            plot_heatmap(
                selected_freqs,
                days,
                mat_mean,
                path_mean,
                colorbar_label="mean amp_psd for that day & freq bin",
                title=f"{base_title} — mean amp_psd",
            )
            print(f"Saved heatmap: {path_mean}")

            mat_max = build_heatmap_matrix(selected_freqs, days, day_bins, agg="max")
            plot_heatmap(
                selected_freqs,
                days,
                mat_max,
                path_max,
                colorbar_label="max amp_psd for that day & freq bin",
                title=f"{base_title} — max amp_psd",
            )
            print(f"Saved heatmap: {path_max}")


if __name__ == "__main__":
    main()

