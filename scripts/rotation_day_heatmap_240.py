"""
Same as rotation_day_heatmap.py but only for 240 fps,
filtering out ultra-low frequencies (< 0.6 Hz by default).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, "src")

from marajomodes.visualization.peaks_heatmap import (
    PEAKS_FILENAME,
    _freq_accepted,
    build_heatmap_matrix,
    collect_peak_freqs_per_day_pooled,
    discover_rotations,
    fps_output_dir,
    load_day_bins_pooled_rotations,
    plot_heatmap,
    plot_peak_frequency_evolution,
    select_frequencies,
)

DEFAULT_BASE_PATH = "out/processed/0.5/500"
FPS = "240"

MIN_APPEARANCES = 100
ROUND_DECIMALS = 6
MIN_FREQ = 0.6
EXCLUDE_BASE = 0.6
EXCLUDE_TOL = 0.05
TRIM_FRACTION = 0.1


def run(
    base_path: str,
    *,
    min_appearances: int,
    round_decimals: int,
    min_freq: float,
    exclude_base: float,
    exclude_tol: float,
) -> None:
    fps = FPS

    if not discover_rotations(base_path, fps):
        print(f"No {PEAKS_FILENAME} under {base_path}/*/{fps}/rotation_*/ — skip.")
        return

    (
        freq_counter,
        day_bins,
        days,
        presence_counts,
        rotations_per_day,
    ) = load_day_bins_pooled_rotations(base_path, fps, round_decimals)

    # filter out frequencies below min_freq and multiples of exclude_base
    def keep(f: float) -> bool:
        return _freq_accepted(
            f, min_freq=min_freq, exclude_base=exclude_base, exclude_tol=exclude_tol
        )

    freq_counter = {f: c for f, c in freq_counter.items() if keep(f)}
    for day in days:
        day_bins[day] = {f: v for f, v in day_bins[day].items() if keep(f)}

    selected_freqs = select_frequencies(freq_counter, min_appearances)
    if not selected_freqs:
        print(
            f"[{fps} fps] No frequencies >= {min_freq} Hz with total appearances > {min_appearances}; skip."
        )
        return

    out_dir = fps_output_dir(base_path, fps)
    os.makedirs(out_dir, exist_ok=True)

    base_title = (
        f"All rotations pooled @ {fps} fps — freq heatmap "
        f"(excl. multiples of {exclude_base} Hz ±{exclude_tol}, "
        f"count > {min_appearances}; rounded to {round_decimals} dec)"
    )

    stem = "rotation_freq_heatmap_pooled_filtered"
    path_mean = os.path.join(out_dir, f"{stem}_mean.png")
    path_max = os.path.join(out_dir, f"{stem}_max.png")
    path_var = os.path.join(out_dir, f"{stem}_var.png")
    path_trim = os.path.join(out_dir, f"{stem}_trimmed_mean.png")
    path_count = os.path.join(out_dir, f"{stem}_count.png")

    mat_mean = build_heatmap_matrix(selected_freqs, days, day_bins, agg="mean")
    plot_heatmap(
        selected_freqs,
        days,
        mat_mean,
        path_mean,
        colorbar_label="mean amp_psd (all rotations pooled) & freq bin",
        title=f"{base_title} — mean amp_psd",
        xlabel="day",
    )
    print(f"Saved heatmap: {path_mean}")

    mat_max = build_heatmap_matrix(selected_freqs, days, day_bins, agg="max")
    plot_heatmap(
        selected_freqs,
        days,
        mat_max,
        path_max,
        colorbar_label="max amp_psd (all rotations pooled) & freq bin",
        title=f"{base_title} — max amp_psd",
        xlabel="day",
    )
    print(f"Saved heatmap: {path_max}")

    mat_var = build_heatmap_matrix(selected_freqs, days, day_bins, agg="var")
    plot_heatmap(
        selected_freqs,
        days,
        mat_var,
        path_var,
        colorbar_label="variance amp_psd (all rotations pooled) & freq bin",
        title=f"{base_title} — var amp_psd",
        xlabel="day",
    )
    print(f"Saved heatmap: {path_var}")

    mat_trim = build_heatmap_matrix(
        selected_freqs,
        days,
        day_bins,
        agg="trimmed_mean",
        trim_fraction=TRIM_FRACTION,
    )
    plot_heatmap(
        selected_freqs,
        days,
        mat_trim,
        path_trim,
        colorbar_label=f"trimmed_mean amp_psd (trim={TRIM_FRACTION}) & freq bin",
        title=f"{base_title} — trimmed mean amp_psd",
        xlabel="day",
    )
    print(f"Saved heatmap: {path_trim}")

    mat_count = build_heatmap_matrix(selected_freqs, days, day_bins, agg="count")
    plot_heatmap(
        selected_freqs,
        days,
        mat_count,
        path_count,
        colorbar_label="count of peaks mapped to freq bin (all rotations pooled)",
        title=f"{base_title} — count of peaks per day & freq bin",
        xlabel="day",
    )
    print(f"Saved heatmap: {path_count}")

    evo_days, evo_freqs = collect_peak_freqs_per_day_pooled(
        base_path, fps, min_freq=min_freq, exclude_base=exclude_base, exclude_tol=exclude_tol
    )
    path_evolution = os.path.join(out_dir, f"{stem}_peak_freq_evolution.png")
    plot_peak_frequency_evolution(
        evo_days,
        evo_freqs,
        path_evolution,
        title=(
            f"All rotations pooled @ {fps} fps (excl. multiples of {exclude_base} Hz) — "
            "Evolução das frequências de maior intensidade ao longo dos dias"
        ),
    )
    print(f"Saved evolution plot: {path_evolution}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Day × frequency heatmaps for 240 fps only, filtering out "
            "ultra-low frequencies (default < 0.6 Hz)."
        )
    )
    parser.add_argument(
        "--base-path",
        default=DEFAULT_BASE_PATH,
        help="Processed root, e.g. out/processed/0.5/500",
    )
    parser.add_argument(
        "--min-appearances",
        type=int,
        default=MIN_APPEARANCES,
        help="Keep rounded freqs with total count > this.",
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=ROUND_DECIMALS,
        help="Rounding for freq_hz binning.",
    )
    parser.add_argument(
        "--min-freq",
        type=float,
        default=MIN_FREQ,
        help="Ignore frequencies below this value in Hz (default: 0.6).",
    )
    parser.add_argument(
        "--exclude-base",
        type=float,
        default=EXCLUDE_BASE,
        help="Exclude multiples of this frequency in Hz (default: 0.6).",
    )
    parser.add_argument(
        "--exclude-tol",
        type=float,
        default=EXCLUDE_TOL,
        help="Tolerance for multiple detection (default: 0.05).",
    )
    args = parser.parse_args()

    run(
        args.base_path,
        min_appearances=args.min_appearances,
        round_decimals=args.round_decimals,
        min_freq=args.min_freq,
        exclude_base=args.exclude_base,
        exclude_tol=args.exclude_tol,
    )


if __name__ == "__main__":
    main()
