"""
Same heatmap layout as ``rotation_heatmap.py`` (x = day, y = freq_hz, color = mean/max amp_psd),
but each day pools **all** ``rotation_*/peaks.csv`` under that day's fps folder.

Frequency ranking (``> min_appearances``) uses counts summed over **every day and every rotation**.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, "src")

from marajomodes.visualization.peaks_heatmap import (
    PEAKS_FILENAME,
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
DEFAULT_FPS_VALUES = ("60", "240")

MIN_APPEARANCES = 300
ROUND_DECIMALS = 6

TRIM_FRACTION = 0.1


def run_pooled_heatmaps(
    base_path: str,
    fps_values: tuple[str, ...],
    *,
    min_appearances: int,
    round_decimals: int,
) -> None:
    for fps in fps_values:
        if not discover_rotations(base_path, fps):
            print(
                f"No {PEAKS_FILENAME} under {base_path}/*/{fps}/rotation_*/ — skip."
            )
            continue

        (
            freq_counter,
            day_bins,
            days,
            presence_counts,
            rotations_per_day,
        ) = load_day_bins_pooled_rotations(base_path, fps, round_decimals)
        selected_freqs = select_frequencies(freq_counter, min_appearances)
        if not selected_freqs:
            print(
                f"[{fps} fps] No frequencies with total appearances > {min_appearances}; skip."
            )
            continue

        out_dir = fps_output_dir(base_path, fps)
        os.makedirs(out_dir, exist_ok=True)

        base_title = (
            f"All rotations pooled @ {fps} fps — freq heatmap "
            f"(freq count > {min_appearances} appearances; rounded to {round_decimals} decimals)"
        )

        stem = "rotation_freq_heatmap_pooled"
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

        evo_days, evo_freqs = collect_peak_freqs_per_day_pooled(base_path, fps)
        path_evolution = os.path.join(out_dir, f"{stem}_peak_freq_evolution.png")
        plot_peak_frequency_evolution(
            evo_days,
            evo_freqs,
            path_evolution,
            title=(
                f"All rotations pooled @ {fps} fps — "
                "Evolução das frequências de maior intensidade ao longo dos dias"
            ),
        )
        print(f"Saved evolution plot: {path_evolution}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Day × frequency heatmaps with all rotations pooled per day (same layout as "
            "rotation_heatmap.py, one figure per fps)."
        )
    )
    parser.add_argument(
        "--base-path",
        default=DEFAULT_BASE_PATH,
        help="Processed root for scale+frames, e.g. out/processed/0.5/500",
    )
    parser.add_argument(
        "--fps",
        nargs="+",
        default=list(DEFAULT_FPS_VALUES),
        help="Fps folder name(s), e.g. 60 240 (default: both).",
    )
    parser.add_argument(
        "--min-appearances",
        type=int,
        default=MIN_APPEARANCES,
        help="Keep rounded freqs with total count > this (all days, all rotations).",
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=ROUND_DECIMALS,
        help="Rounding for freq_hz binning.",
    )
    args = parser.parse_args()

    run_pooled_heatmaps(
        args.base_path,
        tuple(args.fps),
        min_appearances=args.min_appearances,
        round_decimals=args.round_decimals,
    )


if __name__ == "__main__":
    main()
