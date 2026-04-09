import os
import sys

sys.path.insert(0, "src")

from marajomodes.visualization.peaks_heatmap import (
    PEAKS_FILENAME,
    build_heatmap_matrix,
    collect_peak_freqs_per_day,
    discover_rotations,
    fps_output_dir,
    load_day_bins,
    plot_heatmap,
    plot_peak_frequency_evolution,
    select_frequencies,
)

import argparse

DEFAULT_BASE_PATH = "out/processed/0.5/500"
DEFAULT_FPS_VALUES = ("60", "240")

MIN_APPEARANCES = 60
ROUND_DECIMALS = 6
TRIM_FRACTION = 0.1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rotation-wise heatmaps (x = day, y = freq_hz) for every rotation_*/peaks.csv "
            "under a processed base path. Generates mean/max/var/trimmed_mean/count."
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
        help="Keep rounded freqs with total count > this (per rotation).",
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=ROUND_DECIMALS,
        help="Rounding for freq_hz binning.",
    )
    parser.add_argument(
        "--trim-fraction",
        type=float,
        default=TRIM_FRACTION,
        help="Trim fraction for trimmed-mean heatmap (0.1 = drop 10%% from both tails).",
    )
    args = parser.parse_args()

    base_path = args.base_path
    fps_values = tuple(args.fps)
    min_appearances = args.min_appearances
    round_decimals = args.round_decimals
    trim_fraction = args.trim_fraction

    for fps in fps_values:
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
                base_path, fps, rotation_folder, round_decimals
            )
            selected_freqs = select_frequencies(freq_counter, min_appearances)

            if not selected_freqs:
                print(
                    f"[{fps} / {rotation_folder}] No frequencies with total appearances > "
                    f"{min_appearances}; skip."
                )
                continue

            base_title = (
                f"{rotation_folder} @ {fps} fps — freq heatmap "
                f"(freq count > {min_appearances} appearances; rounded to {round_decimals} decimals)"
            )

            stem = f"rotation_freq_heatmap_{rotation_folder}"
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
                colorbar_label="mean amp_psd for that day & freq bin",
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
                colorbar_label="max amp_psd for that day & freq bin",
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
                colorbar_label="variance amp_psd for that day & freq bin",
                title=f"{base_title} — var amp_psd",
                xlabel="day",
            )
            print(f"Saved heatmap: {path_var}")

            mat_trim = build_heatmap_matrix(
                selected_freqs,
                days,
                day_bins,
                agg="trimmed_mean",
                trim_fraction=trim_fraction,
            )
            plot_heatmap(
                selected_freqs,
                days,
                mat_trim,
                path_trim,
                colorbar_label=f"trimmed_mean amp_psd (trim={trim_fraction}) for that day & freq bin",
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
                colorbar_label="count of peaks mapped to freq bin for that day",
                title=f"{base_title} — count of peaks per day & freq bin",
                xlabel="day",
            )
            print(f"Saved heatmap: {path_count}")

            evo_days, evo_freqs = collect_peak_freqs_per_day(
                base_path, fps, rotation_folder
            )
            path_evolution = os.path.join(out_dir, f"{stem}_peak_freq_evolution.png")
            plot_peak_frequency_evolution(
                evo_days,
                evo_freqs,
                path_evolution,
                title=(
                    f"{rotation_folder} @ {fps} fps — "
                    "Evolução das frequências de maior intensidade ao longo dos dias"
                ),
            )
            print(f"Saved evolution plot: {path_evolution}")


if __name__ == "__main__":
    main()
