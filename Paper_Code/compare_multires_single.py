from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiresolution and single-resolution CV summaries by model."
    )
    parser.add_argument("--multires-dir", type=Path, required=True)
    parser.add_argument("--single-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("multires_vs_single_comparison.csv")
    )
    return parser.parse_args()


def load_summary(directory: Path, variant: str) -> pd.DataFrame:
    path = directory.expanduser().resolve() / "cv_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing cross-validation summary: {path}")
    frame = pd.read_csv(path)
    frame.insert(0, "experiment_variant", variant)
    return frame


def main() -> None:
    args = parse_args()
    multires = load_summary(args.multires_dir, "multiresolution_temporal_pyramid")
    single = load_summary(args.single_dir, "single_resolution_64ms")
    combined = pd.concat([multires, single], ignore_index=True)
    metrics = [
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "balanced_accuracy_mean",
        "mean_dataset_macro_f1_mean",
        "min_dataset_macro_f1_mean",
        "frontend_and_model_ms_per_crop_mean",
        "total_parameters",
    ]
    available = [column for column in metrics if column in combined.columns]
    comparison = combined.pivot_table(
        index=["run_name", "model", "training_regime"],
        columns="experiment_variant",
        values=available,
        aggfunc="first",
    )
    comparison.columns = [f"{metric}__{variant}" for metric, variant in comparison.columns]
    comparison = comparison.reset_index()
    multi_column = "macro_f1_mean__multiresolution_temporal_pyramid"
    single_column = "macro_f1_mean__single_resolution_64ms"
    if multi_column in comparison and single_column in comparison:
        comparison["macro_f1_multires_minus_single"] = (
            comparison[multi_column] - comparison[single_column]
        )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output, index=False)
    print(f"Saved comparison: {output}")


if __name__ == "__main__":
    main()
