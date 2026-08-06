from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "loss",
    "mean_dataset_macro_f1",
    "min_dataset_macro_f1",
    "frontend_and_model_ms_per_crop",
]

PARAMETER_COLUMNS = [
    "total_parameters",
    "trainable_parameters",
    "frontend_parameters",
    "encoder_parameters",
    "classifier_parameters",
    "band_fusion_parameters",
    "contrastive_head_parameters",
    "model_size_fp32_mb",
]


def plot_history(history: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(history["epoch"], history["train_loss"], label="train")
    axis.plot(history["epoch"], history["val_loss"], label="validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_accuracy_macro_f1(
    frame: pd.DataFrame,
    output_path: Path,
    title: str,
    error_suffix: str | None = None,
) -> None:
    """Plot accuracy and macro-F1, optionally with standard-deviation bars."""
    if frame.empty:
        return
    data = frame.copy()
    accuracy_column = "accuracy_mean" if "accuracy_mean" in data else "accuracy"
    f1_column = "macro_f1_mean" if "macro_f1_mean" in data else "macro_f1"
    data = data.sort_values(f1_column, ascending=True, kind="stable").reset_index(drop=True)
    labels = data["run_name"].astype(str).tolist()
    accuracy = data[accuracy_column].to_numpy(dtype=float)
    macro_f1 = data[f1_column].to_numpy(dtype=float)
    accuracy_error = None
    f1_error = None
    if error_suffix:
        accuracy_error = data.get(f"accuracy_{error_suffix}")
        f1_error = data.get(f"macro_f1_{error_suffix}")
        accuracy_error = None if accuracy_error is None else accuracy_error.to_numpy(dtype=float)
        f1_error = None if f1_error is None else f1_error.to_numpy(dtype=float)

    positions = np.arange(len(data))
    height = 0.38
    figure_height = max(6.5, 0.55 * len(data) + 1.8)
    figure, axis = plt.subplots(figsize=(14, figure_height))
    bars_accuracy = axis.barh(
        positions - height / 2,
        accuracy,
        height=height,
        xerr=accuracy_error,
        capsize=3 if accuracy_error is not None else 0,
        label="Accuracy",
    )
    bars_f1 = axis.barh(
        positions + height / 2,
        macro_f1,
        height=height,
        xerr=f1_error,
        capsize=3 if f1_error is not None else 0,
        label="Macro-F1",
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xlabel("Score")
    axis.set_title(title)
    axis.legend()
    axis.grid(axis="x", alpha=0.25)
    minimum = float(min(accuracy.min(), macro_f1.min()))
    lower = max(0.0, np.floor((minimum - 0.04) * 20.0) / 20.0)
    upper = min(1.0, float(max(accuracy.max(), macro_f1.max())) + 0.05)
    axis.set_xlim(lower, upper)
    padding = max((upper - lower) * 0.006, 0.002)
    for bars in (bars_accuracy, bars_f1):
        for bar in bars:
            value = float(bar.get_width())
            axis.text(
                value + padding,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                fontsize=7,
            )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(figure)


def aggregate_cross_validation(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one row per fold into mean, std, SEM, and 95% CI columns."""
    if fold_results.empty:
        return pd.DataFrame()
    group_columns = ["run_name", "model", "training_regime"]
    rows: list[dict[str, object]] = []
    for keys, group in fold_results.groupby(group_columns, sort=False):
        row: dict[str, object] = dict(zip(group_columns, keys))
        row["num_folds"] = int(group["fold"].nunique())
        for metric in METRIC_COLUMNS:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            sem = std / np.sqrt(len(values)) if len(values) else float("nan")
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_sem"] = sem
            row[f"{metric}_ci95"] = 1.96 * sem
        for column in PARAMETER_COLUMNS:
            if column in group:
                row[column] = float(group[column].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_fold_trajectories(fold_results: pd.DataFrame, output_path: Path) -> None:
    if fold_results.empty:
        return
    figure, axis = plt.subplots(figsize=(12, 7))
    for run_name, group in fold_results.groupby("run_name", sort=False):
        ordered = group.sort_values("fold")
        axis.plot(ordered["fold"], ordered["macro_f1"], marker="o", label=run_name)
    axis.set_xlabel("Cross-validation fold")
    axis.set_ylabel("Macro-F1")
    axis.set_xticks(sorted(fold_results["fold"].unique()))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
