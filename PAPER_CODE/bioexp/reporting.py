from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FACTORIAL_COMPARISONS = [
    ("FNO effect: mel single", "mel_single_fno", "mel_single_cnn"),
    ("FNO effect: mel multires", "mel_multires_fno", "mel_multires_cnn"),
    ("FNO effect: physical single", "sfi_single_fno", "sfi_single_cnn"),
    ("FNO effect: physical multires", "sfi_multires_fno", "sfi_multires_cnn"),
    ("Multires effect: mel CNN", "mel_multires_cnn", "mel_single_cnn"),
    ("Multires effect: mel FNO", "mel_multires_fno", "mel_single_fno"),
    ("Multires effect: physical CNN", "sfi_multires_cnn", "sfi_single_cnn"),
    ("Multires effect: physical FNO", "sfi_multires_fno", "sfi_single_fno"),
    ("Physical frontend effect: single CNN", "sfi_single_cnn", "mel_single_cnn"),
    ("Physical frontend effect: single FNO", "sfi_single_fno", "mel_single_fno"),
    ("Physical frontend effect: multires CNN", "sfi_multires_cnn", "mel_multires_cnn"),
    ("Physical frontend effect: multires FNO", "sfi_multires_fno", "mel_multires_fno"),
    (
        "Two-band equivariance effect",
        "sfi_multires_fno_bandwidth_equivariant",
        "sfi_multires_fno",
    ),
    (
        "Three-band equivariance effect",
        "sfi_multires_fno_three_bandwidth_equivariant",
        "sfi_multires_fno",
    ),
    (
        "Three-band versus two-band",
        "sfi_multires_fno_three_bandwidth_equivariant",
        "sfi_multires_fno_bandwidth_equivariant",
    ),
]


def make_contrasts(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_names = [
        "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
        "mean_dataset_macro_f1", "min_dataset_macro_f1",
    ]
    rows = []
    for regime, group in metrics.groupby("training_regime", sort=False):
        indexed = group.set_index("model")
        for name, positive, negative in FACTORIAL_COMPARISONS:
            if positive not in indexed.index or negative not in indexed.index:
                continue
            row = {
                "training_regime": regime,
                "contrast": name,
                "positive_model": positive,
                "negative_model": negative,
            }
            for metric in metric_names:
                row[f"delta_{metric}"] = float(
                    indexed.loc[positive, metric] - indexed.loc[negative, metric]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def make_objective_contrasts(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_names = [
        "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
        "mean_dataset_macro_f1", "min_dataset_macro_f1",
    ]
    rows = []
    for model, group in metrics.groupby("model", sort=False):
        indexed = group.set_index("training_regime")
        if not {"cross_entropy", "focal_contrastive"}.issubset(indexed.index):
            continue
        row = {"model": model}
        for metric in metric_names:
            row[f"delta_{metric}"] = float(
                indexed.loc["focal_contrastive", metric]
                - indexed.loc["cross_entropy", metric]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_summary(metrics: pd.DataFrame, output_path: Path) -> None:
    frame = metrics.copy()
    frame["label"] = frame["model"] + "\n" + frame["training_regime"]
    figure, axis = plt.subplots(figsize=(16, 7))
    axis.bar(range(len(frame)), frame["macro_f1"])
    axis.set_xticks(range(len(frame)))
    axis.set_xticklabels(frame["label"], rotation=70, ha="right", fontsize=8)
    axis.set_ylabel("Test macro-F1")
    axis.set_title("All 22 model/objective combinations")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_history(history: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(history["epoch"], history["train_loss"], label="train")
    axis.plot(history["epoch"], history["val_loss"], label="validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
