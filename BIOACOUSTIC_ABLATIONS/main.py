from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import (
    AudioSegmentDataset,
    DatasetBalancedPKBatchSampler,
    assign_cv_folds,
    build_manifest,
    collate_audio_batch,
    filter_minimum_class_support,
    make_fold_manifest,
    split_class_counts,
)
from losses import build_focal_loss
from models import DEFAULT_CONFIG, MODEL_NAMES, MODEL_SPECS, build_model, count_parameters
from train import (
    aggregate_cross_validation,
    aggregate_group_metrics,
    collect_recording_outputs,
    fit_model,
    per_class_metrics,
    per_dataset_metrics,
    plot_accuracy_macro_f1,
    plot_fold_trajectories,
    plot_tsne_by_class,
)
from utils import choose_device, ensure_dir, parse_dataset_paths, save_json, seed_everything, worker_init_fn


DEFAULT_DATASET_PATHS = [
    "Watkins_Full_Cuts=/home/ardan/ARDAN/BIOACOUSTICS/Watkins_Full_Cuts",
    "Birds=/mnt/volDISI_conci_Datasets/audio/bioacoustics/birds2",
    "Frogs=/mnt/volDISI_conci_Datasets/audio/bioacoustics/frogs",
    "Bats=/mnt/volDISI_conci_Datasets/audio/bioacoustics/bats",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Five-fold multiresolution bioacoustic classification.")
    parser.add_argument("--dataset-paths", nargs="+", default=DEFAULT_DATASET_PATHS, metavar="NAME=/ABSOLUTE/PATH")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/bioacoustic_refactored_v11_stratified_globalval"))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--minimum-samples-per-class", type=int, default=15)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--eval-crops", type=int, default=3)
    parser.add_argument("--waveform-normalization", choices=["none", "peak", "rms"], default="peak")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument(
        "--contrastive-memory-size",
        type=int,
        default=0,
        help=(
            "Detached supervised-contrastive memory size. Zero automatically uses "
            "batch_size * (gradient_accumulation_steps - 1), so batch 16 + accumulation 2 "
            "uses up to 32 contrastive candidates after warm-up."
        ),
    )
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--train-samples-per-epoch", type=int, default=0)
    parser.add_argument("--max-rate", type=int, default=0, help="Override the corpus-maximum resampling rate. Zero uses the corpus maximum.")

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--frontend-lr-factor", type=float, default=0.35)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--minimum-delta", type=float, default=1e-4)

    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=0.10)
    parser.add_argument("--contrastive-temperature", type=float, default=0.10)
    parser.add_argument("--label-smoothing", type=float, default=0.01)
    parser.add_argument("--no-class-weights", action="store_true")

    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-fold1-tsne", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tsne-max-points", type=int, default=5000)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    return parser.parse_args()


def make_loader(
    dataset: AudioSegmentDataset,
    args: argparse.Namespace,
    training: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    if training:
        sampler = DatasetBalancedPKBatchSampler(
            dataset,
            batch_size=args.batch_size,
            samples_per_class=args.samples_per_class,
            seed=seed,
            num_samples=args.train_samples_per_epoch or None,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_audio_batch,
            worker_init_fn=worker_init_fn if args.num_workers else None,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_audio_batch,
        worker_init_fn=worker_init_fn if args.num_workers else None,
        generator=generator,
    )


def make_datasets(
    model_name: str,
    fold_manifest: pd.DataFrame,
    args: argparse.Namespace,
    corpus_max_rate: int,
) -> tuple[AudioSegmentDataset, AudioSegmentDataset, AudioSegmentDataset]:
    rate_mode = MODEL_SPECS[model_name]["rate_mode"]
    forced_rate = 48000 if rate_mode == "fixed_48k" else corpus_max_rate if rate_mode == "corpus_max" else None
    common = (args.segment_seconds, args.waveform_normalization, forced_rate)
    train = AudioSegmentDataset(fold_manifest, "train", common[0], True, 1, common[1], common[2])
    validation = AudioSegmentDataset(fold_manifest, "val", common[0], False, args.eval_crops, common[1], common[2])
    test = AudioSegmentDataset(fold_manifest, "test", common[0], False, args.eval_crops, common[1], common[2])
    return train, validation, test


def update_partial_reports(output_dir: Path, fold_rows: list[dict[str, Any]], fold_number: int) -> None:
    frame = pd.DataFrame(fold_rows)
    frame.to_csv(output_dir / "cv_fold_results_partial.csv", index=False)
    summary = aggregate_cross_validation(frame)
    summary.to_csv(output_dir / "cv_summary_partial.csv", index=False)
    plot_accuracy_macro_f1(summary, output_dir / "cv_accuracy_macro_f1_partial.png", f"Cross-validation through fold {fold_number}")

    fold1 = frame[frame["fold"] == 1].copy()
    if not fold1.empty:
        fold1.to_csv(output_dir / "intermediate_fold1_summary.csv", index=False)
        fold1_summary = aggregate_cross_validation(fold1)
        plot_accuracy_macro_f1(
            fold1_summary,
            output_dir / "intermediate_fold1_accuracy_macro_f1.png",
            "Fold 1 intermediate results",
        )


def run_one_model(
    model_name: str,
    fold_number: int,
    fold_manifest: pd.DataFrame,
    label_map: pd.DataFrame,
    corpus_max_rate: int,
    maximum_frequency_hz: float,
    args: argparse.Namespace,
    device: torch.device,
    fold_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    run_dir = ensure_dir(fold_dir / model_name)
    metrics_path = run_dir / "test_metrics.csv"
    if args.resume and metrics_path.is_file():
        summary = pd.read_csv(metrics_path).iloc[0].to_dict()
        return summary, pd.read_csv(run_dir / "test_per_dataset.csv"), pd.read_csv(run_dir / "test_per_class.csv")

    train_dataset, validation_dataset, test_dataset = make_datasets(model_name, fold_manifest, args, corpus_max_rate)
    run_seed = int(args.seed) + 1000 * int(fold_number) + MODEL_NAMES.index(model_name)
    seed_everything(run_seed)
    pin_memory = device.type == "cuda"
    train_loader = make_loader(train_dataset, args, True, run_seed, pin_memory)
    validation_loader = make_loader(validation_dataset, args, False, run_seed + 1, pin_memory)
    test_loader = make_loader(test_dataset, args, False, run_seed + 2, pin_memory)

    model = build_model(model_name, len(label_map), maximum_frequency_hz).to(device)
    parameter_info = count_parameters(model)
    pd.DataFrame([{**MODEL_SPECS[model_name], **parameter_info}]).to_csv(run_dir / "parameter_count.csv", index=False)

    class_counts = (
        fold_manifest[fold_manifest["split"] == "train"]["label_id"]
        .value_counts().reindex(range(len(label_map)), fill_value=0).to_numpy(dtype=np.float32)
    )
    focal_loss = build_focal_loss(
        torch.tensor(class_counts),
        device,
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        use_class_weights=not args.no_class_weights,
    )
    validation_loss = nn.CrossEntropyLoss()

    print(f"\n=== fold {fold_number:02d} / {model_name} ===")
    contrastive_memory_size = int(args.contrastive_memory_size)
    if contrastive_memory_size <= 0:
        contrastive_memory_size = (
            int(args.batch_size) * max(0, int(args.gradient_accumulation_steps) - 1)
        )

    fit_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        focal_loss=focal_loss,
        validation_loss=validation_loss,
        device=device,
        output_dir=run_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        frontend_lr_factor=args.frontend_lr_factor,
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
        patience=args.patience,
        minimum_delta=args.minimum_delta,
        grad_clip_norm=args.grad_clip_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        contrastive_memory_size=contrastive_memory_size,
        checkpoint_metadata={
            "fold": fold_number,
            "model": model_name,
            "num_classes": len(label_map),
            "corpus_max_rate": corpus_max_rate,
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "effective_optimizer_batch_size": int(args.batch_size) * int(args.gradient_accumulation_steps),
            "contrastive_memory_size": int(contrastive_memory_size),
        },
    )

    predictions, embeddings, test_metrics = collect_recording_outputs(model, test_loader, device, validation_loss)
    spec = MODEL_SPECS[model_name]
    summary = {
        "fold": fold_number,
        "model": model_name,
        **spec,
        **parameter_info,
        **test_metrics,
    }
    per_dataset = per_dataset_metrics(predictions)
    per_dataset.insert(0, "fold", fold_number)
    per_dataset.insert(1, "model", model_name)
    per_class = per_class_metrics(predictions, label_map)
    per_class.insert(0, "fold", fold_number)
    per_class.insert(1, "model", model_name)

    pd.DataFrame([summary]).to_csv(metrics_path, index=False)
    predictions.to_csv(run_dir / "test_predictions.csv", index=False)
    np.save(run_dir / "test_embeddings.npy", embeddings)
    per_dataset.to_csv(run_dir / "test_per_dataset.csv", index=False)
    per_class.to_csv(run_dir / "test_per_class.csv", index=False)

    if fold_number == 1 and args.save_fold1_tsne:
        colours = plot_tsne_by_class(
            embeddings, predictions, run_dir / "test_tsne_by_class.png",
            seed=run_seed, maximum_points=args.tsne_max_points, perplexity=args.tsne_perplexity,
        )
        colours.to_csv(run_dir / "test_tsne_class_colours.csv", index=False)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, per_dataset, per_class


def main() -> None:
    args = parse_args()
    if args.batch_size % args.samples_per_class:
        raise ValueError("batch-size must be divisible by samples-per-class")
    if args.samples_per_class < 2:
        raise ValueError("samples-per-class must be at least 2 for supervised contrastive training")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient-accumulation-steps must be at least 1")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    seed_everything(args.seed)
    device = choose_device(args.device)

    dataset_paths = parse_dataset_paths(args.dataset_paths)
    manifest = build_manifest(dataset_paths)
    manifest, retained_support, excluded_support = filter_minimum_class_support(manifest, args.minimum_samples_per_class)
    manifest = assign_cv_folds(manifest, args.num_folds, args.seed)
    manifest.to_csv(output_dir / "manifest_with_cv_folds.csv", index=False)
    retained_support.to_csv(output_dir / "retained_class_support.csv", index=False)
    excluded_support.to_csv(output_dir / "excluded_class_support.csv", index=False)

    label_map = manifest[["label_id", "label_name", "dataset", "class_name"]].drop_duplicates().sort_values("label_id").reset_index(drop=True)
    label_map.to_csv(output_dir / "label_map.csv", index=False)
    corpus_max_rate = int(args.max_rate or manifest["sample_rate"].max())
    maximum_frequency_hz = float(manifest["sample_rate"].max()) * 0.5 * float(DEFAULT_CONFIG["nyquist_rolloff"])
    save_json(
        {
            "models": args.models,
            "objective": "focal_plus_supervised_contrastive",
            "analysis_windows_ms": list(DEFAULT_CONFIG["analysis_windows_ms"]),
            "analysis_hops_ms": list(DEFAULT_CONFIG["analysis_hops_ms"]),
            "corpus_max_rate": corpus_max_rate,
            "micro_batch_size": int(args.batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "effective_optimizer_batch_size": int(args.batch_size) * int(args.gradient_accumulation_steps),
            "contrastive_memory_size": (
                int(args.contrastive_memory_size)
                if int(args.contrastive_memory_size) > 0
                else int(args.batch_size) * max(0, int(args.gradient_accumulation_steps) - 1)
            ),
            "device": str(device),
            "arguments": vars(args) | {"output_dir": str(args.output_dir)},
        },
        output_dir / "config.json",
    )

    fold_rows: list[dict[str, Any]] = []
    dataset_rows: list[pd.DataFrame] = []
    class_rows: list[pd.DataFrame] = []

    for fold_number in range(1, args.num_folds + 1):
        fold_manifest = make_fold_manifest(manifest, fold_number, args.validation_fraction, args.seed)
        fold_dir = ensure_dir(output_dir / f"fold_{fold_number:02d}")
        split_class_counts(fold_manifest).to_csv(
            fold_dir / "class_split_counts.csv", index=False
        )
        for model_name in args.models:
            summary, per_dataset, per_class = run_one_model(
                model_name, fold_number, fold_manifest, label_map,
                corpus_max_rate, maximum_frequency_hz, args, device, fold_dir,
            )
            fold_rows.append(summary)
            dataset_rows.append(per_dataset)
            class_rows.append(per_class)
            update_partial_reports(output_dir, fold_rows, fold_number)

        if fold_number == 1:
            (output_dir / "FOLD_1_INTERMEDIATE_RESULTS_READY.txt").write_text(
                "All selected fold-1 models are complete.\n", encoding="utf-8"
            )

    fold_results = pd.DataFrame(fold_rows)
    fold_results.to_csv(output_dir / "cv_fold_results.csv", index=False)
    summary = aggregate_cross_validation(fold_results)
    summary.to_csv(output_dir / "cv_summary.csv", index=False)
    plot_accuracy_macro_f1(summary, output_dir / "cv_accuracy_macro_f1_mean_std.png", "Five-fold classification results")
    plot_fold_trajectories(fold_results, output_dir / "cv_macro_f1_by_fold.png")

    all_dataset = pd.concat(dataset_rows, ignore_index=True)
    all_dataset.to_csv(output_dir / "cv_per_dataset_metrics.csv", index=False)
    aggregate_group_metrics(
        all_dataset,
        ["model", "dataset"],
        ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"],
    ).to_csv(output_dir / "cv_per_dataset_summary.csv", index=False)

    all_class = pd.concat(class_rows, ignore_index=True)
    all_class.to_csv(output_dir / "cv_per_class_metrics.csv", index=False)
    aggregate_group_metrics(
        all_class,
        ["model", "label_id", "label_name", "dataset", "class_name"],
        ["precision", "recall", "f1", "support"],
    ).to_csv(output_dir / "cv_per_class_summary.csv", index=False)


if __name__ == "__main__":
    main()
