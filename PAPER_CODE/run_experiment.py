from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from bioexp.data import (
    AudioSegmentDataset,
    CounterfactualView,
    NestedRateAudioDataset,
    collate_audio_batch,
    collate_nested_batch,
)
from bioexp.evaluation import collect_recording_outputs, per_class_metrics, per_dataset_metrics
from bioexp.losses import OBJECTIVE_NAMES, ObjectiveConfig, build_classification_loss
from bioexp.manifest import build_manifest, split_counts, stratified_recording_split
from bioexp.models import count_parameters
from bioexp.registry import (
    MODEL_NAMES,
    MODEL_SPECS,
    ArchitectureConfig,
    build_model,
    model_table,
)
from bioexp.reporting import (
    make_contrasts,
    make_objective_contrasts,
    plot_history,
    plot_summary,
)
from bioexp.samplers import DatasetBalancedPKBatchSampler
from bioexp.trainer import fit_model
from bioexp.utils import choose_device, ensure_dir, save_json, seed_everything, worker_init_fn


DEFAULT_MODELS = MODEL_NAMES
DEFAULT_OBJECTIVES = OBJECTIVE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 11 modular architectures under CE and focal+SupCon (22 runs)."
    )
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/volDISI_conci_Datasets/audio/bioacoustics"))
    parser.add_argument("--datasets", nargs="+", default=["watkins"])
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ablations_WATK"))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=DEFAULT_MODELS)
    parser.add_argument("--objectives", nargs="+", choices=OBJECTIVE_NAMES, default=DEFAULT_OBJECTIVES)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--estimate-bandwidths", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bandwidth-scan-seconds", type=float, default=2.0)
    parser.add_argument("--bandwidth-energy-quantile", type=float, default=0.995)

    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--eval-crops", type=int, default=3)
    parser.add_argument("--waveform-normalization", choices=["none", "peak", "rms"], default="peak")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--train-samples-per-epoch", type=int, default=0)

    parser.add_argument("--single-window-ms", type=float, default=64.0)
    parser.add_argument("--multiresolution-window-ms", nargs="+", type=float, default=[16.0, 64.0, 256.0])
    parser.add_argument("--frequency-bins", type=int, default=96)
    parser.add_argument("--time-bins", type=int, default=96)
    parser.add_argument("--minimum-frequency-hz", type=float, default=5.0)
    parser.add_argument("--maximum-frequency-hz", type=float, default=0.0)
    parser.add_argument("--q-factor", type=float, default=12.0)
    parser.add_argument("--modulation-rates-hz", nargs="+", type=float, default=[2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--nyquist-rolloff", type=float, default=0.92)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--fno-modes-frequency", type=int, default=8)
    parser.add_argument("--fno-modes-time", type=int, default=8)
    parser.add_argument("--token-dim", type=int, default=96)
    parser.add_argument("--token-layers", type=int, default=2)
    parser.add_argument("--token-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--contrastive-projection-dim", type=int, default=128)

    parser.add_argument("--two-band-cutoff-hz", type=float, default=20000.0)
    parser.add_argument("--two-band-counterfactual-rate", type=int, default=48000)
    parser.add_argument("--three-band-low-cutoff-hz", type=float, default=8000.0)
    parser.add_argument("--three-band-mid-cutoff-hz", type=float, default=20000.0)
    parser.add_argument("--three-band-low-rate", type=int, default=24000)
    parser.add_argument("--three-band-mid-rate", type=int, default=48000)
    parser.add_argument("--residual-gate-bias", type=float, default=-1.0)
    parser.add_argument("--counterfactual-weight", type=float, default=0.5)
    parser.add_argument("--equivariance-weight", type=float, default=0.25)

    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha-mode", choices=["none", "effective_num"], default="none")
    parser.add_argument("--contrastive-weight", type=float, default=0.10)
    parser.add_argument("--contrastive-temperature", type=float, default=0.10)
    parser.add_argument("--label-smoothing", type=float, default=0.01)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--frontend-lr-factor", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--minimum-delta", type=float, default=1e-4)
    parser.add_argument("--early-stopping-metric", choices=["global_macro_f1", "mean_dataset_macro_f1"], default="mean_dataset_macro_f1")
    parser.add_argument("--device", default="auto")

    parser.add_argument("--dataset-mel-candidate-rates", nargs="+", type=int, default=[8000, 12000, 16000, 24000, 32000, 48000, 64000, 96000, 128000, 192000, 250000, 300000, 384000, 500000])
    parser.add_argument("--dataset-mel-bandwidth-quantile", type=float, default=0.99)
    parser.add_argument("--dataset-mel-safety-factor", type=float, default=1.10)
    parser.add_argument("--dataset-mel-rates-json", type=Path, default=None)
    return parser.parse_args()


def resolve_dataset_mel_rates(args: argparse.Namespace, manifest: pd.DataFrame) -> dict[str, int]:
    manual: dict[str, int] = {}
    if args.dataset_mel_rates_json:
        with args.dataset_mel_rates_json.open("r", encoding="utf-8") as handle:
            manual = {str(k): int(v) for k, v in json.load(handle).items()}
    candidates = sorted({int(rate) for rate in args.dataset_mel_candidate_rates if rate > 0})
    mapping: dict[str, int] = {}
    training = manifest[manifest["split"] == "train"]
    for dataset, group in training.groupby("dataset", sort=True):
        name = str(dataset)
        if name in manual:
            mapping[name] = manual[name]
            continue
        bandwidth = float(group["usable_bandwidth_hz"].quantile(args.dataset_mel_bandwidth_quantile))
        required = int(math.ceil(2.0 * args.dataset_mel_safety_factor * bandwidth))
        maximum_native = int(group["sample_rate"].max())
        possible = sorted(set([rate for rate in candidates if rate <= maximum_native] + [maximum_native]))
        valid = [rate for rate in possible if rate >= required]
        mapping[name] = int(valid[0] if valid else maximum_native)
    return mapping


def make_loader(
    dataset,
    batch_size: int,
    samples_per_class: int,
    training: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    train_samples_per_epoch: int,
):
    collate = collate_nested_batch if isinstance(dataset, NestedRateAudioDataset) else collate_audio_batch
    generator = torch.Generator().manual_seed(seed)
    if training:
        batch_sampler = DatasetBalancedPKBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            samples_per_class=samples_per_class,
            seed=seed,
            num_samples=train_samples_per_epoch or None,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate,
            worker_init_fn=worker_init_fn if num_workers else None,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
        worker_init_fn=worker_init_fn if num_workers else None,
        generator=generator,
    )


def nested_views(model_name: str, args: argparse.Namespace) -> list[CounterfactualView]:
    if model_name == "sfi_multires_fno_bandwidth_equivariant":
        return [
            CounterfactualView(
                "shared_20k",
                args.two_band_cutoff_hz,
                args.two_band_counterfactual_rate,
                max_band=0,
            )
        ]
    if model_name == "sfi_multires_fno_three_bandwidth_equivariant":
        return [
            CounterfactualView(
                "low_8k",
                args.three_band_low_cutoff_hz,
                args.three_band_low_rate,
                max_band=0,
            ),
            CounterfactualView(
                "low_mid_20k",
                args.three_band_mid_cutoff_hz,
                args.three_band_mid_rate,
                max_band=1,
            ),
        ]
    return []


def make_datasets(
    model_name: str,
    manifest: pd.DataFrame,
    args: argparse.Namespace,
    dataset_mel_rates: dict[str, int],
):
    views = nested_views(model_name, args)
    if views:
        train = NestedRateAudioDataset(
            manifest, "train", args.segment_seconds, True, 1,
            args.waveform_normalization, views,
        )
    else:
        rates = dataset_mel_rates if MODEL_SPECS[model_name].uses_dataset_resampling else None
        train = AudioSegmentDataset(
            manifest, "train", args.segment_seconds, True, 1,
            args.waveform_normalization, dataset_output_rates=rates,
        )
    rates = dataset_mel_rates if MODEL_SPECS[model_name].uses_dataset_resampling else None
    validation = AudioSegmentDataset(
        manifest, "val", args.segment_seconds, False, args.eval_crops,
        args.waveform_normalization, dataset_output_rates=rates,
    )
    test = AudioSegmentDataset(
        manifest, "test", args.segment_seconds, False, args.eval_crops,
        args.waveform_normalization, dataset_output_rates=rates,
    )
    return train, validation, test


def main() -> None:
    args = parse_args()
    if args.batch_size % args.samples_per_class:
        raise ValueError("batch-size must be divisible by samples-per-class")
    device = choose_device(args.device)
    pin_memory = device.type == "cuda"
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)

    seed_everything(args.seed)
    if args.split_csv:
        manifest = pd.read_csv(args.split_csv)
        if "class_name" not in manifest:
            manifest["class_name"] = manifest["label_name"].astype(str).str.split("/", n=1).str[-1]
    else:
        manifest = build_manifest(
            args.data_root,
            args.datasets,
            args.estimate_bandwidths,
            args.bandwidth_scan_seconds,
            args.bandwidth_energy_quantile,
        )
        manifest = stratified_recording_split(
            manifest, args.val_fraction, args.test_fraction, args.seed
        )
    manifest.to_csv(output_dir / "manifest_with_splits.csv", index=False)
    split_counts(manifest).to_csv(output_dir / "split_counts.csv", index=False)
    label_map = (
        manifest[["label_id", "label_name", "dataset", "class_name"]]
        .drop_duplicates().sort_values("label_id").reset_index(drop=True)
    )
    label_map.to_csv(output_dir / "label_map.csv", index=False)
    num_classes = len(label_map)

    dataset_rates = resolve_dataset_mel_rates(args, manifest)
    pd.DataFrame(
        [{"dataset": key, "target_sample_rate": value} for key, value in sorted(dataset_rates.items())]
    ).to_csv(output_dir / "dataset_mel_rates.csv", index=False)
    pd.DataFrame(model_table()).to_csv(output_dir / "model_registry.csv", index=False)

    maximum_frequency = float(args.maximum_frequency_hz or manifest["sample_rate"].max() / 2.0)
    architecture = ArchitectureConfig(
        single_window_ms=args.single_window_ms,
        multiresolution_windows_ms=tuple(args.multiresolution_window_ms),
        frequency_bins=args.frequency_bins,
        time_bins=args.time_bins,
        minimum_frequency_hz=args.minimum_frequency_hz,
        maximum_frequency_hz=maximum_frequency,
        q_factor=args.q_factor,
        modulation_rates_hz=tuple(args.modulation_rates_hz),
        nyquist_rolloff=args.nyquist_rolloff,
        base_channels=args.base_channels,
        embedding_dim=args.embedding_dim,
        depth=args.depth,
        fno_modes_frequency=args.fno_modes_frequency,
        fno_modes_time=args.fno_modes_time,
        token_dim=args.token_dim,
        token_layers=args.token_layers,
        token_heads=args.token_heads,
        dropout=args.dropout,
        two_band_cutoff_hz=args.two_band_cutoff_hz,
        three_band_low_cutoff_hz=args.three_band_low_cutoff_hz,
        three_band_mid_cutoff_hz=args.three_band_mid_cutoff_hz,
        residual_gate_bias=args.residual_gate_bias,
        contrastive_projection_dim=args.contrastive_projection_dim,
    )
    save_json(
        {
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "architecture": asdict(architecture),
            "dataset_mel_rates": dataset_rates,
            "device": str(device),
        },
        output_dir / "config.json",
    )

    class_counts = (
        manifest[manifest["split"] == "train"]["label_id"]
        .value_counts().reindex(range(num_classes), fill_value=0)
        .to_numpy(dtype=np.float32)
    )
    class_counts_tensor = torch.tensor(class_counts)
    validation_loss = nn.CrossEntropyLoss()

    summary_rows: list[dict[str, Any]] = []
    dataset_rows: list[pd.DataFrame] = []
    class_rows: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, Any]] = []

    for model_name in args.models:
        train_dataset, validation_dataset, test_dataset = make_datasets(
            model_name, manifest, args, dataset_rates
        )
        for objective_name in args.objectives:
            run_name = f"{model_name}__{objective_name}"
            print(f"\n=== {run_name} ===")
            run_dir = ensure_dir(output_dir / run_name)
            seed_everything(args.seed)
            model = build_model(
                model_name=model_name,
                num_classes=num_classes,
                config=architecture,
                use_contrastive_head=objective_name == "focal_contrastive",
            ).to(device)
            with (run_dir / "model_architecture.txt").open("w", encoding="utf-8") as handle:
                handle.write(str(model))
                handle.write("\n")
            parameter_info = count_parameters(model)
            parameter_row = {
                "run_name": run_name,
                "model": model_name,
                "training_regime": objective_name,
                **parameter_info,
            }
            parameter_rows.append(parameter_row)
            pd.DataFrame([parameter_row]).to_csv(run_dir / "parameter_count.csv", index=False)

            train_loader = make_loader(
                train_dataset, args.batch_size, args.samples_per_class, True,
                args.num_workers, pin_memory, args.seed,
                args.train_samples_per_epoch,
            )
            validation_loader = make_loader(
                validation_dataset, args.batch_size, args.samples_per_class, False,
                args.num_workers, pin_memory, args.seed + 1, 0,
            )
            test_loader = make_loader(
                test_dataset, args.batch_size, args.samples_per_class, False,
                args.num_workers, pin_memory, args.seed + 2, 0,
            )
            objective = ObjectiveConfig(
                name=objective_name,
                focal_gamma=args.focal_gamma,
                contrastive_weight=args.contrastive_weight,
                contrastive_temperature=args.contrastive_temperature,
                label_smoothing=args.label_smoothing,
                focal_alpha_mode=args.focal_alpha_mode,
            )
            classification_loss = build_classification_loss(
                objective, class_counts_tensor, device
            )
            history = fit_model(
                model=model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                classification_loss=classification_loss,
                validation_loss=validation_loss,
                objective=objective,
                device=device,
                output_dir=run_dir,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                frontend_lr_factor=args.frontend_lr_factor,
                patience=args.patience,
                minimum_delta=args.minimum_delta,
                grad_clip_norm=args.grad_clip_norm,
                early_stopping_metric=args.early_stopping_metric,
                counterfactual_weight=args.counterfactual_weight,
                equivariance_weight=args.equivariance_weight,
                checkpoint_metadata={
                    "model_name": model_name,
                    "training_regime": objective_name,
                    "architecture": asdict(architecture),
                    "num_classes": num_classes,
                    "dataset_mel_rates": dataset_rates,
                    "parameter_count": parameter_info,
                },
            )
            plot_history(history, run_dir / "training_loss.png")
            predictions, embeddings, metrics = collect_recording_outputs(
                model, test_loader, device, validation_loss
            )
            per_dataset = per_dataset_metrics(predictions)
            metrics["mean_dataset_macro_f1"] = float(per_dataset["macro_f1"].mean())
            metrics["min_dataset_macro_f1"] = float(per_dataset["macro_f1"].min())
            summary = {
                "run_name": run_name,
                "model": model_name,
                "training_regime": objective_name,
                **metrics,
                **parameter_info,
            }
            summary_rows.append(summary)
            per_dataset.insert(0, "training_regime", objective_name)
            per_dataset.insert(0, "model", model_name)
            dataset_rows.append(per_dataset)
            per_class = per_class_metrics(predictions, label_map)
            per_class.insert(0, "training_regime", objective_name)
            per_class.insert(0, "model", model_name)
            class_rows.append(per_class)
            pd.DataFrame([summary]).to_csv(run_dir / "test_metrics.csv", index=False)
            per_dataset.to_csv(run_dir / "test_per_dataset.csv", index=False)
            per_class.to_csv(run_dir / "test_per_class.csv", index=False)
            predictions.to_csv(run_dir / "test_predictions.csv", index=False)
            np.save(run_dir / "test_embeddings.npy", embeddings)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    metrics_frame = pd.DataFrame(summary_rows)
    parameters_frame = pd.DataFrame(parameter_rows)
    per_dataset_frame = pd.concat(dataset_rows, ignore_index=True)
    per_class_frame = pd.concat(class_rows, ignore_index=True)
    metrics_frame.to_csv(output_dir / "summary_22_models.csv", index=False)
    metrics_frame.to_csv(output_dir / "summary_test_metrics.csv", index=False)
    parameters_frame.to_csv(output_dir / "summary_parameter_counts.csv", index=False)
    per_dataset_frame.to_csv(output_dir / "summary_test_per_dataset.csv", index=False)
    per_class_frame.to_csv(output_dir / "summary_test_per_class.csv", index=False)
    make_contrasts(metrics_frame).to_csv(output_dir / "summary_architecture_contrasts.csv", index=False)
    make_objective_contrasts(metrics_frame).to_csv(output_dir / "summary_objective_contrasts.csv", index=False)
    plot_summary(metrics_frame, output_dir / "summary_22_models_macro_f1.png")

    expected = len(args.models) * len(args.objectives)
    if len(metrics_frame) != expected:
        raise RuntimeError(f"Expected {expected} summary rows, found {len(metrics_frame)}")
    print(f"\nFinished {len(metrics_frame)} runs. Results: {output_dir}")


if __name__ == "__main__":
    main()
