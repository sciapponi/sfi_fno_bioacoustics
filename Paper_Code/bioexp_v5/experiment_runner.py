from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import AudioSegmentDataset, collate_audio_batch
from .evaluation import (
    collect_recording_outputs,
    per_class_metrics,
    per_dataset_metrics,
    plot_tsne_by_class,
)
from .losses import ObjectiveConfig, build_classification_loss
from .manifest import (
    assign_stratified_cv_folds,
    build_manifest_from_dataset_paths,
    filter_minimum_class_support,
    make_fold_manifest,
    split_counts,
)
from .models import count_parameters
from .registry import (
    MODEL_SPECS,
    RUN_NAMES,
    RUN_SPECS,
    build_model,
    default_architecture_config,
    model_table,
)
from .reporting import (
    aggregate_cross_validation,
    plot_accuracy_macro_f1,
    plot_fold_trajectories,
    plot_history,
)
from .samplers import DatasetBalancedPKBatchSampler
from .trainer import fit_model
from .utils import (
    choose_device,
    ensure_dir,
    save_json,
    seed_everything,
    worker_init_fn,
)


DEFAULT_DATASET_PATHS = (
    "Watkins_Full_Cuts=/home/ardan/ARDAN/BIOACOUSTICS/Watkins_Full_Cuts",
    "Birds=/mnt/volDISI_conci_Datasets/audio/bioacoustics/birds2",
    "Frogs=/mnt/volDISI_conci_Datasets/audio/bioacoustics/frogs",
    "Bats=/mnt/volDISI_conci_Datasets/audio/bioacoustics/bats",
)


def parse_args(frontend_variant: str) -> argparse.Namespace:
    if frontend_variant not in {"multiresolution", "single_resolution"}:
        raise ValueError(f"Unknown frontend variant: {frontend_variant}")
    is_multiresolution = frontend_variant == "multiresolution"
    default_output = (
        Path("runs/bioacoustic_cv_v5_standard_multires")
        if is_multiresolution
        else Path("runs/bioacoustic_cv_v5_standard_single_64ms")
    )
    description = (
        "Run nine natural-resolution temporal-pyramid bioacoustic systems "
        "under five-fold cross-validation."
        if is_multiresolution
        else "Run the matching nine single-resolution 64 ms bioacoustic systems "
        "under five-fold cross-validation."
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset-paths",
        nargs="+",
        default=list(DEFAULT_DATASET_PATHS),
        metavar="NAME=/ABSOLUTE/PATH",
        help=(
            "One or more dataset directories. Each directory must directly contain "
            "class subdirectories. Use NAME=/absolute/path to set a stable dataset name."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--runs", nargs="+", choices=RUN_NAMES, default=RUN_NAMES)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Reuse completed fold/run directories when test_metrics.csv exists.",
    )

    parser.add_argument("--minimum-samples-per-class", type=int, default=15)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--estimate-bandwidths", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--bandwidth-scan-seconds", type=float, default=2.0)
    parser.add_argument("--bandwidth-energy-quantile", type=float, default=0.995)

    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--eval-crops", type=int, default=3)
    parser.add_argument(
        "--waveform-normalization", choices=["none", "peak", "rms"], default="peak"
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--train-samples-per-epoch", type=int, default=0)

    parser.add_argument(
        "--multiresolution-window-ms", nargs=3, type=float,
        default=[16.0, 64.0, 512.0],
        help="Fine, intermediate, and coarse windows for the multiresolution runner.",
    )
    parser.add_argument(
        "--single-window-ms", type=float, default=64.0,
        help="The only analysis window used by run_single_experiments.py.",
    )
    parser.add_argument(
        "--hop-ratio", type=float, default=0.5,
        help="Hop as a fraction of each realized window; 0.25 gives 75 percent overlap.",
    )
    parser.add_argument(
        "--max-window-samples", type=int, default=0,
        help="Optional safety cap. Zero preserves requested physical durations without a cap.",
    )
    parser.add_argument("--frequency-bins", type=int, default=128)
    parser.add_argument("--minimum-frequency-hz", type=float, default=5.0)
    parser.add_argument(
        "--maximum-frequency-hz", type=float, default=0.0,
        help="Physical frontend maximum. Zero uses the highest corpus Nyquist.",
    )
    parser.add_argument("--q-factor", type=float, default=12.0)
    parser.add_argument(
        "--modulation-rates-hz", nargs="+", type=float, default=[2.0, 4.0, 8.0, 16.0]
    )
    parser.add_argument("--nyquist-rolloff", type=float, default=0.92)
    parser.add_argument("--base-channels", type=int, default=12)
    parser.add_argument("--embedding-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--fno-modes-frequency", type=int, default=4)
    parser.add_argument("--fno-modes-time", type=int, default=4)
    parser.add_argument(
        "--mbconv-stage-channels", nargs="+", type=int, default=[24, 48, 96, 128]
    )
    parser.add_argument(
        "--mbconv-stage-depths", nargs="+", type=int, default=[2, 2, 2, 2]
    )
    parser.add_argument(
        "--mbconv-kernel-sizes", nargs="+", type=int, default=[3, 3, 5, 5]
    )
    parser.add_argument("--mbconv-expansion-ratio", type=float, default=4.0)
    parser.add_argument("--mbconv-se-ratio", type=float, default=0.25)
    parser.add_argument("--mbconv-stochastic-depth", type=float, default=0.10)
    parser.add_argument("--mbconv-head-channels", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--contrastive-projection-dim", type=int, default=128)

    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument(
        "--focal-alpha-mode", choices=["none", "effective_num"], default="effective_num"
    )
    parser.add_argument("--contrastive-weight", type=float, default=0.10)
    parser.add_argument("--contrastive-temperature", type=float, default=0.10)
    parser.add_argument("--label-smoothing", type=float, default=0.01)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--frontend-lr-factor", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--minimum-delta", type=float, default=1e-4)
    parser.add_argument(
        "--early-stopping-metric",
        choices=["global_macro_f1", "mean_dataset_macro_f1"],
        default="mean_dataset_macro_f1",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-database-rate", type=int, default=0,
        help="Override the common maximum-rate baselines. Zero uses the corpus maximum.",
    )
    parser.add_argument(
        "--save-fold1-tsne", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--tsne-max-points", type=int, default=5000)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    args = parser.parse_args()
    args.frontend_variant = frontend_variant
    return args

def parse_dataset_paths(
    values: list[str] | tuple[str, ...] | None,
) -> dict[str, Path]:
    resolved_values = list(values) if values else list(DEFAULT_DATASET_PATHS)
    result: dict[str, Path] = {}
    for value in resolved_values:
        if "=" in value:
            name, path_text = value.split("=", 1)
            name = name.strip()
        else:
            path_text = value
            name = Path(path_text).expanduser().name
        path = Path(path_text).expanduser()
        if not path.is_absolute():
            raise ValueError(
                f"Dataset paths must be absolute. Received: {path_text!r}"
            )
        path = path.resolve()
        if not name:
            raise ValueError(f"Dataset name is empty for path: {path}")
        if name in result:
            raise ValueError(f"Duplicate dataset name: {name}")
        result[name] = path
    return result


def select_run_specs(run_names: list[str]) -> list[dict[str, str]]:
    wanted = set(run_names)
    selected = [dict(item) for item in RUN_SPECS if item["run_name"] in wanted]
    if len(selected) != len(wanted):
        missing = wanted.difference(item["run_name"] for item in selected)
        raise KeyError(f"Unknown run names: {sorted(missing)}")
    return selected



def make_loader(
    dataset,
    batch_size: int,
    samples_per_class: int,
    training: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    train_samples_per_epoch: int,
) -> DataLoader:
    collate = collate_audio_batch
    generator = torch.Generator().manual_seed(int(seed))
    if training:
        sampler = DatasetBalancedPKBatchSampler(
            dataset=dataset,
            batch_size=int(batch_size),
            samples_per_class=int(samples_per_class),
            seed=int(seed),
            num_samples=int(train_samples_per_epoch) or None,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            collate_fn=collate,
            worker_init_fn=worker_init_fn if num_workers else None,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=collate,
        worker_init_fn=worker_init_fn if num_workers else None,
        generator=generator,
    )


def make_datasets(
    model_name: str,
    fold_manifest: pd.DataFrame,
    args: argparse.Namespace,
    corpus_max_rate: int,
):
    spec = MODEL_SPECS[model_name]
    forced_rate = None
    if spec["rate_mode"] == "fixed_48k":
        forced_rate = 48000
    elif spec["rate_mode"] == "corpus_max":
        forced_rate = int(corpus_max_rate)

    train = AudioSegmentDataset(
        fold_manifest,
        "train",
        args.segment_seconds,
        True,
        1,
        args.waveform_normalization,
        forced_output_rate=forced_rate,
    )
    validation = AudioSegmentDataset(
        fold_manifest,
        "val",
        args.segment_seconds,
        False,
        args.eval_crops,
        args.waveform_normalization,
        forced_output_rate=forced_rate,
    )
    test = AudioSegmentDataset(
        fold_manifest,
        "test",
        args.segment_seconds,
        False,
        args.eval_crops,
        args.waveform_normalization,
        forced_output_rate=forced_rate,
    )
    return train, validation, test

def architecture_config_from_args(
    args: argparse.Namespace, maximum_frequency_hz: float
) -> dict[str, Any]:
    config = default_architecture_config()
    if args.frontend_variant == "multiresolution":
        windows = tuple(float(value) for value in args.multiresolution_window_ms)
        experiment_variant = "multiresolution_temporal_pyramid"
    else:
        windows = (float(args.single_window_ms),)
        experiment_variant = f"single_resolution_{float(args.single_window_ms):g}ms"
    config.update(
        {
            "experiment_variant": experiment_variant,
            "analysis_windows_ms": windows,
            "hop_ratio": float(args.hop_ratio),
            "max_window_samples": int(args.max_window_samples),
            "frequency_bins": int(args.frequency_bins),
            "minimum_frequency_hz": float(args.minimum_frequency_hz),
            "maximum_frequency_hz": float(maximum_frequency_hz),
            "q_factor": float(args.q_factor),
            "modulation_rates_hz": tuple(args.modulation_rates_hz),
            "nyquist_rolloff": float(args.nyquist_rolloff),
            "base_channels": int(args.base_channels),
            "embedding_dim": int(args.embedding_dim),
            "depth": int(args.depth),
            "fno_modes_frequency": int(args.fno_modes_frequency),
            "fno_modes_time": int(args.fno_modes_time),
            "mbconv_stage_channels": tuple(args.mbconv_stage_channels),
            "mbconv_stage_depths": tuple(args.mbconv_stage_depths),
            "mbconv_kernel_sizes": tuple(args.mbconv_kernel_sizes),
            "mbconv_expansion_ratio": float(args.mbconv_expansion_ratio),
            "mbconv_se_ratio": float(args.mbconv_se_ratio),
            "mbconv_stochastic_depth": float(args.mbconv_stochastic_depth),
            "mbconv_head_channels": int(args.mbconv_head_channels),
            "dropout": float(args.dropout),
            "contrastive_projection_dim": int(args.contrastive_projection_dim),
        }
    )
    return config

def update_partial_reports(
    output_dir: Path,
    all_fold_rows: list[dict[str, Any]],
    fold_number: int,
) -> None:
    frame = pd.DataFrame(all_fold_rows)
    frame.to_csv(output_dir / "cv_fold_results_partial.csv", index=False)
    summary = aggregate_cross_validation(frame)
    summary.to_csv(output_dir / "cv_summary_partial.csv", index=False)
    if not summary.empty:
        plot_accuracy_macro_f1(
            summary,
            output_dir / "cv_accuracy_macro_f1_partial.png",
            f"Cross-validation summary after fold {fold_number}",
            error_suffix="std",
        )


def run_one_system(
    run_spec: dict[str, str],
    fold_number: int,
    fold_manifest: pd.DataFrame,
    label_map: pd.DataFrame,
    architecture: dict[str, Any],
    corpus_max_rate: int,
    args: argparse.Namespace,
    device: torch.device,
    pin_memory: bool,
    fold_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    run_name = run_spec["run_name"]
    model_name = run_spec["model"]
    objective_name = run_spec["training_regime"]
    run_dir = ensure_dir(fold_dir / run_name)
    metrics_path = run_dir / "test_metrics.csv"

    if args.resume and metrics_path.is_file():
        print(f"Skipping completed run: fold {fold_number} / {run_name}")
        summary = pd.read_csv(metrics_path).iloc[0].to_dict()
        per_dataset = pd.read_csv(run_dir / "test_per_dataset.csv")
        per_class = pd.read_csv(run_dir / "test_per_class.csv")
        if (
            fold_number == 1
            and args.save_fold1_tsne
            and not (run_dir / "test_tsne_by_class.png").is_file()
            and (run_dir / "test_embeddings.npy").is_file()
            and (run_dir / "test_predictions.csv").is_file()
        ):
            embeddings = np.load(run_dir / "test_embeddings.npy")
            predictions = pd.read_csv(run_dir / "test_predictions.csv")
            colours = plot_tsne_by_class(
                embeddings,
                predictions,
                run_dir / "test_tsne_by_class.png",
                seed=args.seed,
                maximum_points=args.tsne_max_points,
                perplexity=args.tsne_perplexity,
            )
            colours.to_csv(run_dir / "test_tsne_class_colours.csv", index=False)
        return summary, per_dataset, per_class

    train_dataset, validation_dataset, test_dataset = make_datasets(
        model_name,
        fold_manifest,
        args,
        corpus_max_rate,
    )
    run_seed = int(args.seed) + 1000 * int(fold_number)
    seed_everything(run_seed)
    model = build_model(
        model_name=model_name,
        num_classes=len(label_map),
        architecture_config=architecture,
        use_contrastive_head=objective_name == "focal_contrastive",
    ).to(device)
    with (run_dir / "model_architecture.txt").open("w", encoding="utf-8") as handle:
        handle.write(str(model))
        handle.write("\n")
    parameter_info = count_parameters(model)
    pd.DataFrame([{**run_spec, **parameter_info}]).to_csv(
        run_dir / "parameter_count.csv", index=False
    )

    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        args.samples_per_class,
        True,
        args.num_workers,
        pin_memory,
        run_seed,
        args.train_samples_per_epoch,
    )
    validation_loader = make_loader(
        validation_dataset,
        args.batch_size,
        args.samples_per_class,
        False,
        args.num_workers,
        pin_memory,
        run_seed + 1,
        0,
    )
    test_loader = make_loader(
        test_dataset,
        args.batch_size,
        args.samples_per_class,
        False,
        args.num_workers,
        pin_memory,
        run_seed + 2,
        0,
    )

    class_counts = (
        fold_manifest[fold_manifest["split"] == "train"]["label_id"]
        .value_counts()
        .reindex(range(len(label_map)), fill_value=0)
        .to_numpy(dtype=np.float32)
    )
    class_counts_tensor = torch.tensor(class_counts)
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
    validation_loss = nn.CrossEntropyLoss()

    print(f"\n=== fold {fold_number:02d} / {run_name} ===")
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
        checkpoint_metadata={
            "fold": fold_number,
            "run_name": run_name,
            "model_name": model_name,
            "training_regime": objective_name,
            "architecture": architecture,
            "num_classes": len(label_map),
            "corpus_max_rate": corpus_max_rate,
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
        "fold": int(fold_number),
        **run_spec,
        **metrics,
        **parameter_info,
    }
    per_dataset.insert(0, "fold", int(fold_number))
    per_dataset.insert(1, "run_name", run_name)
    per_dataset.insert(2, "model", model_name)
    per_dataset.insert(3, "training_regime", objective_name)
    per_class = per_class_metrics(predictions, label_map)
    per_class.insert(0, "fold", int(fold_number))
    per_class.insert(1, "run_name", run_name)
    per_class.insert(2, "model", model_name)
    per_class.insert(3, "training_regime", objective_name)

    pd.DataFrame([summary]).to_csv(metrics_path, index=False)
    per_dataset.to_csv(run_dir / "test_per_dataset.csv", index=False)
    per_class.to_csv(run_dir / "test_per_class.csv", index=False)
    predictions.to_csv(run_dir / "test_predictions.csv", index=False)
    np.save(run_dir / "test_embeddings.npy", embeddings)

    if fold_number == 1 and args.save_fold1_tsne:
        colours = plot_tsne_by_class(
            embeddings,
            predictions,
            run_dir / "test_tsne_by_class.png",
            seed=run_seed,
            maximum_points=args.tsne_max_points,
            perplexity=args.tsne_perplexity,
        )
        colours.to_csv(run_dir / "test_tsne_class_colours.csv", index=False)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, per_dataset, per_class


def main(frontend_variant: str) -> None:
    args = parse_args(frontend_variant)
    if args.batch_size % args.samples_per_class:
        raise ValueError("batch-size must be divisible by samples-per-class")
    if args.samples_per_class < 2:
        raise ValueError("samples-per-class must be at least 2 for contrastive training")
    if args.minimum_samples_per_class < args.num_folds:
        raise ValueError(
            "minimum-samples-per-class must be at least the number of CV folds"
        )

    dataset_paths = parse_dataset_paths(args.dataset_paths)
    selected_runs = select_run_specs(args.runs)
    device = choose_device(args.device)
    pin_memory = device.type == "cuda"
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and not args.resume and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite or --resume."
        )
    ensure_dir(output_dir)
    seed_everything(args.seed)

    manifest = build_manifest_from_dataset_paths(
        dataset_paths=dataset_paths,
        estimate_bandwidths=args.estimate_bandwidths,
        bandwidth_scan_seconds=args.bandwidth_scan_seconds,
        bandwidth_energy_quantile=args.bandwidth_energy_quantile,
    )
    manifest, retained_support, excluded_support = filter_minimum_class_support(
        manifest, args.minimum_samples_per_class
    )
    manifest = assign_stratified_cv_folds(manifest, args.num_folds, args.seed)
    manifest.to_csv(output_dir / "manifest_with_cv_folds.csv", index=False)
    retained_support.to_csv(output_dir / "retained_class_support.csv", index=False)
    excluded_support.to_csv(output_dir / "excluded_class_support.csv", index=False)

    label_map = (
        manifest[["label_id", "label_name", "dataset", "class_name"]]
        .drop_duplicates()
        .sort_values("label_id")
        .reset_index(drop=True)
    )
    label_map.to_csv(output_dir / "label_map.csv", index=False)

    corpus_max_rate = int(args.max_database_rate or manifest["sample_rate"].max())
    maximum_frequency = float(
        args.maximum_frequency_hz or manifest["sample_rate"].max() / 2.0
    )
    architecture = architecture_config_from_args(args, maximum_frequency)
    pd.DataFrame(
        model_table(
            architecture["experiment_variant"],
            tuple(architecture["analysis_windows_ms"]),
        )
    ).to_csv(output_dir / "run_registry.csv", index=False)
    run_signature = {
        "dataset_paths": {name: str(path) for name, path in dataset_paths.items()},
        "selected_runs": selected_runs,
        "architecture": architecture,
        "corpus_max_rate": corpus_max_rate,
        "minimum_samples_per_class": int(args.minimum_samples_per_class),
        "num_folds": int(args.num_folds),
        "validation_fraction": float(args.validation_fraction),
        "seed": int(args.seed),
        "segment_seconds": float(args.segment_seconds),
        "eval_crops": int(args.eval_crops),
    }
    config_path = output_dir / "config.json"
    if args.resume and config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            previous_config = json.load(handle)
        if previous_config.get("run_signature") != json.loads(json.dumps(run_signature)):
            raise ValueError(
                "The existing output directory was created with a different run "
                "configuration. Use a new --output-dir or pass --overwrite."
            )
    save_json(
        {
            "run_signature": run_signature,
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "num_recordings": len(manifest),
            "num_classes": len(label_map),
            "device": str(device),
        },
        config_path,
    )

    all_fold_rows: list[dict[str, Any]] = []
    all_dataset_rows: list[pd.DataFrame] = []
    all_class_rows: list[pd.DataFrame] = []

    for fold_number in range(1, int(args.num_folds) + 1):
        fold_dir = ensure_dir(output_dir / f"fold_{fold_number:02d}")
        fold_manifest = make_fold_manifest(
            manifest,
            test_fold=fold_number,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        fold_manifest.to_csv(fold_dir / "manifest.csv", index=False)
        split_counts(fold_manifest).to_csv(fold_dir / "split_counts.csv", index=False)

        fold_rows: list[dict[str, Any]] = []
        for run_spec in selected_runs:
            summary, per_dataset, per_class = run_one_system(
                run_spec=run_spec,
                fold_number=fold_number,
                fold_manifest=fold_manifest,
                label_map=label_map,
                architecture=architecture,
                corpus_max_rate=corpus_max_rate,
                args=args,
                device=device,
                pin_memory=pin_memory,
                fold_dir=fold_dir,
            )
            fold_rows.append(summary)
            all_fold_rows.append(summary)
            all_dataset_rows.append(per_dataset)
            all_class_rows.append(per_class)

            fold_frame = pd.DataFrame(fold_rows)
            fold_frame.to_csv(fold_dir / "fold_summary.csv", index=False)
            if fold_number == 1:
                fold_frame.to_csv(
                    output_dir / "intermediate_fold1_summary.csv", index=False
                )
                plot_accuracy_macro_f1(
                    fold_frame,
                    output_dir / "intermediate_fold1_accuracy_macro_f1.png",
                    "Preliminary results after fold 1",
                )
            update_partial_reports(output_dir, all_fold_rows, fold_number)

        (fold_dir / "FOLD_COMPLETE.txt").write_text(
            f"Fold {fold_number} completed with {len(fold_rows)} runs.\n",
            encoding="utf-8",
        )
        if fold_number == 1:
            (output_dir / "FOLD_1_INTERMEDIATE_RESULTS_READY.txt").write_text(
                "Fold 1 is complete. Inspect intermediate_fold1_summary.csv and "
                "intermediate_fold1_accuracy_macro_f1.png.\n",
                encoding="utf-8",
            )

    fold_results = pd.DataFrame(all_fold_rows)
    fold_results.to_csv(output_dir / "cv_fold_results.csv", index=False)
    cv_summary = aggregate_cross_validation(fold_results)
    cv_summary.to_csv(output_dir / "cv_summary.csv", index=False)
    if all_dataset_rows:
        per_dataset_frame = pd.concat(all_dataset_rows, ignore_index=True)
        per_dataset_frame.to_csv(output_dir / "cv_per_dataset_metrics.csv", index=False)
        dataset_summary = (
            per_dataset_frame.groupby(
                ["run_name", "model", "training_regime", "dataset"], sort=False
            )["macro_f1"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        dataset_summary.to_csv(output_dir / "cv_per_dataset_summary.csv", index=False)
    if all_class_rows:
        per_class_frame = pd.concat(all_class_rows, ignore_index=True)
        per_class_frame.to_csv(output_dir / "cv_per_class_metrics.csv", index=False)
        class_summary = (
            per_class_frame.groupby(
                [
                    "run_name", "model", "training_regime", "label_id",
                    "label_name", "dataset", "class_name",
                ],
                sort=False,
            )[["precision", "recall", "f1", "support"]]
            .agg(["mean", "std"])
        )
        class_summary.columns = ["_".join(column) for column in class_summary.columns]
        class_summary.reset_index().to_csv(
            output_dir / "cv_per_class_summary.csv", index=False
        )

    plot_accuracy_macro_f1(
        cv_summary,
        output_dir / "cv_accuracy_macro_f1_mean_std.png",
        f"{args.num_folds}-fold cross-validation",
        error_suffix="std",
    )
    plot_fold_trajectories(
        fold_results, output_dir / "cv_macro_f1_by_fold.png"
    )
    expected_rows = int(args.num_folds) * len(selected_runs)
    if len(fold_results) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} fold/run rows, found {len(fold_results)}"
        )
    print(f"\nFinished {expected_rows} runs. Results: {output_dir}")


