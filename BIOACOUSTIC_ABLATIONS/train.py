from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses import SupervisedContrastiveLoss
from utils import atomic_torch_save


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result = dict(batch)
    for key in ["waveform", "lengths", "sample_rate", "original_sample_rate", "label", "recording_id", "crop_index"]:
        result[key] = result[key].to(device, non_blocking=True)
    return result


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
    }


@torch.no_grad()
def collect_recording_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, float]]:
    model.eval()
    records: dict[int, dict[str, Any]] = {}
    total_loss = 0.0
    total_crops = 0
    total_forward_seconds = 0.0

    for batch in loader:
        batch = move_batch(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        output = model(batch["waveform"], batch["lengths"], batch["sample_rate"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_forward_seconds += time.perf_counter() - start
        logits = output["logits"]
        total_loss += float(criterion(logits, batch["label"]).item()) * len(logits)
        total_crops += len(logits)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        embeddings = output["embedding"].cpu().numpy()

        for index in range(len(logits)):
            recording_id = int(batch["recording_id"][index])
            item = records.setdefault(
                recording_id,
                {
                    "probabilities": [],
                    "embeddings": [],
                    "label": int(batch["label"][index]),
                    "dataset": batch["dataset"][index],
                    "label_name": batch["label_name"][index],
                    "original_sample_rate": int(batch["original_sample_rate"][index]),
                },
            )
            item["probabilities"].append(probabilities[index])
            item["embeddings"].append(embeddings[index])

    rows, embedding_rows = [], []
    for recording_id, item in sorted(records.items()):
        probability = np.mean(np.stack(item["probabilities"]), axis=0)
        embedding = np.mean(np.stack(item["embeddings"]), axis=0)
        rows.append(
            {
                "recording_id": recording_id,
                "dataset": item["dataset"],
                "label_id": item["label"],
                "label_name": item["label_name"],
                "predicted_label_id": int(probability.argmax()),
                "confidence": float(probability.max()),
                "original_sample_rate": item["original_sample_rate"],
            }
        )
        embedding_rows.append(embedding)

    predictions = pd.DataFrame(rows)
    embedding_array = np.stack(embedding_rows)
    metrics = classification_metrics(
        predictions["label_id"].to_numpy(),
        predictions["predicted_label_id"].to_numpy(),
    )
    metrics.update(
        loss=total_loss / max(1, total_crops),
        num_recordings=len(predictions),
        num_crops=total_crops,
        model_ms_per_crop=1000.0 * total_forward_seconds / max(1, total_crops),
    )
    return predictions, embedding_array, metrics


def per_dataset_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in predictions.groupby("dataset", sort=True):
        y_true = group["label_id"].to_numpy(dtype=np.int64)
        y_pred = group["predicted_label_id"].to_numpy(dtype=np.int64)
        labels = np.unique(y_true)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        rows.append(
            {
                "dataset": dataset,
                "num_recordings": len(group),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(np.mean(recall)) if len(recall) else 0.0,
                "macro_f1": float(np.mean(f1)) if len(f1) else 0.0,
                "weighted_f1": float(np.average(f1, weights=support)) if int(np.sum(support)) > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def per_class_metrics(predictions: pd.DataFrame, label_map: pd.DataFrame) -> pd.DataFrame:
    labels = label_map["label_id"].to_numpy(dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        predictions["label_id"].to_numpy(),
        predictions["predicted_label_id"].to_numpy(),
        labels=labels,
        zero_division=0,
    )
    result = label_map.copy()
    result["precision"] = precision
    result["recall"] = recall
    result["f1"] = f1
    result["support"] = support
    return result


def plot_tsne_by_class(
    embeddings: np.ndarray,
    predictions: pd.DataFrame,
    output_path: Path,
    seed: int,
    maximum_points: int = 5000,
    perplexity: float = 30.0,
) -> pd.DataFrame:
    values = np.asarray(embeddings, dtype=np.float32)
    frame = predictions.reset_index(drop=True).copy()
    rng = np.random.default_rng(int(seed))
    if len(values) > int(maximum_points):
        selected = rng.choice(len(values), size=int(maximum_points), replace=False)
        selected.sort()
        values = values[selected]
        frame = frame.iloc[selected].reset_index(drop=True)
    use_perplexity = min(float(perplexity), max(2.0, (len(values) - 1) / 3.0))
    projection = TSNE(
        n_components=2,
        perplexity=use_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=int(seed),
        metric="cosine",
    ).fit_transform(values)
    labels = sorted(frame["label_id"].unique().tolist())
    colour_index = {label: index for index, label in enumerate(labels)}
    colours = np.asarray([colour_index[value] for value in frame["label_id"]])
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.scatter(projection[:, 0], projection[:, 1], c=colours, cmap="gist_ncar", s=12, alpha=0.75, linewidths=0)
    axis.set_title("Test embeddings by class (fold 1)")
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    lookup = frame[["label_id", "label_name"]].drop_duplicates().sort_values("label_id").reset_index(drop=True)
    lookup["colour_index"] = lookup["label_id"].map(colour_index)
    return lookup


def make_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    frontend_lr_factor: float,
) -> torch.optim.Optimizer:
    frontend_parameters = [p for p in model.frontend.parameters() if p.requires_grad]
    frontend_ids = {id(p) for p in frontend_parameters}
    main_parameters = [p for p in model.parameters() if p.requires_grad and id(p) not in frontend_ids]
    groups = [{"params": main_parameters, "lr": learning_rate, "weight_decay": weight_decay}]
    if frontend_parameters:
        groups.append({"params": frontend_parameters, "lr": learning_rate * frontend_lr_factor, "weight_decay": weight_decay})
    return torch.optim.AdamW(groups)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    focal_loss: nn.Module,
    contrastive_loss: SupervisedContrastiveLoss,
    contrastive_weight: float,
    device: torch.device,
    epoch: int,
    grad_clip_norm: float,
    gradient_accumulation_steps: int,
    contrastive_memory_size: int,
) -> dict[str, float]:
    model.train()
    accumulation_steps = max(1, int(gradient_accumulation_steps))
    memory_size = max(0, int(contrastive_memory_size))

    totals = {
        "loss": 0.0,
        "focal": 0.0,
        "contrastive": 0.0,
        "correct": 0.0,
        "examples": 0.0,
    }
    optimizer_steps = 0
    skipped_nonfinite_batches = 0
    skipped_nonfinite_steps = 0
    gradient_norm_sum = 0.0
    gradient_norm_max = 0.0

    memory_embeddings: torch.Tensor | None = None
    memory_labels: torch.Tensor | None = None
    accumulated_batches = 0
    optimizer.zero_grad(set_to_none=True)

    def enqueue(embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        nonlocal memory_embeddings, memory_labels
        if memory_size <= 0:
            return
        detached_embeddings = embeddings.detach()
        detached_labels = labels.detach()
        if memory_embeddings is None:
            memory_embeddings = detached_embeddings
            memory_labels = detached_labels
        else:
            memory_embeddings = torch.cat([memory_embeddings, detached_embeddings], dim=0)
            memory_labels = torch.cat([memory_labels, detached_labels], dim=0)
        if len(memory_labels) > memory_size:
            memory_embeddings = memory_embeddings[-memory_size:]
            memory_labels = memory_labels[-memory_size:]

    def finish_optimizer_step(valid_batches: int) -> bool:
        nonlocal optimizer_steps, skipped_nonfinite_steps, gradient_norm_sum, gradient_norm_max
        if valid_batches <= 0:
            return False

        scale = 1.0 / float(valid_batches)
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)

        clip_value = float(grad_clip_norm) if float(grad_clip_norm) > 0 else 1.0e12
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip_value, error_if_nonfinite=False
        )
        if not torch.isfinite(gradient_norm):
            skipped_nonfinite_steps += 1
            optimizer.zero_grad(set_to_none=True)
            return False

        gradient_norm_value = float(gradient_norm.item())
        gradient_norm_sum += gradient_norm_value
        gradient_norm_max = max(gradient_norm_max, gradient_norm_value)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        return True

    progress = tqdm(loader, desc=f"train {epoch:03d}", leave=False)
    for batch in progress:
        batch = move_batch(batch, device)
        output = model(batch["waveform"], batch["lengths"], batch["sample_rate"])

        logits = output["logits"]
        contrastive_embedding = output["contrastive_embedding"]
        if not torch.isfinite(logits).all() or not torch.isfinite(contrastive_embedding).all():
            skipped_nonfinite_batches += 1
            progress.set_postfix(skip="nonfinite-output")
            continue

        focal = focal_loss(logits, batch["label"])
        contrastive = contrastive_loss(
            contrastive_embedding,
            batch["label"],
            memory_embeddings=memory_embeddings,
            memory_labels=memory_labels,
        )
        loss = focal + float(contrastive_weight) * contrastive

        if not (torch.isfinite(focal) and torch.isfinite(contrastive) and torch.isfinite(loss)):
            skipped_nonfinite_batches += 1
            progress.set_postfix(skip="nonfinite-loss")
            continue

        loss.backward()
        accumulated_batches += 1
        enqueue(contrastive_embedding, batch["label"])

        count = len(batch["label"])
        totals["loss"] += float(loss.detach().item()) * count
        totals["focal"] += float(focal.detach().item()) * count
        totals["contrastive"] += float(contrastive.detach().item()) * count
        totals["correct"] += int((logits.argmax(1) == batch["label"]).sum())
        totals["examples"] += count

        if accumulated_batches >= accumulation_steps:
            finish_optimizer_step(accumulated_batches)
            accumulated_batches = 0

        if totals["examples"] > 0:
            progress.set_postfix(
                loss=f"{totals['loss'] / totals['examples']:.4f}",
                skipped=skipped_nonfinite_batches,
            )

    if accumulated_batches > 0:
        finish_optimizer_step(accumulated_batches)

    denominator = max(1.0, totals["examples"])
    return {
        "train_loss": totals["loss"] / denominator,
        "train_focal_loss": totals["focal"] / denominator,
        "train_contrastive_loss": totals["contrastive"] / denominator,
        "train_accuracy": totals["correct"] / denominator,
        "train_optimizer_steps": float(optimizer_steps),
        "train_skipped_nonfinite_batches": float(skipped_nonfinite_batches),
        "train_skipped_nonfinite_steps": float(skipped_nonfinite_steps),
        "train_gradient_norm_mean": gradient_norm_sum / max(1, optimizer_steps),
        "train_gradient_norm_max": gradient_norm_max,
    }

def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    focal_loss: nn.Module,
    validation_loss: nn.Module,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    frontend_lr_factor: float,
    contrastive_weight: float,
    contrastive_temperature: float,
    patience: int,
    minimum_delta: float,
    grad_clip_norm: float,
    gradient_accumulation_steps: int,
    contrastive_memory_size: int,
    checkpoint_metadata: dict[str, Any],
) -> pd.DataFrame:
    optimizer = make_optimizer(model, learning_rate, weight_decay, frontend_lr_factor)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(1, patience // 3), min_lr=1e-6
    )
    contrastive_loss = SupervisedContrastiveLoss(contrastive_temperature)
    history: list[dict[str, float]] = []
    best_score = -float("inf")
    stale = 0
    checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(1, int(epochs) + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, focal_loss, contrastive_loss,
            contrastive_weight, device, epoch, grad_clip_norm,
            gradient_accumulation_steps, contrastive_memory_size,
        )
        _, _, validation_metrics = collect_recording_outputs(
            model, validation_loader, device, validation_loss
        )
        score = float(validation_metrics["macro_f1"])
        scheduler.step(score)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **train_metrics,
            "val_loss": validation_metrics["loss"],
            "val_accuracy": validation_metrics["accuracy"],
            "val_balanced_accuracy": validation_metrics["balanced_accuracy"],
            "val_macro_f1": score,
            "val_weighted_f1": validation_metrics["weighted_f1"],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} | train {row['train_loss']:.4f} | "
            f"val macro-F1 {score:.4f}"
        )
        if score > best_score + float(minimum_delta):
            best_score = score
            stale = 0
            atomic_torch_save(
                {"model_state": model.state_dict(), "epoch": epoch, "best_validation_score": best_score, **checkpoint_metadata},
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= int(patience):
                break

    frame = pd.DataFrame(history)
    frame.to_csv(output_dir / "training_history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    plot_history(frame, output_dir / "training_history.png")
    return frame


def plot_history(history: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history["epoch"], history["train_loss"], label="train loss")
    axis.plot(history["epoch"], history["val_loss"], label="validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def aggregate_cross_validation(fold_results: pd.DataFrame) -> pd.DataFrame:
    if fold_results.empty:
        return pd.DataFrame()
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "model_ms_per_crop"]
    rows = []
    for model, group in fold_results.groupby("model", sort=False):
        row: dict[str, Any] = {"model": model, "folds_completed": int(group["fold"].nunique())}
        for metric in metrics:
            if metric in group:
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_accuracy_macro_f1(summary: pd.DataFrame, output_path: Path, title: str) -> None:
    if summary.empty:
        return
    x = np.arange(len(summary))
    width = 0.38
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(x - width / 2, summary["accuracy_mean"], width, yerr=summary["accuracy_std"], label="Accuracy")
    axis.bar(x + width / 2, summary["macro_f1_mean"], width, yerr=summary["macro_f1_std"], label="Macro F1")
    axis.set_xticks(x)
    axis.set_xticklabels(summary["model"], rotation=35, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_fold_trajectories(fold_results: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    for model, group in fold_results.groupby("model", sort=False):
        group = group.sort_values("fold")
        axis.plot(group["fold"], group["macro_f1"], marker="o", label=model)
    axis.set_xlabel("Fold")
    axis.set_ylabel("Macro F1")
    axis.set_xticks(sorted(fold_results["fold"].unique()))
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def aggregate_group_metrics(frame: pd.DataFrame, group_columns: list[str], metric_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(group_columns, observed=True, sort=True)
    rows = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)
