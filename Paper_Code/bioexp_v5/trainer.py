from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .evaluation import collect_recording_outputs, move_batch, per_dataset_metrics
from .losses import ObjectiveConfig, SupervisedContrastiveLoss
from .utils import atomic_torch_save


def make_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    frontend_lr_factor: float,
) -> torch.optim.Optimizer:
    frontend_parameters = [
        parameter for parameter in model.frontend.parameters() if parameter.requires_grad
    ]
    frontend_ids = {id(parameter) for parameter in frontend_parameters}
    main_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in frontend_ids
    ]
    groups = [
        {
            "params": main_parameters,
            "lr": learning_rate,
            "weight_decay": weight_decay,
        }
    ]
    if frontend_parameters:
        groups.append(
            {
                "params": frontend_parameters,
                "lr": learning_rate * frontend_lr_factor,
                "weight_decay": weight_decay,
            }
        )
    return torch.optim.AdamW(groups)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    classification_loss: nn.Module,
    objective: ObjectiveConfig,
    device: torch.device,
    epoch: int,
    grad_clip_norm: float,
) -> dict[str, float]:
    model.train()
    contrastive_loss_function = (
        SupervisedContrastiveLoss(objective.contrastive_temperature)
        if objective.name == "focal_contrastive"
        else None
    )
    totals = {
        "loss": 0.0,
        "classification": 0.0,
        "contrastive": 0.0,
        "correct": 0.0,
        "examples": 0.0,
    }
    progress = tqdm(loader, desc=f"train {epoch:03d}", leave=False)

    for batch in progress:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["waveform"], batch["lengths"], batch["sample_rate"])
        classification = classification_loss(output["logits"], batch["label"])
        contrastive = classification * 0.0
        if contrastive_loss_function is not None:
            contrastive = contrastive_loss_function(
                output["contrastive_embedding"], batch["label"]
            )

        loss = classification
        if contrastive_loss_function is not None:
            loss = loss + objective.contrastive_weight * contrastive

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss: {float(loss)}")
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        count = len(batch["label"])
        totals["loss"] += float(loss.item()) * count
        totals["classification"] += float(classification.item()) * count
        totals["contrastive"] += float(contrastive.item()) * count
        totals["correct"] += int(
            (output["logits"].argmax(1) == batch["label"]).sum()
        )
        totals["examples"] += count
        progress.set_postfix(
            loss=f"{totals['loss']/max(1, totals['examples']):.4f}",
            acc=f"{totals['correct']/max(1, totals['examples']):.4f}",
        )

    denominator = max(1.0, totals["examples"])
    return {
        "train_loss": totals["loss"] / denominator,
        "train_classification_loss": totals["classification"] / denominator,
        "train_contrastive_loss": totals["contrastive"] / denominator,
        "train_accuracy": totals["correct"] / denominator,
    }


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    classification_loss: nn.Module,
    validation_loss: nn.Module,
    objective: ObjectiveConfig,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    frontend_lr_factor: float,
    patience: int,
    minimum_delta: float,
    grad_clip_norm: float,
    early_stopping_metric: str,
    checkpoint_metadata: dict[str, Any],
) -> pd.DataFrame:
    optimizer = make_optimizer(
        model, learning_rate, weight_decay, frontend_lr_factor
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(1, patience // 3),
        min_lr=1e-6,
    )
    history = []
    best_score = -float("inf")
    stale = 0
    checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            classification_loss=classification_loss,
            objective=objective,
            device=device,
            epoch=epoch,
            grad_clip_norm=grad_clip_norm,
        )
        predictions, _, validation_metrics = collect_recording_outputs(
            model, validation_loader, device, validation_loss
        )
        per_dataset = per_dataset_metrics(predictions)
        mean_dataset = float(per_dataset["macro_f1"].mean())
        minimum_dataset = float(per_dataset["macro_f1"].min())
        score = (
            float(validation_metrics["macro_f1"])
            if early_stopping_metric == "global_macro_f1"
            else mean_dataset
        )
        scheduler.step(score)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **train_metrics,
            "val_loss": validation_metrics["loss"],
            "val_accuracy": validation_metrics["accuracy"],
            "val_balanced_accuracy": validation_metrics["balanced_accuracy"],
            "val_macro_f1": validation_metrics["macro_f1"],
            "val_mean_dataset_macro_f1": mean_dataset,
            "val_min_dataset_macro_f1": minimum_dataset,
            "early_stopping_score": score,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} | train {row['train_loss']:.4f} | "
            f"val macro-F1 {row['val_macro_f1']:.4f} | "
            f"mean dataset {mean_dataset:.4f}"
        )
        if score > best_score + minimum_delta:
            best_score = score
            stale = 0
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_validation_score": best_score,
                    **checkpoint_metadata,
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= patience:
                break

    frame = pd.DataFrame(history)
    frame.to_csv(output_dir / "training_history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return frame
