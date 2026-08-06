from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class FocalLoss(nn.Module):
    """Multiclass focal loss with optional per-class alpha weights."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        if alpha is None:
            self.register_buffer("alpha", torch.empty(0), persistent=False)
        else:
            self.register_buffer("alpha", alpha.float(), persistent=True)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probabilities = torch.nn.functional.log_softmax(logits, dim=1)
        probabilities = log_probabilities.exp()
        target_log_probability = log_probabilities.gather(1, labels[:, None]).squeeze(1)
        target_probability = probabilities.gather(1, labels[:, None]).squeeze(1)
        modulation = (1.0 - target_probability).clamp_min(0.0).pow(self.gamma)
        if self.alpha.numel():
            modulation = modulation * self.alpha[labels]
        focal = -modulation * target_log_probability
        if self.label_smoothing <= 0:
            return focal.mean()
        smooth = -log_probabilities.mean(dim=1)
        return (
            (1.0 - self.label_smoothing) * focal
            + self.label_smoothing * smooth
        ).mean()


class SupervisedContrastiveLoss(nn.Module):
    """Same-class supervised contrastive loss for P x K batches."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        similarities = embeddings @ embeddings.T / self.temperature
        similarities = similarities - similarities.max(dim=1, keepdim=True).values.detach()
        count = len(labels)
        identity = torch.eye(count, device=labels.device, dtype=torch.bool)
        positive = labels[:, None].eq(labels[None, :]) & ~identity
        valid = positive.any(dim=1)
        if not valid.any():
            return embeddings.sum() * 0.0
        exp_values = torch.exp(similarities) * (~identity).to(similarities.dtype)
        log_probability = similarities - torch.log(
            exp_values.sum(dim=1, keepdim=True).clamp_min(1e-12)
        )
        mean_positive = (
            positive.to(log_probability.dtype) * log_probability
        ).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
        return -mean_positive[valid].mean()


@dataclass(frozen=True)
class ObjectiveConfig:
    name: str
    focal_gamma: float = 2.0
    contrastive_weight: float = 0.10
    contrastive_temperature: float = 0.10
    label_smoothing: float = 0.0
    focal_alpha_mode: str = "none"


OBJECTIVE_NAMES = ["cross_entropy", "focal_contrastive"]


def effective_number_weights(
    class_counts: torch.Tensor,
    beta: float = 0.999,
) -> torch.Tensor:
    counts = class_counts.float().clamp_min(1.0)
    effective = 1.0 - torch.pow(torch.tensor(float(beta)), counts)
    weights = (1.0 - float(beta)) / effective.clamp_min(1e-12)
    return weights / weights.mean().clamp_min(1e-12)


def build_classification_loss(
    objective: ObjectiveConfig,
    class_counts: torch.Tensor,
    device: torch.device,
) -> nn.Module:
    if objective.name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=objective.label_smoothing)
    if objective.name == "focal_contrastive":
        alpha = (
            effective_number_weights(class_counts).to(device)
            if objective.focal_alpha_mode == "effective_num"
            else None
        )
        return FocalLoss(
            gamma=objective.focal_gamma,
            alpha=alpha,
            label_smoothing=objective.label_smoothing,
        )
    raise ValueError(f"Unknown objective: {objective.name}")
