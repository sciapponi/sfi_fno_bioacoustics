from __future__ import annotations

import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 1.0,
        alpha: torch.Tensor | None = None,
        label_smoothing: float = 0.01,
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
        modulation = (1.0 - target_probability).pow(self.gamma)
        if self.alpha.numel():
            modulation = modulation * self.alpha[labels]
        focal = -modulation * target_log_probability
        if self.label_smoothing <= 0:
            return focal.mean()
        smooth = -log_probabilities.mean(dim=1)
        return ((1.0 - self.label_smoothing) * focal + self.label_smoothing * smooth).mean()


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.10) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        memory_embeddings: torch.Tensor | None = None,
        memory_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        anchors = torch.nn.functional.normalize(embeddings, dim=1)

        if memory_embeddings is not None and memory_embeddings.numel():
            memory = torch.nn.functional.normalize(memory_embeddings.detach(), dim=1)
            candidates = torch.cat([anchors, memory], dim=0)
            candidate_labels = torch.cat([labels, memory_labels.detach()], dim=0)
        else:
            candidates = anchors
            candidate_labels = labels

        similarities = anchors @ candidates.T / self.temperature
        similarities = similarities - similarities.max(dim=1, keepdim=True).values.detach()

        self_mask = torch.zeros(
            len(labels), len(candidate_labels), device=labels.device, dtype=torch.bool
        )
        self_mask[:, : len(labels)] = torch.eye(
            len(labels), device=labels.device, dtype=torch.bool
        )
        positive = labels[:, None].eq(candidate_labels[None, :]) & ~self_mask
        valid = positive.any(dim=1)
        if not valid.any():
            return anchors.sum() * 0.0

        denominator_mask = ~self_mask
        exp_values = torch.exp(similarities) * denominator_mask.to(similarities.dtype)
        log_probability = similarities - torch.log(
            exp_values.sum(dim=1, keepdim=True).clamp_min(1e-12)
        )
        mean_positive = (
            positive.to(log_probability.dtype) * log_probability
        ).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
        return -mean_positive[valid].mean()


def effective_number_weights(class_counts: torch.Tensor, beta: float = 0.999) -> torch.Tensor:
    counts = class_counts.float().clamp_min(1.0)
    effective = 1.0 - torch.pow(torch.tensor(float(beta)), counts)
    weights = (1.0 - float(beta)) / effective.clamp_min(1e-12)
    return weights / weights.mean().clamp_min(1e-12)


def build_focal_loss(
    class_counts: torch.Tensor,
    device: torch.device,
    gamma: float = 1.0,
    label_smoothing: float = 0.01,
    use_class_weights: bool = True,
) -> FocalLoss:
    alpha = effective_number_weights(class_counts).to(device) if use_class_weights else None
    return FocalLoss(gamma=gamma, alpha=alpha, label_smoothing=label_smoothing)
