from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import nn

from .backends import GridEncoderBase, TokenSetEncoder
from .frontends import FeatureFrontend


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    embedding: torch.Tensor
    contrastive_embedding: torch.Tensor | None
    row_weights: torch.Tensor
    extras: dict[str, torch.Tensor]

    def as_dict(self) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {
            "logits": self.logits,
            "embedding": self.embedding,
            "row_weights": self.row_weights,
            **self.extras,
        }
        if self.contrastive_embedding is not None:
            result["contrastive_embedding"] = self.contrastive_embedding
        return result


class ContrastiveProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.network(embedding), dim=1)


class GridClassifier(nn.Module):
    """Composable frontend + encoder + linear classifier.

    Any FeatureFrontend can be paired with any compatible grid encoder. This is
    the class used by all non-equivariant ablation models.
    """

    def __init__(
        self,
        frontend: FeatureFrontend,
        encoder: GridEncoderBase,
        num_classes: int,
        contrastive_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.frontend = frontend
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.output_dim, int(num_classes))
        self.contrastive_head = (
            ContrastiveProjection(encoder.output_dim, int(contrastive_dim))
            if contrastive_dim
            else None
        )
        self.embedding_dim = encoder.output_dim
        self.num_classes = int(num_classes)

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        sample_rates: torch.Tensor,
        return_features: bool = False,
        max_band: int | None = None,
    ) -> dict[str, torch.Tensor]:
        del max_band
        front = self.frontend(waveforms, lengths, sample_rates, normalize=True)
        if isinstance(self.encoder, TokenSetEncoder):
            embedding, weights = self.encoder(
                front["features"], front["valid_mask"], front["centres_hz"]
            )
        else:
            embedding, weights = self.encoder(front["features"], front["valid_mask"])
        logits = self.classifier(embedding)
        contrastive = (
            self.contrastive_head(embedding)
            if self.contrastive_head is not None
            else None
        )
        extras = {
            "centres_hz": front["centres_hz"],
            "bandwidths_hz": front["bandwidths_hz"],
            "valid_mask": front["valid_mask"],
        }
        if return_features:
            extras["features"] = front["features"]
            extras["raw_features"] = front["raw_features"]
        return ModelOutput(
            logits, embedding, contrastive, weights, extras
        ).as_dict()


class NestedBandwidthClassifier(nn.Module):
    """Additive nested-band classifier with two or three physical bands.

    `boundaries_hz=(20000,)` creates two bands:
      band 0: f <= 20 kHz
      band 1: f > 20 kHz

    `boundaries_hz=(8000, 20000)` creates three bands:
      band 0: f <= 8 kHz
      band 1: 8 < f <= 20 kHz
      band 2: f > 20 kHz

    Each band has an independent encoder. Band 0 supplies the base prediction;
    higher bands add gated residual logits and residual embedding corrections.
    """

    def __init__(
        self,
        frontend: FeatureFrontend,
        encoder_factory: Callable[[], GridEncoderBase],
        num_classes: int,
        boundaries_hz: Sequence[float],
        residual_gate_bias: float = -1.0,
        contrastive_dim: int | None = None,
    ) -> None:
        super().__init__()
        boundaries = tuple(float(value) for value in boundaries_hz)
        if not boundaries or tuple(sorted(boundaries)) != boundaries:
            raise ValueError("boundaries_hz must be a non-empty increasing sequence")
        self.frontend = frontend
        self.boundaries_hz = boundaries
        self.num_bands = len(boundaries) + 1
        self.encoders = nn.ModuleList([encoder_factory() for _ in range(self.num_bands)])
        embedding_dim = self.encoders[0].output_dim
        if any(encoder.output_dim != embedding_dim for encoder in self.encoders):
            raise ValueError("All band encoders must have the same embedding dimension")
        self.classifiers = nn.ModuleList(
            [
                nn.Linear(embedding_dim, int(num_classes), bias=(index == 0))
                for index in range(self.num_bands)
            ]
        )
        self.adapters = nn.ModuleList(
            [
                nn.Identity()
                if index == 0
                else nn.Sequential(
                    nn.LayerNorm(embedding_dim),
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.Tanh(),
                )
                for index in range(self.num_bands)
            ]
        )
        self.gates = nn.ModuleList(
            [
                nn.Identity()
                if index == 0
                else nn.Sequential(
                    nn.Linear(embedding_dim + 1, max(16, embedding_dim // 2)),
                    nn.SiLU(),
                    nn.Linear(max(16, embedding_dim // 2), 1),
                )
                for index in range(self.num_bands)
            ]
        )
        for index in range(1, self.num_bands):
            nn.init.constant_(self.gates[index][-1].bias, float(residual_gate_bias))
        self.contrastive_head = (
            ContrastiveProjection(embedding_dim, int(contrastive_dim))
            if contrastive_dim
            else None
        )
        self.embedding_dim = embedding_dim
        self.num_classes = int(num_classes)

    def _band_masks(
        self, centres_hz: torch.Tensor, valid_mask: torch.Tensor
    ) -> list[torch.Tensor]:
        masks: list[torch.Tensor] = []
        lower = None
        for boundary in self.boundaries_hz:
            mask = centres_hz <= boundary
            if lower is not None:
                mask &= centres_hz > lower
            masks.append(valid_mask & mask)
            lower = boundary
        masks.append(valid_mask & (centres_hz > self.boundaries_hz[-1]))
        return masks

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        sample_rates: torch.Tensor,
        return_features: bool = False,
        max_band: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if max_band is None:
            max_band = self.num_bands - 1
        if not 0 <= int(max_band) < self.num_bands:
            raise ValueError(f"max_band must be between 0 and {self.num_bands - 1}")

        front = self.frontend(waveforms, lengths, sample_rates, normalize=False)
        masks = self._band_masks(front["centres_hz"], front["valid_mask"])
        batch = len(waveforms)
        cumulative_logits = torch.zeros(
            batch, self.num_classes, device=waveforms.device, dtype=waveforms.dtype
        )
        cumulative_embedding = torch.zeros(
            batch, self.embedding_dim, device=waveforms.device, dtype=waveforms.dtype
        )
        all_band_embeddings: list[torch.Tensor] = []
        all_band_logits: list[torch.Tensor] = []
        all_cumulative_embeddings: list[torch.Tensor] = []
        all_cumulative_logits: list[torch.Tensor] = []
        all_gates: list[torch.Tensor] = []
        all_weights: list[torch.Tensor] = []
        normalized_bands: list[torch.Tensor] = []

        for band_index, (mask, encoder, classifier) in enumerate(
            zip(masks, self.encoders, self.classifiers)
        ):
            band_features = self.frontend.normalize_active(front["raw_features"], mask)
            normalized_bands.append(band_features)
            embedding, weights = encoder(band_features, mask)
            logits = classifier(embedding)
            available = mask.any(dim=1).to(embedding.dtype)
            if band_index == 0:
                gate = available
                contribution = embedding
            else:
                coverage = mask.to(embedding.dtype).mean(dim=1, keepdim=True)
                gate = torch.sigmoid(
                    self.gates[band_index](torch.cat([embedding, coverage], dim=1))
                ).squeeze(1)
                gate = gate * available
                contribution = self.adapters[band_index](embedding) * gate.unsqueeze(1)
                logits = logits * gate.unsqueeze(1)
            cumulative_embedding = cumulative_embedding + contribution
            cumulative_logits = cumulative_logits + logits
            all_band_embeddings.append(embedding)
            all_band_logits.append(logits)
            all_cumulative_embeddings.append(cumulative_embedding)
            all_cumulative_logits.append(cumulative_logits)
            all_gates.append(gate)
            all_weights.append(weights)

        embedding = all_cumulative_embeddings[int(max_band)]
        logits = all_cumulative_logits[int(max_band)]
        contrastive = (
            self.contrastive_head(embedding)
            if self.contrastive_head is not None
            else None
        )
        combined_weights = torch.zeros_like(all_weights[0])
        for band_index in range(int(max_band) + 1):
            combined_weights += all_weights[band_index] * all_gates[band_index].unsqueeze(1)

        extras = {
            "centres_hz": front["centres_hz"],
            "bandwidths_hz": front["bandwidths_hz"],
            "valid_mask": front["valid_mask"],
            "band_embeddings": torch.stack(all_band_embeddings, dim=1),
            "band_logits": torch.stack(all_band_logits, dim=1),
            "cumulative_embeddings": torch.stack(all_cumulative_embeddings, dim=1),
            "cumulative_logits": torch.stack(all_cumulative_logits, dim=1),
            "band_gates": torch.stack(all_gates, dim=1),
            "band_masks": torch.stack(masks, dim=1),
        }
        if return_features:
            extras["features"] = torch.stack(normalized_bands, dim=1)
            extras["raw_features"] = front["raw_features"]
        return ModelOutput(
            logits, embedding, contrastive, combined_weights, extras
        ).as_dict()


def count_parameters(module: nn.Module) -> dict[str, int | float]:
    """Total and component-level parameter counts for one tested model."""

    def count(child: nn.Module | None) -> int:
        if child is None:
            return 0
        return sum(parameter.numel() for parameter in child.parameters())

    total = count(module)
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    frontend = count(getattr(module, "frontend", None))
    if hasattr(module, "encoder"):
        encoder = count(getattr(module, "encoder"))
    else:
        encoder = count(getattr(module, "encoders", None))
    classifier = count(getattr(module, "classifier", None)) + count(
        getattr(module, "classifiers", None)
    )
    fusion = count(getattr(module, "adapters", None)) + count(
        getattr(module, "gates", None)
    )
    contrastive = count(getattr(module, "contrastive_head", None))
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frontend_parameters": frontend,
        "encoder_parameters": encoder,
        "classifier_parameters": classifier,
        "band_fusion_parameters": fusion,
        "contrastive_head_parameters": contrastive,
        "model_size_fp32_mb": total * 4.0 / 1024.0**2,
    }
