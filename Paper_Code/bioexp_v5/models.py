from __future__ import annotations

import torch
from torch import nn

from .backends import GridEncoderBase
from .frontends import FeatureFrontend


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


def frontend_metadata(front: dict) -> dict[str, torch.Tensor]:
    return {
        "centres_hz": front["centres_hz"],
        "bandwidths_hz": front["bandwidths_hz"],
        "valid_mask": front["valid_mask"],
        "scale_frame_counts": torch.stack(
            [scale["frame_counts"] for scale in front["scales"]], dim=1
        ),
        "scale_realized_window_ms": torch.stack(
            [scale["realized_window_ms"] for scale in front["scales"]], dim=1
        ),
        "scale_realized_hop_ms": torch.stack(
            [scale["realized_hop_ms"] for scale in front["scales"]], dim=1
        ),
    }


class ScaleFusionClassifier(nn.Module):
    """Frontend plus one natural-resolution temporal-pyramid encoder."""

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
    ) -> dict[str, torch.Tensor]:
        front = self.frontend(waveforms, lengths, sample_rates, normalize=True)
        scale_features = [scale["features"] for scale in front["scales"]]
        time_masks = [scale["time_mask"] for scale in front["scales"]]
        embedding, weights = self.encoder(
            scale_features, front["valid_mask"], time_masks
        )
        logits = self.classifier(embedding)
        result: dict[str, torch.Tensor] = {
            "logits": logits,
            "embedding": embedding,
            "row_weights": weights,
            **frontend_metadata(front),
        }
        if self.contrastive_head is not None:
            result["contrastive_embedding"] = self.contrastive_head(embedding)
        if return_features:
            for index, scale in enumerate(front["scales"]):
                result[f"features_scale_{index}"] = scale["features"]
                result[f"raw_features_scale_{index}"] = scale["raw_features"]
                result[f"time_mask_scale_{index}"] = scale["time_mask"]
        return result


def count_parameters(module: nn.Module) -> dict[str, int | float]:
    def count(child: nn.Module | None) -> int:
        if child is None:
            return 0
        return sum(parameter.numel() for parameter in child.parameters())

    total = count(module)
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    frontend = count(getattr(module, "frontend", None))
    encoder = count(getattr(module, "encoder", None))
    classifier = count(getattr(module, "classifier", None))
    contrastive = count(getattr(module, "contrastive_head", None))
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frontend_parameters": frontend,
        "encoder_parameters": encoder,
        "classifier_parameters": classifier,
        "contrastive_head_parameters": contrastive,
        "model_size_fp32_mb": total * 4.0 / 1024.0**2,
    }
