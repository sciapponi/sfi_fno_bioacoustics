from __future__ import annotations

import math

import torch
from torch import nn


def group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(int(maximum), int(channels)), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock2d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: tuple[int, int] = (1, 1),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                input_channels, output_channels, 3,
                stride=stride, padding=1, bias=False,
            ),
            nn.GroupNorm(group_count(output_channels), output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(output_channels), output_channels),
        )
        self.skip = nn.Identity()
        if input_channels != output_channels or stride != (1, 1):
            self.skip = nn.Conv2d(
                input_channels, output_channels, 1, stride=stride, bias=False
            )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(torch.nn.functional.gelu(self.main(values) + self.skip(values)))


class SpectralConv2d(nn.Module):
    """Low-mode Fourier convolution used by the FNO backend."""
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        modes_frequency: int,
        modes_time: int,
    ) -> None:
        super().__init__()
        self.output_channels = int(output_channels)
        self.modes_frequency = int(modes_frequency)
        self.modes_time = int(modes_time)
        scale = 1.0 / math.sqrt(max(1, input_channels * output_channels))
        shape = (
            input_channels,
            output_channels,
            self.modes_frequency,
            self.modes_time,
        )
        self.real_weight = nn.Parameter(scale * torch.randn(*shape))
        self.imag_weight = nn.Parameter(scale * torch.randn(*shape))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, _, frequency, time = values.shape
        spectrum = torch.fft.rfft2(values, norm="ortho")
        output = spectrum.new_zeros(
            batch, self.output_channels, frequency, time // 2 + 1
        )
        use_frequency = min(self.modes_frequency, frequency)
        use_time = min(self.modes_time, time // 2 + 1)
        weight = torch.complex(
            self.real_weight[:, :, :use_frequency, :use_time],
            self.imag_weight[:, :, :use_frequency, :use_time],
        )
        output[:, :, :use_frequency, :use_time] = torch.einsum(
            "bcft,coft->boft",
            spectrum[:, :, :use_frequency, :use_time],
            weight,
        )
        return torch.fft.irfft2(output, s=(frequency, time), norm="ortho")


class FNOBlock2d(nn.Module):
    def __init__(
        self,
        channels: int,
        modes_frequency: int,
        modes_time: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(
            channels, channels, modes_frequency, modes_time
        )
        self.local = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(group_count(channels), channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.spectral(values) + self.local(values)
        update = self.dropout(torch.nn.functional.gelu(self.norm(update)))
        return values + update


class GridEncoderBase(nn.Module):
    output_dim: int

    def forward(
        self, features: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def prepare_grid(
        features: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        present = valid_mask.any(dim=1)
        safe_mask = valid_mask.clone()
        safe_features = features
        if (~present).any():
            safe_mask[~present, 0] = True
            safe_features = features.clone()
            safe_features[~present, 0] = 0.0
        mask = safe_mask[:, None, :, None].to(safe_features.dtype)
        inputs = safe_features.permute(0, 2, 1, 3).contiguous() * mask
        return inputs, safe_mask, present

    @staticmethod
    def row_weights(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        energy = features.abs().mean(dim=(2, 3)).masked_fill(~mask, -torch.inf)
        return torch.softmax(energy, dim=1)


class CNNGridEncoder(GridEncoderBase):
    """Local-CNN backend matched to the FNO lift, tail, pooling, and projection."""
    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        embedding_dim: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.output_dim = int(embedding_dim)
        channels = int(base_channels)
        self.lift = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(channels), channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock2d(channels, channels, dropout=dropout) for _ in range(depth)]
        )
        self.tail = nn.Sequential(
            ResidualBlock2d(channels, channels * 2, (2, 2), dropout),
            ResidualBlock2d(channels * 2, channels * 4, (2, 2), dropout),
        )
        self.projection = nn.Sequential(
            nn.Linear(channels * 8, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(
        self, features: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, safe_mask, present = self.prepare_grid(features, valid_mask)
        mask = safe_mask[:, None, :, None].to(inputs.dtype)
        hidden = self.lift(inputs)
        for block in self.blocks:
            hidden = block(hidden) * mask
        hidden = self.tail(hidden)
        pooled = torch.cat(
            [hidden.mean(dim=(-2, -1)), hidden.amax(dim=(-2, -1))], dim=1
        )
        embedding = self.projection(pooled)
        embedding = embedding * present.to(embedding.dtype).unsqueeze(1)
        weights = self.row_weights(features, safe_mask)
        weights = weights * present.to(weights.dtype).unsqueeze(1)
        return embedding, weights


class FNOGridEncoder(GridEncoderBase):
    """FNO backend with the same non-spectral components as CNNGridEncoder."""
    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        embedding_dim: int,
        depth: int,
        modes_frequency: int,
        modes_time: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.output_dim = int(embedding_dim)
        channels = int(base_channels)
        self.lift = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(channels), channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                FNOBlock2d(
                    channels, modes_frequency, modes_time, dropout
                )
                for _ in range(depth)
            ]
        )
        self.tail = nn.Sequential(
            ResidualBlock2d(channels, channels * 2, (2, 2), dropout),
            ResidualBlock2d(channels * 2, channels * 4, (2, 2), dropout),
        )
        self.projection = nn.Sequential(
            nn.Linear(channels * 8, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(
        self, features: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, safe_mask, present = self.prepare_grid(features, valid_mask)
        mask = safe_mask[:, None, :, None].to(inputs.dtype)
        hidden = self.lift(inputs)
        for block in self.blocks:
            hidden = block(hidden) * mask
        hidden = self.tail(hidden)
        pooled = torch.cat(
            [hidden.mean(dim=(-2, -1)), hidden.amax(dim=(-2, -1))], dim=1
        )
        embedding = self.projection(pooled)
        embedding = embedding * present.to(embedding.dtype).unsqueeze(1)
        weights = self.row_weights(features, safe_mask)
        weights = weights * present.to(weights.dtype).unsqueeze(1)
        return embedding, weights


class TokenSetEncoder(GridEncoderBase):
    """Per-frequency temporal encoder followed by a coordinate-aware set mixer.

    This is used only for the `dataset_mel` reference. It is kept as an explicit
    interchangeable backend rather than being hidden inside a special model.
    """

    def __init__(
        self,
        input_channels: int,
        token_dim: int,
        embedding_dim: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if token_dim % heads:
            raise ValueError("token_dim must be divisible by heads")
        self.output_dim = int(embedding_dim)
        self.token_dim = int(token_dim)
        self.temporal = nn.Sequential(
            nn.Conv1d(input_channels, token_dim, 5, padding=2, bias=False),
            nn.GroupNorm(group_count(token_dim), token_dim),
            nn.GELU(),
            nn.Conv1d(
                token_dim, token_dim, 5, padding=2,
                groups=group_count(token_dim), bias=False,
            ),
            nn.GroupNorm(group_count(token_dim), token_dim),
            nn.GELU(),
        )
        self.frequency_position = nn.Sequential(
            nn.Linear(3, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=3 * token_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mixer = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.score = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, 1)
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        centres_hz: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frequency, channels, time = features.shape
        present = valid_mask.any(dim=1)
        safe_mask = valid_mask.clone()
        safe_features = features
        if (~present).any():
            safe_mask[~present, 0] = True
            safe_features = features.clone()
            safe_features[~present, 0] = 0.0
        temporal = self.temporal(
            safe_features.reshape(batch * frequency, channels, time)
        )
        tokens = temporal.mean(dim=-1).reshape(batch, frequency, self.token_dim)
        if centres_hz is None:
            position = torch.linspace(
                0.0, 1.0, frequency, device=features.device, dtype=features.dtype
            )[None].expand(batch, -1)
        else:
            log_frequency = torch.log(centres_hz.clamp_min(1.0))
            minimum = log_frequency.amin(dim=1, keepdim=True)
            maximum = log_frequency.amax(dim=1, keepdim=True)
            position = (log_frequency - minimum) / (maximum - minimum).clamp_min(1e-6)
        position_features = torch.stack(
            [position, torch.sin(math.pi * position), torch.cos(math.pi * position)],
            dim=-1,
        )
        tokens = tokens + self.frequency_position(position_features)
        tokens = tokens.masked_fill(~safe_mask.unsqueeze(-1), 0.0)
        mixed = self.mixer(tokens, src_key_padding_mask=~safe_mask)
        scores = self.score(mixed).squeeze(-1).masked_fill(~safe_mask, -torch.inf)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(mixed * weights.unsqueeze(-1), dim=1)
        embedding = self.projection(pooled)
        embedding = embedding * present.to(embedding.dtype).unsqueeze(1)
        weights = weights * present.to(weights.dtype).unsqueeze(1)
        return embedding, weights
