from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
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
                input_channels,
                output_channels,
                3,
                stride=stride,
                padding=1,
                bias=False,
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
        return self.dropout(F.gelu(self.main(values) + self.skip(values)))


class StochasticDepth(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        probability = float(probability)
        if not 0.0 <= probability < 1.0:
            raise ValueError("stochastic-depth probability must be in [0, 1)")
        self.probability = probability

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return values
        survival = 1.0 - self.probability
        shape = (values.shape[0],) + (1,) * (values.ndim - 1)
        keep = torch.empty(shape, device=values.device, dtype=values.dtype).bernoulli_(survival)
        return values * keep / survival


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, squeeze_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, squeeze_channels, 1)
        self.expand = nn.Conv2d(squeeze_channels, channels, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        scale = self.pool(values)
        scale = F.silu(self.reduce(scale))
        scale = torch.sigmoid(self.expand(scale))
        return values * scale


class MBConvBlock(nn.Module):
    """Mobile inverted bottleneck with depthwise convolution and SE gating."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        expansion_ratio: float,
        kernel_size: int,
        stride: int,
        se_ratio: float,
        stochastic_depth: float,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size < 3:
            raise ValueError("MBConv kernel_size must be an odd integer >= 3")
        if stride not in (1, 2):
            raise ValueError("MBConv stride must be 1 or 2")
        expanded_channels = max(
            int(input_channels), int(round(input_channels * expansion_ratio))
        )
        layers: list[nn.Module] = []
        if expanded_channels != input_channels:
            layers.extend(
                [
                    nn.Conv2d(input_channels, expanded_channels, 1, bias=False),
                    nn.GroupNorm(group_count(expanded_channels), expanded_channels),
                    nn.SiLU(),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(
                    expanded_channels,
                    expanded_channels,
                    kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    groups=expanded_channels,
                    bias=False,
                ),
                nn.GroupNorm(group_count(expanded_channels), expanded_channels),
                nn.SiLU(),
                SqueezeExcitation(
                    expanded_channels,
                    max(8, int(round(input_channels * se_ratio))),
                ),
                nn.Conv2d(expanded_channels, output_channels, 1, bias=False),
                nn.GroupNorm(group_count(output_channels), output_channels),
            ]
        )
        self.main = nn.Sequential(*layers)
        self.use_residual = stride == 1 and input_channels == output_channels
        self.stochastic_depth = StochasticDepth(stochastic_depth)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.main(values)
        if self.use_residual:
            return values + self.stochastic_depth(update)
        return update


class SpectralConv2d(nn.Module):
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
        update = self.dropout(F.gelu(self.norm(update)))
        return values + update


def projection_block(input_channels: int, output_channels: int, activation: str) -> nn.Module:
    active: nn.Module = nn.SiLU() if activation == "silu" else nn.GELU()
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
        nn.GroupNorm(group_count(output_channels), output_channels),
        active,
    )


def transition_block(input_channels: int, output_channels: int, activation: str) -> nn.Module:
    active: nn.Module = nn.SiLU() if activation == "silu" else nn.GELU()
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 1, bias=False),
        nn.GroupNorm(group_count(output_channels), output_channels),
        active,
    )


class GridEncoderBase(nn.Module):
    output_dim: int

    def forward(
        self,
        scale_features: Sequence[torch.Tensor],
        valid_mask: torch.Tensor,
        time_masks: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class TemporalPyramidEncoderBase(GridEncoderBase):
    """One encoder hierarchy with progressive natural-resolution scale fusion.

    The finest grid is processed first. Learned hidden features are average-pooled
    to the exact natural frame count of the next frontend scale, concatenated with
    a lightweight projection of that scale, and fused with a 1x1 convolution.
    Raw frontend grids are never temporally interpolated or stacked as channels.
    """

    def __init__(
        self,
        input_channels: int,
        num_scales: int,
        stage_channels: Sequence[int],
        embedding_dim: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_scales = int(num_scales)
        self.stage_channels = tuple(int(value) for value in stage_channels)
        if self.num_scales not in (1, 3):
            raise ValueError("This experiment supports one or three frontend scales.")
        if len(self.stage_channels) < self.num_scales:
            raise ValueError("The encoder needs at least one stage per frontend scale.")
        self.output_dim = int(embedding_dim)
        self.dropout = float(dropout)
        self.activation_name = str(activation)

        self.input_projections = nn.ModuleList(
            [
                projection_block(
                    self.input_channels,
                    self.stage_channels[scale_index],
                    self.activation_name,
                )
                for scale_index in range(self.num_scales)
            ]
        )
        self.transitions = nn.ModuleList(
            [
                transition_block(
                    self.stage_channels[index],
                    self.stage_channels[index + 1],
                    self.activation_name,
                )
                for index in range(len(self.stage_channels) - 1)
            ]
        )
        self.fusions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        2 * self.stage_channels[index],
                        self.stage_channels[index],
                        1,
                        bias=False,
                    ),
                    nn.GroupNorm(
                        group_count(self.stage_channels[index]),
                        self.stage_channels[index],
                    ),
                    nn.SiLU() if self.activation_name == "silu" else nn.GELU(),
                )
                for index in range(1, self.num_scales)
            ]
        )
        self.stages = nn.ModuleList()
        self.stage_downsamples: list[bool] = []
        self.tail: nn.Module = nn.Identity()
        self.embedding_projection: nn.Module = nn.Identity()

    @staticmethod
    def safe_frequency_mask(
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        present = valid_mask.any(dim=1)
        safe_mask = valid_mask.clone()
        if (~present).any():
            safe_mask[~present, 0] = True
        return safe_mask, present

    @staticmethod
    def prepare_scale(
        features: torch.Tensor,
        safe_frequency_mask: torch.Tensor,
        time_mask: torch.Tensor,
        present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        safe_features = features
        if (~present).any():
            safe_features = features.clone()
            safe_features[~present] = 0.0
        spatial_mask = (
            safe_frequency_mask[:, None, :, None]
            & time_mask[:, None, None, :]
        ).to(features.dtype)
        inputs = safe_features.permute(0, 2, 1, 3).contiguous() * spatial_mask
        return inputs, spatial_mask

    @staticmethod
    def downsample_hidden_to_time(
        hidden: torch.Tensor,
        target_time: int,
    ) -> torch.Tensor:
        current_time = int(hidden.shape[-1])
        target_time = int(target_time)
        if target_time > current_time:
            raise ValueError(
                "Natural scale order is invalid: the next scale has more frames "
                f"({target_time}) than the current hidden map ({current_time})."
            )
        if target_time == current_time:
            return hidden
        return F.adaptive_avg_pool2d(hidden, (hidden.shape[-2], target_time))

    @staticmethod
    def halve_time(
        hidden: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.shape[-1] <= 1:
            return hidden, time_mask
        hidden = F.avg_pool2d(hidden, kernel_size=(1, 2), stride=(1, 2), ceil_mode=True)
        pooled_mask = F.max_pool1d(
            time_mask.to(hidden.dtype).unsqueeze(1),
            kernel_size=2,
            stride=2,
            ceil_mode=True,
        ).squeeze(1).bool()
        return hidden, pooled_mask

    @staticmethod
    def spatial_mask(
        frequency_mask: torch.Tensor,
        time_mask: torch.Tensor,
        frequency_size: int,
        time_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        base = (
            frequency_mask[:, None, :, None]
            & time_mask[:, None, None, :]
        ).to(dtype)
        if base.shape[-2:] != (frequency_size, time_size):
            base = F.interpolate(base, size=(frequency_size, time_size), mode="nearest")
        return base

    @staticmethod
    def masked_global_pool(
        hidden: torch.Tensor,
        spatial_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = spatial_mask.to(hidden.dtype)
        count = mask.sum(dim=(-2, -1)).clamp_min(1.0)
        mean = (hidden * mask).sum(dim=(-2, -1)) / count
        masked = hidden.masked_fill(mask == 0, -torch.inf)
        maximum = masked.amax(dim=(-2, -1))
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return torch.cat([mean, maximum], dim=1)

    @staticmethod
    def row_weights(
        scale_features: Sequence[torch.Tensor],
        safe_frequency_mask: torch.Tensor,
        time_masks: Sequence[torch.Tensor],
        present: torch.Tensor,
    ) -> torch.Tensor:
        energies: list[torch.Tensor] = []
        for features, time_mask in zip(scale_features, time_masks):
            active = time_mask[:, None, None, :].to(features.dtype)
            count = active.sum(dim=(2, 3)).clamp_min(1.0) * features.shape[2]
            energy = (features.abs() * active).sum(dim=(2, 3)) / count
            energies.append(energy)
        mean_energy = torch.stack(energies, dim=0).mean(dim=0)
        mean_energy = mean_energy.masked_fill(~safe_frequency_mask, -torch.inf)
        weights = torch.softmax(mean_energy, dim=1)
        return weights * present.to(weights.dtype).unsqueeze(1)

    def forward(
        self,
        scale_features: Sequence[torch.Tensor],
        valid_mask: torch.Tensor,
        time_masks: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(scale_features) != self.num_scales or len(time_masks) != self.num_scales:
            raise ValueError(
                f"Encoder expects {self.num_scales} scales, received "
                f"{len(scale_features)} features and {len(time_masks)} masks."
            )
        safe_mask, present = self.safe_frequency_mask(valid_mask)
        prepared: list[torch.Tensor] = []
        input_masks: list[torch.Tensor] = []
        for features, time_mask in zip(scale_features, time_masks):
            inputs, spatial = self.prepare_scale(features, safe_mask, time_mask, present)
            prepared.append(inputs)
            input_masks.append(spatial)

        hidden = self.input_projections[0](prepared[0]) * input_masks[0]
        current_time_mask = time_masks[0]
        hidden = self.stages[0](hidden)
        current_spatial_mask = self.spatial_mask(
            safe_mask,
            current_time_mask,
            hidden.shape[-2],
            hidden.shape[-1],
            hidden.dtype,
        )
        hidden = hidden * current_spatial_mask

        for stage_index in range(1, len(self.stages)):
            hidden = self.transitions[stage_index - 1](hidden)
            if stage_index < self.num_scales:
                target_time = int(prepared[stage_index].shape[-1])
                hidden = self.downsample_hidden_to_time(hidden, target_time)
                current_time_mask = time_masks[stage_index]
                injected = self.input_projections[stage_index](prepared[stage_index])
                hidden = self.fusions[stage_index - 1](
                    torch.cat([hidden, injected], dim=1)
                )
            elif not self.stage_downsamples[stage_index]:
                hidden, current_time_mask = self.halve_time(hidden, current_time_mask)

            current_spatial_mask = self.spatial_mask(
                safe_mask,
                current_time_mask,
                hidden.shape[-2],
                hidden.shape[-1],
                hidden.dtype,
            )
            hidden = hidden * current_spatial_mask
            hidden = self.stages[stage_index](hidden)
            current_spatial_mask = self.spatial_mask(
                safe_mask,
                current_time_mask,
                hidden.shape[-2],
                hidden.shape[-1],
                hidden.dtype,
            )
            hidden = hidden * current_spatial_mask

        hidden = self.tail(hidden)
        final_mask = self.spatial_mask(
            safe_mask,
            current_time_mask,
            hidden.shape[-2],
            hidden.shape[-1],
            hidden.dtype,
        )
        hidden = hidden * final_mask
        pooled = self.masked_global_pool(hidden, final_mask)
        embedding = self.embedding_projection(pooled)
        embedding = embedding * present.to(embedding.dtype).unsqueeze(1)
        weights = self.row_weights(scale_features, safe_mask, time_masks, present)
        return embedding, weights


class CNNTemporalPyramidEncoder(TemporalPyramidEncoderBase):
    def __init__(
        self,
        input_channels: int,
        num_scales: int,
        base_channels: int,
        embedding_dim: int,
        depth: int,
        dropout: float,
    ) -> None:
        stage_channels = (int(base_channels), int(base_channels) * 2, int(base_channels) * 4)
        super().__init__(
            input_channels=input_channels,
            num_scales=num_scales,
            stage_channels=stage_channels,
            embedding_dim=embedding_dim,
            activation="gelu",
            dropout=dropout,
        )
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        ResidualBlock2d(channels, channels, dropout=dropout)
                        for _ in range(int(depth))
                    ]
                )
                for channels in stage_channels
            ]
        )
        self.stage_downsamples = [False] * len(self.stages)
        last = stage_channels[-1]
        self.tail = nn.Sequential(
            ResidualBlock2d(last, last * 2, (2, 2), dropout),
            ResidualBlock2d(last * 2, last * 4, (2, 2), dropout),
        )
        self.embedding_projection = nn.Sequential(
            nn.Linear(last * 8, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )


class FNOTemporalPyramidEncoder(TemporalPyramidEncoderBase):
    def __init__(
        self,
        input_channels: int,
        num_scales: int,
        base_channels: int,
        embedding_dim: int,
        depth: int,
        modes_frequency: int,
        modes_time: int,
        dropout: float,
    ) -> None:
        stage_channels = (int(base_channels), int(base_channels) * 2, int(base_channels) * 4)
        super().__init__(
            input_channels=input_channels,
            num_scales=num_scales,
            stage_channels=stage_channels,
            embedding_dim=embedding_dim,
            activation="gelu",
            dropout=dropout,
        )
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        FNOBlock2d(
                            channels,
                            modes_frequency=int(modes_frequency),
                            modes_time=int(modes_time),
                            dropout=dropout,
                        )
                        for _ in range(int(depth))
                    ]
                )
                for channels in stage_channels
            ]
        )
        self.stage_downsamples = [False] * len(self.stages)
        last = stage_channels[-1]
        self.tail = nn.Sequential(
            ResidualBlock2d(last, last * 2, (2, 2), dropout),
            ResidualBlock2d(last * 2, last * 4, (2, 2), dropout),
        )
        self.embedding_projection = nn.Sequential(
            nn.Linear(last * 8, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )


class MBConvTemporalPyramidEncoder(TemporalPyramidEncoderBase):
    def __init__(
        self,
        input_channels: int,
        num_scales: int,
        embedding_dim: int,
        stage_channels: Sequence[int],
        stage_depths: Sequence[int],
        kernel_sizes: Sequence[int],
        expansion_ratio: float,
        se_ratio: float,
        stochastic_depth: float,
        head_channels: int,
        dropout: float,
    ) -> None:
        stage_channels = tuple(int(value) for value in stage_channels)
        stage_depths = tuple(int(value) for value in stage_depths)
        kernel_sizes = tuple(int(value) for value in kernel_sizes)
        if not (len(stage_channels) == len(stage_depths) == len(kernel_sizes)):
            raise ValueError(
                "MBConv stage_channels, stage_depths, and kernel_sizes must have equal lengths."
            )
        super().__init__(
            input_channels=input_channels,
            num_scales=num_scales,
            stage_channels=stage_channels,
            embedding_dim=embedding_dim,
            activation="silu",
            dropout=dropout,
        )
        total_blocks = sum(stage_depths)
        block_index = 0
        stages: list[nn.Module] = []
        downsample_flags: list[bool] = []
        for stage_index, (channels, repeats, kernel_size) in enumerate(
            zip(stage_channels, stage_depths, kernel_sizes)
        ):
            blocks: list[nn.Module] = []
            stage_downsample = stage_index >= 3
            for repeat_index in range(repeats):
                stride = 2 if stage_downsample and repeat_index == 0 else 1
                probability = float(stochastic_depth) * block_index / max(1, total_blocks - 1)
                blocks.append(
                    MBConvBlock(
                        input_channels=channels,
                        output_channels=channels,
                        expansion_ratio=float(expansion_ratio),
                        kernel_size=kernel_size,
                        stride=stride,
                        se_ratio=float(se_ratio),
                        stochastic_depth=probability,
                    )
                )
                block_index += 1
            stages.append(nn.Sequential(*blocks))
            downsample_flags.append(stage_downsample)
        self.stages = nn.ModuleList(stages)
        self.stage_downsamples = downsample_flags
        self.tail = nn.Sequential(
            nn.Conv2d(stage_channels[-1], int(head_channels), 1, bias=False),
            nn.GroupNorm(group_count(int(head_channels)), int(head_channels)),
            nn.SiLU(),
        )
        self.embedding_projection = nn.Sequential(
            nn.Linear(2 * int(head_channels), embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
