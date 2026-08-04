from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from torch import nn

from .backends import CNNGridEncoder, FNOGridEncoder, GridEncoderBase, TokenSetEncoder
from .frontends import (
    FeatureFrontend,
    MelFrontendConfig,
    MelSpectrogramFrontend,
    PhysicalModulationConfig,
    PhysicalModulationFrontend,
)
from .models import GridClassifier, NestedBandwidthClassifier


FrontendName = Literal["mel", "physical_modulation"]
BackendName = Literal["set", "cnn", "fno"]
TrainingStructure = Literal["standard", "two_band", "three_band"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    frontend: FrontendName
    backend: BackendName
    multiresolution: bool
    training_structure: TrainingStructure = "standard"
    description: str = ""

    @property
    def uses_dataset_resampling(self) -> bool:
        return self.frontend == "mel"


MODEL_SPECS: dict[str, ModelSpec] = {
    "dataset_mel": ModelSpec(
        "dataset_mel", "mel", "set", False,
        description="Dataset-specific mel resampling with a frequency-token set encoder.",
    ),
    "mel_single_cnn": ModelSpec(
        "mel_single_cnn", "mel", "cnn", False,
        description="Single-window mel frontend with a local CNN backend.",
    ),
    "mel_single_fno": ModelSpec(
        "mel_single_fno", "mel", "fno", False,
        description="Single-window mel frontend with an FNO backend.",
    ),
    "mel_multires_cnn": ModelSpec(
        "mel_multires_cnn", "mel", "cnn", True,
        description="Three-window mel frontend with a local CNN backend.",
    ),
    "mel_multires_fno": ModelSpec(
        "mel_multires_fno", "mel", "fno", True,
        description="Three-window mel frontend with an FNO backend.",
    ),
    "sfi_single_cnn": ModelSpec(
        "sfi_single_cnn", "physical_modulation", "cnn", False,
        description="Single-window native-rate physical modulation frontend with CNN.",
    ),
    "sfi_single_fno": ModelSpec(
        "sfi_single_fno", "physical_modulation", "fno", False,
        description="Single-window native-rate physical modulation frontend with FNO.",
    ),
    "sfi_multires_cnn": ModelSpec(
        "sfi_multires_cnn", "physical_modulation", "cnn", True,
        description="Three-window native-rate physical modulation frontend with CNN.",
    ),
    "sfi_multires_fno": ModelSpec(
        "sfi_multires_fno", "physical_modulation", "fno", True,
        description="Three-window native-rate physical modulation frontend with FNO.",
    ),
    "sfi_multires_fno_bandwidth_equivariant": ModelSpec(
        "sfi_multires_fno_bandwidth_equivariant",
        "physical_modulation", "fno", True, "two_band",
        "Two nested bands: <=20 kHz and >20 kHz.",
    ),
    "sfi_multires_fno_three_bandwidth_equivariant": ModelSpec(
        "sfi_multires_fno_three_bandwidth_equivariant",
        "physical_modulation", "fno", True, "three_band",
        "Three nested bands: <=8 kHz, 8-20 kHz, and >20 kHz.",
    ),
}

MODEL_NAMES = list(MODEL_SPECS)


@dataclass(frozen=True)
class ArchitectureConfig:
    single_window_ms: float = 64.0
    multiresolution_windows_ms: tuple[float, ...] = (16.0, 64.0, 256.0)
    frequency_bins: int = 96
    time_bins: int = 96
    minimum_frequency_hz: float = 5.0
    maximum_frequency_hz: float = 120000.0
    q_factor: float = 12.0
    modulation_rates_hz: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0)
    nyquist_rolloff: float = 0.92
    base_channels: int = 16
    embedding_dim: int = 192
    depth: int = 2
    fno_modes_frequency: int = 8
    fno_modes_time: int = 8
    token_dim: int = 96
    token_layers: int = 2
    token_heads: int = 4
    dropout: float = 0.10
    two_band_cutoff_hz: float = 20000.0
    three_band_low_cutoff_hz: float = 8000.0
    three_band_mid_cutoff_hz: float = 20000.0
    residual_gate_bias: float = -1.0
    contrastive_projection_dim: int = 128


def build_frontend(spec: ModelSpec, config: ArchitectureConfig) -> FeatureFrontend:
    windows = (
        config.multiresolution_windows_ms
        if spec.multiresolution
        else (config.single_window_ms,)
    )
    if spec.frontend == "mel":
        return MelSpectrogramFrontend(
            MelFrontendConfig(
                frequency_bins=config.frequency_bins,
                time_bins=config.time_bins,
                minimum_hz=max(5.0, config.minimum_frequency_hz),
                window_ms=tuple(windows),
                normalize=True,
            )
        )
    return PhysicalModulationFrontend(
        PhysicalModulationConfig(
            minimum_hz=config.minimum_frequency_hz,
            maximum_hz=config.maximum_frequency_hz,
            frequency_bins=config.frequency_bins,
            time_bins=config.time_bins,
            q_factor=config.q_factor,
            window_ms=tuple(windows),
            modulation_rates_hz=config.modulation_rates_hz,
            nyquist_rolloff=config.nyquist_rolloff,
            normalize=spec.training_structure == "standard",
        )
    )


def build_encoder(
    backend: BackendName,
    input_channels: int,
    config: ArchitectureConfig,
) -> GridEncoderBase:
    if backend == "cnn":
        return CNNGridEncoder(
            input_channels=input_channels,
            base_channels=config.base_channels,
            embedding_dim=config.embedding_dim,
            depth=config.depth,
            dropout=config.dropout,
        )
    if backend == "fno":
        return FNOGridEncoder(
            input_channels=input_channels,
            base_channels=config.base_channels,
            embedding_dim=config.embedding_dim,
            depth=config.depth,
            modes_frequency=config.fno_modes_frequency,
            modes_time=config.fno_modes_time,
            dropout=config.dropout,
        )
    return TokenSetEncoder(
        input_channels=input_channels,
        token_dim=config.token_dim,
        embedding_dim=config.embedding_dim,
        layers=config.token_layers,
        heads=config.token_heads,
        dropout=config.dropout,
    )


def build_model(
    model_name: str,
    num_classes: int,
    config: ArchitectureConfig,
    use_contrastive_head: bool,
) -> nn.Module:
    """Construct one model from an explicit, inspectable specification."""
    if model_name not in MODEL_SPECS:
        raise KeyError(f"Unknown model {model_name!r}. Available: {MODEL_NAMES}")
    spec = MODEL_SPECS[model_name]
    frontend = build_frontend(spec, config)
    contrastive_dim = (
        config.contrastive_projection_dim if use_contrastive_head else None
    )

    if spec.training_structure == "standard":
        encoder = build_encoder(spec.backend, frontend.output_channels, config)
        return GridClassifier(
            frontend=frontend,
            encoder=encoder,
            num_classes=num_classes,
            contrastive_dim=contrastive_dim,
        )

    def encoder_factory() -> GridEncoderBase:
        return build_encoder(spec.backend, frontend.output_channels, config)

    boundaries = (
        (config.two_band_cutoff_hz,)
        if spec.training_structure == "two_band"
        else (
            config.three_band_low_cutoff_hz,
            config.three_band_mid_cutoff_hz,
        )
    )
    return NestedBandwidthClassifier(
        frontend=frontend,
        encoder_factory=encoder_factory,
        num_classes=num_classes,
        boundaries_hz=boundaries,
        residual_gate_bias=config.residual_gate_bias,
        contrastive_dim=contrastive_dim,
    )


def model_table() -> list[dict[str, object]]:
    return [
        {
            "model": spec.name,
            "frontend": spec.frontend,
            "backend": spec.backend,
            "multiresolution": spec.multiresolution,
            "training_structure": spec.training_structure,
            "uses_dataset_resampling": spec.uses_dataset_resampling,
            "description": spec.description,
        }
        for spec in MODEL_SPECS.values()
    ]
