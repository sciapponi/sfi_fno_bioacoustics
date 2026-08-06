from __future__ import annotations

from copy import deepcopy
from typing import Any

from torch import nn

from .backends import (
    CNNTemporalPyramidEncoder,
    FNOTemporalPyramidEncoder,
    GridEncoderBase,
    MBConvTemporalPyramidEncoder,
)
from .frontends import MelSpectrogramFrontend, PhysicalModulationFrontend
from .models import ScaleFusionClassifier


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "max_frequency_cnn": {
        "frontend": "mel",
        "backend": "cnn",
        "rate_mode": "corpus_max",
        "description": "Corpus-maximum-rate mel plus CNN temporal-pyramid encoder.",
    },
    "max_frequency_fno": {
        "frontend": "mel",
        "backend": "fno",
        "rate_mode": "corpus_max",
        "description": "Corpus-maximum-rate mel plus FNO temporal-pyramid encoder.",
    },
    "max_frequency_mbconv": {
        "frontend": "mel",
        "backend": "mbconv",
        "rate_mode": "corpus_max",
        "description": "Corpus-maximum-rate mel plus compact MBConv temporal pyramid.",
    },
    "mel_48k_cnn": {
        "frontend": "mel",
        "backend": "cnn",
        "rate_mode": "fixed_48k",
        "description": "Fixed-48-kHz mel plus CNN temporal-pyramid encoder.",
    },
    "mel_48k_fno": {
        "frontend": "mel",
        "backend": "fno",
        "rate_mode": "fixed_48k",
        "description": "Fixed-48-kHz mel plus FNO temporal-pyramid encoder.",
    },
    "mel_48k_mbconv": {
        "frontend": "mel",
        "backend": "mbconv",
        "rate_mode": "fixed_48k",
        "description": "Fixed-48-kHz mel plus compact MBConv temporal pyramid.",
    },
    "sfi_cnn": {
        "frontend": "physical_modulation",
        "backend": "cnn",
        "rate_mode": "native",
        "description": "Native-rate physical modulation plus CNN temporal pyramid.",
    },
    "sfi_fno": {
        "frontend": "physical_modulation",
        "backend": "fno",
        "rate_mode": "native",
        "description": "Native-rate physical modulation plus FNO temporal pyramid.",
    },
    "sfi_mbconv": {
        "frontend": "physical_modulation",
        "backend": "mbconv",
        "rate_mode": "native",
        "description": "Native-rate physical modulation plus compact MBConv temporal pyramid.",
    },
}

RUN_SPECS: list[dict[str, str]] = [
    {
        "run_name": f"{model_name}__focal_contrastive",
        "model": model_name,
        "training_regime": "focal_contrastive",
    }
    for model_name in MODEL_SPECS
]
RUN_NAMES = [item["run_name"] for item in RUN_SPECS]
MODEL_NAMES = list(MODEL_SPECS)


def default_architecture_config() -> dict[str, Any]:
    return {
        "experiment_variant": "multiresolution_temporal_pyramid",
        "analysis_windows_ms": (16.0, 64.0, 256.0),
        "hop_ratio": 0.25,
        "max_window_samples": 0,
        "frequency_bins": 128,
        "minimum_frequency_hz": 5.0,
        "maximum_frequency_hz": 120000.0,
        "q_factor": 12.0,
        "modulation_rates_hz": (2.0, 4.0, 8.0, 16.0),
        "nyquist_rolloff": 0.92,
        "base_channels": 12,
        "embedding_dim": 192,
        "depth": 2,
        "fno_modes_frequency": 4,
        "fno_modes_time": 4,
        "mbconv_stage_channels": (24, 48, 96, 128),
        "mbconv_stage_depths": (2, 2, 2, 2),
        "mbconv_kernel_sizes": (3, 3, 5, 5),
        "mbconv_expansion_ratio": 4.0,
        "mbconv_se_ratio": 0.25,
        "mbconv_stochastic_depth": 0.10,
        "mbconv_head_channels": 192,
        "dropout": 0.10,
        "contrastive_projection_dim": 128,
    }


def build_frontend(model_name: str, config: dict[str, Any]) -> nn.Module:
    spec = MODEL_SPECS[model_name]
    windows = tuple(float(value) for value in config["analysis_windows_ms"])
    if len(windows) not in (1, 3):
        raise ValueError("analysis_windows_ms must contain one or three durations.")
    if spec["frontend"] == "mel":
        return MelSpectrogramFrontend(
            frequency_bins=int(config["frequency_bins"]),
            minimum_hz=max(5.0, float(config["minimum_frequency_hz"])),
            window_ms=windows,
            hop_ratio=float(config["hop_ratio"]),
            max_window_samples=int(config["max_window_samples"]),
            normalize=True,
        )
    return PhysicalModulationFrontend(
        minimum_hz=float(config["minimum_frequency_hz"]),
        maximum_hz=float(config["maximum_frequency_hz"]),
        frequency_bins=int(config["frequency_bins"]),
        q_factor=float(config["q_factor"]),
        window_ms=windows,
        modulation_rates_hz=tuple(float(value) for value in config["modulation_rates_hz"]),
        nyquist_rolloff=float(config["nyquist_rolloff"]),
        hop_ratio=float(config["hop_ratio"]),
        max_window_samples=int(config["max_window_samples"]),
        normalize=True,
    )


def build_encoder(
    backend_name: str,
    input_channels: int,
    num_scales: int,
    config: dict[str, Any],
) -> GridEncoderBase:
    if backend_name == "cnn":
        return CNNTemporalPyramidEncoder(
            input_channels=int(input_channels),
            num_scales=int(num_scales),
            base_channels=int(config["base_channels"]),
            embedding_dim=int(config["embedding_dim"]),
            depth=int(config["depth"]),
            dropout=float(config["dropout"]),
        )
    if backend_name == "fno":
        return FNOTemporalPyramidEncoder(
            input_channels=int(input_channels),
            num_scales=int(num_scales),
            base_channels=int(config["base_channels"]),
            embedding_dim=int(config["embedding_dim"]),
            depth=int(config["depth"]),
            modes_frequency=int(config["fno_modes_frequency"]),
            modes_time=int(config["fno_modes_time"]),
            dropout=float(config["dropout"]),
        )
    if backend_name == "mbconv":
        return MBConvTemporalPyramidEncoder(
            input_channels=int(input_channels),
            num_scales=int(num_scales),
            embedding_dim=int(config["embedding_dim"]),
            stage_channels=tuple(int(value) for value in config["mbconv_stage_channels"]),
            stage_depths=tuple(int(value) for value in config["mbconv_stage_depths"]),
            kernel_sizes=tuple(int(value) for value in config["mbconv_kernel_sizes"]),
            expansion_ratio=float(config["mbconv_expansion_ratio"]),
            se_ratio=float(config["mbconv_se_ratio"]),
            stochastic_depth=float(config["mbconv_stochastic_depth"]),
            head_channels=int(config["mbconv_head_channels"]),
            dropout=float(config["dropout"]),
        )
    raise KeyError(f"Unknown backend: {backend_name}")


def build_model(
    model_name: str,
    num_classes: int,
    architecture_config: dict[str, Any],
    use_contrastive_head: bool,
) -> nn.Module:
    if model_name not in MODEL_SPECS:
        raise KeyError(f"Unknown model {model_name!r}. Available: {MODEL_NAMES}")
    config = deepcopy(architecture_config)
    spec = MODEL_SPECS[model_name]
    frontend = build_frontend(model_name, config)
    encoder = build_encoder(
        spec["backend"], frontend.scale_channels, frontend.num_scales, config
    )
    contrastive_dim = (
        int(config["contrastive_projection_dim"]) if use_contrastive_head else None
    )
    return ScaleFusionClassifier(
        frontend=frontend,
        encoder=encoder,
        num_classes=int(num_classes),
        contrastive_dim=contrastive_dim,
    )


def model_table(
    experiment_variant: str,
    analysis_windows_ms: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in RUN_SPECS:
        rows.append(
            {
                **run,
                **MODEL_SPECS[run["model"]],
                "experiment_variant": experiment_variant,
                "analysis_windows_ms": ",".join(
                    f"{value:g}" for value in analysis_windows_ms
                ),
            }
        )
    return rows
