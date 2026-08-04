"""Modular bioacoustic sample-rate ablation package."""

from .backends import CNNGridEncoder, FNOGridEncoder, TokenSetEncoder
from .frontends import (
    MelFrontendConfig,
    MelSpectrogramFrontend,
    PhysicalModulationConfig,
    PhysicalModulationFrontend,
)
from .models import GridClassifier, NestedBandwidthClassifier, count_parameters
from .registry import ArchitectureConfig, MODEL_NAMES, MODEL_SPECS, build_model

__all__ = [
    "ArchitectureConfig",
    "CNNGridEncoder",
    "FNOGridEncoder",
    "GridClassifier",
    "MelFrontendConfig",
    "MelSpectrogramFrontend",
    "MODEL_NAMES",
    "MODEL_SPECS",
    "NestedBandwidthClassifier",
    "PhysicalModulationConfig",
    "PhysicalModulationFrontend",
    "TokenSetEncoder",
    "build_model",
    "count_parameters",
]
