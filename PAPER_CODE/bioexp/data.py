from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy import signal
from torch.utils.data import Dataset


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Polyphase resampling without torchaudio."""
    if int(source_rate) == int(target_rate):
        return audio.astype(np.float32, copy=False)
    divisor = math.gcd(int(source_rate), int(target_rate))
    result = signal.resample_poly(
        audio,
        up=int(target_rate // divisor),
        down=int(source_rate // divisor),
        window=("kaiser", 8.6),
    )
    return np.asarray(result, dtype=np.float32)


def normalize_waveform(audio: np.ndarray, mode: str) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    audio = audio - np.mean(audio, dtype=np.float64)
    if mode == "none":
        return audio
    if mode == "peak":
        peak = float(np.max(np.abs(audio)))
        return audio / peak if peak > 1e-8 else audio
    if mode == "rms":
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        if rms > 1e-8:
            audio = audio / rms
        return np.clip(audio, -10.0, 10.0).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported waveform normalization: {mode}")


def read_mono_crop(
    path: str | Path,
    sample_rate: int,
    total_frames: int,
    segment_seconds: float,
    training: bool,
    crop_index: int,
    num_crops: int,
) -> np.ndarray:
    desired = max(1, int(round(segment_seconds * sample_rate)))
    max_start = max(0, int(total_frames) - desired)
    if training:
        start = random.randint(0, max_start) if max_start else 0
    elif num_crops <= 1 or max_start == 0:
        start = max_start // 2
    else:
        start = int(round((crop_index / max(1, num_crops - 1)) * max_start))

    with sf.SoundFile(path) as handle:
        if int(handle.samplerate) != int(sample_rate):
            raise RuntimeError(
                f"Manifest rate {sample_rate} does not match {handle.samplerate}: {path}"
            )
        handle.seek(start)
        frames = handle.read(frames=desired, dtype="float32", always_2d=True)

    if frames.size == 0:
        mono = np.zeros(desired, dtype=np.float32)
    else:
        mono = np.mean(frames, axis=1, dtype=np.float32)
        if len(mono) < desired:
            mono = np.pad(mono, (0, desired - len(mono)))
    return np.asarray(mono[:desired], dtype=np.float32)


def lowpass_and_resample(
    audio: np.ndarray,
    source_rate: int,
    cutoff_hz: float,
    target_rate: int,
) -> np.ndarray:
    """Create a bandwidth-limited counterfactual view of one crop."""
    nyquist = 0.5 * float(source_rate)
    if cutoff_hz >= nyquist:
        filtered = audio
    else:
        sos = signal.butter(8, float(cutoff_hz), btype="lowpass", fs=source_rate, output="sos")
        try:
            filtered = signal.sosfiltfilt(sos, audio)
        except ValueError:
            filtered = signal.sosfilt(sos, audio)
    return resample_audio(np.asarray(filtered, dtype=np.float32), source_rate, target_rate)


@dataclass(frozen=True)
class CounterfactualView:
    """A view that preserves frequencies up to cutoff_hz at target_rate."""

    name: str
    cutoff_hz: float
    target_rate: int
    max_band: int


class AudioSegmentDataset(Dataset[dict[str, Any]]):
    """One waveform crop per item, optionally resampled for mel frontends."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        segment_seconds: float,
        training: bool,
        eval_crops: int,
        waveform_normalization: str,
        forced_output_rate: int | None = None,
        dataset_output_rates: dict[str, int] | None = None,
    ) -> None:
        rows = manifest[manifest["split"] == split].copy().reset_index(drop=True)
        if rows.empty:
            raise ValueError(f"No recordings found for split '{split}'.")
        if forced_output_rate is not None and dataset_output_rates is not None:
            raise ValueError("Choose forced_output_rate or dataset_output_rates, not both.")
        self.rows = rows
        self.segment_seconds = float(segment_seconds)
        self.training = bool(training)
        self.eval_crops = 1 if training else max(1, int(eval_crops))
        self.waveform_normalization = str(waveform_normalization)
        self.forced_output_rate = int(forced_output_rate) if forced_output_rate else None
        self.dataset_output_rates = (
            {str(k): int(v) for k, v in dataset_output_rates.items()}
            if dataset_output_rates is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.rows) * self.eval_crops

    def _output_rate(self, row: pd.Series) -> int:
        if self.forced_output_rate is not None:
            return self.forced_output_rate
        if self.dataset_output_rates is not None:
            return self.dataset_output_rates[str(row["dataset"])]
        return int(row["sample_rate"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index = index // self.eval_crops
        crop_index = index % self.eval_crops
        row = self.rows.iloc[row_index]
        native_rate = int(row["sample_rate"])
        waveform = read_mono_crop(
            row["path"], native_rate, int(row["num_frames"]), self.segment_seconds,
            self.training, crop_index, self.eval_crops,
        )
        output_rate = self._output_rate(row)
        if output_rate != native_rate:
            waveform = resample_audio(waveform, native_rate, output_rate)
        expected = max(1, int(round(self.segment_seconds * output_rate)))
        if len(waveform) < expected:
            waveform = np.pad(waveform, (0, expected - len(waveform)))
        waveform = normalize_waveform(waveform[:expected], self.waveform_normalization)
        return {
            "waveform": torch.from_numpy(waveform.copy()),
            "sample_rate": output_rate,
            "original_sample_rate": native_rate,
            "label": int(row["label_id"]),
            "recording_id": int(row["recording_id"]),
            "crop_index": crop_index,
            "dataset": str(row["dataset"]),
            "label_name": str(row["label_name"]),
            "path": str(row["path"]),
            "usable_bandwidth_hz": float(row["usable_bandwidth_hz"]),
        }


class NestedRateAudioDataset(Dataset[dict[str, Any]]):
    """Native crop plus one or more nested lower-bandwidth views.

    The native crop is normalized once. Each counterfactual is then low-pass
    filtered and resampled from that exact crop, so view differences are caused
    by bandwidth/sample-rate changes rather than by different crop locations.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        segment_seconds: float,
        training: bool,
        eval_crops: int,
        waveform_normalization: str,
        views: Sequence[CounterfactualView],
    ) -> None:
        rows = manifest[manifest["split"] == split].copy().reset_index(drop=True)
        if rows.empty:
            raise ValueError(f"No recordings found for split '{split}'.")
        if not views:
            raise ValueError("NestedRateAudioDataset requires at least one view.")
        for view in views:
            if view.target_rate / 2.0 <= view.cutoff_hz * 1.05:
                raise ValueError(
                    f"View {view.name}: target-rate Nyquist needs headroom above cutoff."
                )
        self.rows = rows
        self.segment_seconds = float(segment_seconds)
        self.training = bool(training)
        self.eval_crops = 1 if training else max(1, int(eval_crops))
        self.waveform_normalization = str(waveform_normalization)
        self.views = tuple(views)

    def __len__(self) -> int:
        return len(self.rows) * self.eval_crops

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index = index // self.eval_crops
        crop_index = index % self.eval_crops
        row = self.rows.iloc[row_index]
        native_rate = int(row["sample_rate"])
        native = read_mono_crop(
            row["path"], native_rate, int(row["num_frames"]), self.segment_seconds,
            self.training, crop_index, self.eval_crops,
        )
        native = normalize_waveform(native, self.waveform_normalization)

        view_items: list[dict[str, Any]] = []
        for view in self.views:
            eligible = (
                native_rate > int(round(view.target_rate * 1.05))
                and native_rate / 2.0 > view.cutoff_hz * 1.10
            )
            if eligible:
                value = lowpass_and_resample(
                    native, native_rate, view.cutoff_hz, view.target_rate
                )
                rate = int(view.target_rate)
            else:
                value = native.copy()
                rate = native_rate
            expected = max(1, int(round(self.segment_seconds * rate)))
            if len(value) < expected:
                value = np.pad(value, (0, expected - len(value)))
            view_items.append(
                {
                    "name": view.name,
                    "waveform": torch.from_numpy(np.asarray(value[:expected], dtype=np.float32).copy()),
                    "sample_rate": rate,
                    "available": bool(eligible),
                    "max_band": int(view.max_band),
                    "cutoff_hz": float(view.cutoff_hz),
                }
            )

        return {
            "waveform": torch.from_numpy(native.copy()),
            "sample_rate": native_rate,
            "original_sample_rate": native_rate,
            "views": view_items,
            "label": int(row["label_id"]),
            "recording_id": int(row["recording_id"]),
            "crop_index": crop_index,
            "dataset": str(row["dataset"]),
            "label_name": str(row["label_name"]),
            "path": str(row["path"]),
            "usable_bandwidth_hz": float(row["usable_bandwidth_hz"]),
        }


def _pad_waveforms(waveforms: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(w) for w in waveforms], dtype=torch.long)
    padded = torch.zeros((len(waveforms), int(lengths.max().item())), dtype=torch.float32)
    for index, waveform in enumerate(waveforms):
        padded[index, : len(waveform)] = waveform
    return padded, lengths


def collate_audio_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    waveforms, lengths = _pad_waveforms([item["waveform"] for item in items])
    return {
        "waveform": waveforms,
        "lengths": lengths,
        "sample_rate": torch.tensor([item["sample_rate"] for item in items], dtype=torch.long),
        "original_sample_rate": torch.tensor(
            [item["original_sample_rate"] for item in items], dtype=torch.long
        ),
        "label": torch.tensor([item["label"] for item in items], dtype=torch.long),
        "recording_id": torch.tensor([item["recording_id"] for item in items], dtype=torch.long),
        "crop_index": torch.tensor([item["crop_index"] for item in items], dtype=torch.long),
        "usable_bandwidth_hz": torch.tensor(
            [item["usable_bandwidth_hz"] for item in items], dtype=torch.float32
        ),
        "dataset": [item["dataset"] for item in items],
        "label_name": [item["label_name"] for item in items],
        "path": [item["path"] for item in items],
    }


def collate_nested_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch = collate_audio_batch(items)
    number_of_views = len(items[0]["views"])
    views: list[dict[str, Any]] = []
    for view_index in range(number_of_views):
        view_items = [item["views"][view_index] for item in items]
        waveforms, lengths = _pad_waveforms([item["waveform"] for item in view_items])
        views.append(
            {
                "name": str(view_items[0]["name"]),
                "waveform": waveforms,
                "lengths": lengths,
                "sample_rate": torch.tensor(
                    [item["sample_rate"] for item in view_items], dtype=torch.long
                ),
                "available": torch.tensor(
                    [item["available"] for item in view_items], dtype=torch.bool
                ),
                "max_band": int(view_items[0]["max_band"]),
                "cutoff_hz": float(view_items[0]["cutoff_hz"]),
            }
        )
    batch["views"] = views
    return batch
