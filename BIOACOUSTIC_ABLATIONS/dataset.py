from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy import signal
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, Sampler


AUDIO_EXTENSIONS = {".wav", ".flac", ".aif", ".aiff", ".ogg", ".oga", ".au", ".caf"}


def build_manifest(dataset_paths: dict[str, Path]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for dataset_name, dataset_root in dataset_paths.items():
        for class_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
            label_name = f"{dataset_name}/{class_dir.name}"
            for path in sorted(
                path for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
            ):
                try:
                    info = sf.info(path)
                except RuntimeError:
                    continue
                if info.samplerate <= 0 or info.frames <= 0:
                    continue
                records.append(
                    {
                        "path": str(path.resolve()),
                        "dataset": str(dataset_name),
                        "class_name": str(class_dir.name),
                        "label_name": label_name,
                        "sample_rate": int(info.samplerate),
                        "num_frames": int(info.frames),
                        "duration_seconds": float(info.frames) / float(info.samplerate),
                    }
                )
    if not records:
        raise RuntimeError("No readable audio files were found.")
    return reindex_manifest(pd.DataFrame(records))


def reindex_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    result = manifest.copy()
    labels = sorted(result["label_name"].astype(str).unique())
    label_to_id = {label: index for index, label in enumerate(labels)}
    result["label_id"] = result["label_name"].map(label_to_id).astype(int)
    result = result.sort_values(["dataset", "label_name", "path"]).reset_index(drop=True)
    result["recording_id"] = np.arange(len(result), dtype=np.int64)
    return result


def filter_minimum_class_support(
    manifest: pd.DataFrame,
    minimum_samples_per_class: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    support = (
        manifest.groupby(["dataset", "class_name", "label_name"], observed=True)
        .size().rename("num_recordings").reset_index()
    )
    retained_names = set(
        support.loc[support["num_recordings"] >= int(minimum_samples_per_class), "label_name"]
    )
    excluded = support[~support["label_name"].isin(retained_names)].copy()
    filtered = manifest[manifest["label_name"].isin(retained_names)].copy()
    if filtered.empty:
        raise RuntimeError("No classes remain after minimum-support filtering.")
    filtered = reindex_manifest(filtered)
    retained = (
        filtered.groupby(["dataset", "class_name", "label_name", "label_id"], observed=True)
        .size().rename("num_recordings").reset_index().sort_values("label_id")
    )
    return filtered, retained, excluded


def assign_cv_folds(manifest: pd.DataFrame, num_folds: int, seed: int) -> pd.DataFrame:
    result = manifest.copy().reset_index(drop=True)
    result["cv_fold"] = -1
    labels = result["label_id"].to_numpy(dtype=np.int64)
    splitter = StratifiedKFold(n_splits=int(num_folds), shuffle=True, random_state=int(seed))
    for fold, (_, test_indices) in enumerate(splitter.split(np.zeros(len(result)), labels), start=1):
        result.loc[test_indices, "cv_fold"] = fold
    return result


def split_class_counts(manifest: pd.DataFrame) -> pd.DataFrame:
    counts = (
        manifest.groupby(["label_id", "label_name", "split"], observed=True)
        .size()
        .unstack("split", fill_value=0)
        .reset_index()
    )
    for split in ("train", "val", "test"):
        if split not in counts:
            counts[split] = 0
    return counts[["label_id", "label_name", "train", "val", "test"]].sort_values("label_id")


def validate_fold_manifest(manifest: pd.DataFrame, test_fold: int) -> pd.DataFrame:
    counts = split_class_counts(manifest)
    missing = counts[(counts[["train", "val", "test"]] < 1).any(axis=1)]
    if not missing.empty:
        details = "; ".join(
            f"{row.label_name}: train={int(row.train)}, val={int(row.val)}, test={int(row.test)}"
            for row in missing.itertuples(index=False)
        )
        raise RuntimeError(
            f"Fold {int(test_fold)} does not contain every retained class in train/val/test: {details}"
        )
    return counts


def make_fold_manifest(
    manifest: pd.DataFrame,
    test_fold: int,
    validation_fraction: float,
    seed: int,
) -> pd.DataFrame:
    result = manifest.copy()
    result["split"] = "train"
    result.loc[result["cv_fold"] == int(test_fold), "split"] = "test"

    remaining = result[result["split"] == "train"]
    for label_id, group in remaining.groupby("label_id", sort=True):
        indices = group.index.to_numpy(copy=True)
        if len(indices) < 2:
            raise RuntimeError(
                f"Fold {int(test_fold)} leaves fewer than two non-test recordings for label {int(label_id)}."
            )
        rng = np.random.default_rng(int(seed) + 10007 * int(test_fold) + int(label_id))
        rng.shuffle(indices)
        n_val = max(1, int(round(len(indices) * float(validation_fraction))))
        n_val = min(n_val, len(indices) - 1)
        result.loc[indices[:n_val], "split"] = "val"

    validate_fold_manifest(result, test_fold)
    return result


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
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
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms > 1e-8:
        audio = audio / rms
    return np.clip(audio, -10.0, 10.0).astype(np.float32, copy=False)


def read_mono_crop(
    path: str | Path,
    sample_rate: int,
    total_frames: int,
    segment_seconds: float,
    training: bool,
    crop_index: int,
    num_crops: int,
) -> np.ndarray:
    desired = max(1, int(round(float(segment_seconds) * int(sample_rate))))
    max_start = max(0, int(total_frames) - desired)
    if training:
        start = random.randint(0, max_start) if max_start else 0
    elif num_crops <= 1 or max_start == 0:
        start = max_start // 2
    else:
        start = int(round((crop_index / max(1, num_crops - 1)) * max_start))
    with sf.SoundFile(path) as handle:
        handle.seek(start)
        frames = handle.read(frames=desired, dtype="float32", always_2d=True)
    mono = np.mean(frames, axis=1, dtype=np.float32) if frames.size else np.zeros(0, dtype=np.float32)
    if len(mono) < desired:
        mono = np.pad(mono, (0, desired - len(mono)))
    return np.asarray(mono[:desired], dtype=np.float32)


class AudioSegmentDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        segment_seconds: float,
        training: bool,
        eval_crops: int,
        waveform_normalization: str,
        forced_output_rate: int | None = None,
    ) -> None:
        self.rows = manifest[manifest["split"] == split].copy().reset_index(drop=True)
        self.segment_seconds = float(segment_seconds)
        self.training = bool(training)
        self.eval_crops = 1 if training else max(1, int(eval_crops))
        self.waveform_normalization = str(waveform_normalization)
        self.forced_output_rate = int(forced_output_rate) if forced_output_rate else None

    def __len__(self) -> int:
        return len(self.rows) * self.eval_crops

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index // self.eval_crops]
        crop_index = index % self.eval_crops
        native_rate = int(row["sample_rate"])
        waveform = read_mono_crop(
            row["path"], native_rate, int(row["num_frames"]), self.segment_seconds,
            self.training, crop_index, self.eval_crops,
        )
        output_rate = self.forced_output_rate or native_rate
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
        }


def collate_audio_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.tensor([len(item["waveform"]) for item in items], dtype=torch.long)
    waveforms = torch.zeros(len(items), int(lengths.max()), dtype=torch.float32)
    for index, item in enumerate(items):
        waveforms[index, : lengths[index]] = item["waveform"]
    return {
        "waveform": waveforms,
        "lengths": lengths,
        "sample_rate": torch.tensor([item["sample_rate"] for item in items], dtype=torch.long),
        "original_sample_rate": torch.tensor([item["original_sample_rate"] for item in items], dtype=torch.long),
        "label": torch.tensor([item["label"] for item in items], dtype=torch.long),
        "recording_id": torch.tensor([item["recording_id"] for item in items], dtype=torch.long),
        "crop_index": torch.tensor([item["crop_index"] for item in items], dtype=torch.long),
        "dataset": [item["dataset"] for item in items],
        "label_name": [item["label_name"] for item in items],
        "path": [item["path"] for item in items],
    }


class DatasetBalancedPKBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: AudioSegmentDataset,
        batch_size: int,
        samples_per_class: int,
        seed: int,
        num_samples: int | None = None,
    ) -> None:
        rows = dataset.rows.reset_index(drop=True)
        nested: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, row in rows.iterrows():
            nested[str(row["dataset"])][int(row["label_id"])].append(int(index))
        self.datasets = sorted(nested)
        self.classes = {name: sorted(nested[name]) for name in self.datasets}
        self.indices = {
            name: {label: np.asarray(nested[name][label], dtype=np.int64) for label in self.classes[name]}
            for name in self.datasets
        }
        self.batch_size = int(batch_size)
        self.samples_per_class = int(samples_per_class)
        self.classes_per_batch = self.batch_size // self.samples_per_class
        self.seed = int(seed)
        self.epoch = 0
        target = int(num_samples) if num_samples and num_samples > 0 else len(rows)
        self.num_batches = max(1, int(np.ceil(target / self.batch_size)))

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        total_classes = sum(len(labels) for labels in self.classes.values())
        require_unique = total_classes >= self.classes_per_batch
        for _ in range(self.num_batches):
            batch: list[int] = []
            selected_labels: set[int] = set()
            for _ in range(self.classes_per_batch):
                for _ in range(100):
                    dataset = self.datasets[int(rng.integers(0, len(self.datasets)))]
                    labels = self.classes[dataset]
                    label = labels[int(rng.integers(0, len(labels)))]
                    if not require_unique or label not in selected_labels:
                        break
                selected_labels.add(label)
                candidates = self.indices[dataset][label]
                chosen = rng.choice(
                    candidates,
                    size=self.samples_per_class,
                    replace=len(candidates) < self.samples_per_class,
                )
                batch.extend(int(value) for value in chosen)
            rng.shuffle(batch)
            yield batch
