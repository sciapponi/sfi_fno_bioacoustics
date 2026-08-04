from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

AUDIO_EXTENSIONS = {
    ".wav", ".flac", ".aif", ".aiff", ".ogg", ".oga", ".au", ".caf"
}


def _read_mono_region(path: Path, start: int, frames: int) -> tuple[np.ndarray, int]:
    with sf.SoundFile(path) as handle:
        sr = int(handle.samplerate)
        start = max(0, min(start, max(0, len(handle) - 1)))
        handle.seek(start)
        audio = handle.read(frames=frames, dtype="float32", always_2d=True)
    if audio.size == 0:
        return np.zeros(1, dtype=np.float32), sr
    mono = np.mean(audio, axis=1, dtype=np.float32)
    mono = mono - float(np.mean(mono))
    return mono.astype(np.float32, copy=False), sr


def estimate_usable_bandwidth(
    path: Path,
    sample_rate: int,
    num_frames: int,
    scan_seconds: float = 2.0,
    energy_quantile: float = 0.995,
    margin: float = 1.10,
) -> float:
    """Estimate the highest frequency containing substantial non-floor energy.

    This is deliberately conservative. Three regions are sampled from the file,
    their Welch spectra are median-combined, a robust spectral floor is removed,
    and the requested cumulative-energy quantile is returned with a margin.
    """
    nyquist = sample_rate / 2.0
    region_frames = max(256, int(round(scan_seconds * sample_rate)))
    if num_frames <= region_frames:
        starts = [0]
    else:
        starts = [
            0,
            max(0, (num_frames - region_frames) // 2),
            max(0, num_frames - region_frames),
        ]

    spectra: list[np.ndarray] = []
    freqs_ref: np.ndarray | None = None
    for start in starts:
        audio, sr = _read_mono_region(path, start, region_frames)
        if sr != sample_rate or len(audio) < 64:
            continue
        peak = float(np.max(np.abs(audio)))
        if peak <= 1e-8:
            continue
        audio = audio / peak
        nperseg = min(len(audio), 8192)
        if nperseg < 64:
            continue
        freqs, psd = signal.welch(
            audio,
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend=False,
            scaling="spectrum",
        )
        spectra.append(psd.astype(np.float64))
        freqs_ref = freqs

    if not spectra or freqs_ref is None:
        return float(0.90 * nyquist)

    min_len = min(len(x) for x in spectra)
    stacked = np.stack([x[:min_len] for x in spectra], axis=0)
    psd = np.median(stacked, axis=0)
    freqs = freqs_ref[:min_len]

    # Smooth only enough to avoid isolated-bin spikes defining the bandwidth.
    if len(psd) >= 9:
        kernel = min(31, len(psd) // 2 * 2 - 1)
        kernel = max(5, kernel)
        psd = signal.medfilt(psd, kernel_size=kernel)

    positive = psd[1:] if len(psd) > 1 else psd
    floor = float(np.quantile(positive, 0.20)) if positive.size else 0.0
    excess = np.maximum(psd - floor, 0.0)
    total = float(np.sum(excess))
    if not math.isfinite(total) or total <= 1e-20:
        return float(0.90 * nyquist)

    cumulative = np.cumsum(excess) / total
    idx = int(np.searchsorted(cumulative, energy_quantile, side="left"))
    idx = min(max(idx, 1), len(freqs) - 1)
    estimate = float(freqs[idx] * margin)
    return float(np.clip(estimate, 100.0, 0.98 * nyquist))


def build_manifest(
    data_root: Path,
    dataset_names: list[str],
    estimate_bandwidths: bool = True,
    bandwidth_scan_seconds: float = 2.0,
    bandwidth_energy_quantile: float = 0.995,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for dataset_name in dataset_names:
        dataset_root = data_root / dataset_name
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")

        class_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
        if not class_dirs:
            raise RuntimeError(f"No class subfolders found in: {dataset_root}")

        for class_dir in class_dirs:
            audio_paths = sorted(
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
            )
            if not audio_paths:
                continue

            label_name = f"{dataset_name}/{class_dir.name}"
            for path in audio_paths:
                try:
                    info = sf.info(path)
                except RuntimeError as exc:
                    print(f"Skipping unreadable audio file {path}: {exc}")
                    continue

                sr = int(info.samplerate)
                num_frames = int(info.frames)
                if sr <= 0 or num_frames <= 0:
                    continue

                if estimate_bandwidths:
                    usable_bandwidth = estimate_usable_bandwidth(
                        path=path,
                        sample_rate=sr,
                        num_frames=num_frames,
                        scan_seconds=bandwidth_scan_seconds,
                        energy_quantile=bandwidth_energy_quantile,
                    )
                else:
                    usable_bandwidth = 0.90 * (sr / 2.0)

                records.append(
                    {
                        "path": str(path.resolve()),
                        "dataset": dataset_name,
                        "class_name": class_dir.name,
                        "label_name": label_name,
                        "sample_rate": sr,
                        "num_frames": num_frames,
                        "duration_seconds": num_frames / sr,
                        "channels": int(info.channels),
                        "usable_bandwidth_hz": float(usable_bandwidth),
                    }
                )

    if not records:
        raise RuntimeError("No readable audio files were found.")

    manifest = pd.DataFrame.from_records(records)
    labels = sorted(manifest["label_name"].unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    manifest["label_id"] = manifest["label_name"].map(label_to_id).astype(int)
    manifest["recording_id"] = np.arange(len(manifest), dtype=np.int64)
    return manifest.sort_values(["dataset", "label_name", "path"]).reset_index(drop=True)


def stratified_recording_split(
    manifest: pd.DataFrame,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> pd.DataFrame:
    if not (0.0 < val_fraction < 1.0 and 0.0 < test_fraction < 1.0):
        raise ValueError("Validation and test fractions must be between 0 and 1.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("Validation and test fractions must sum to less than 1.")

    rng = np.random.default_rng(seed)
    result = manifest.copy()
    result["split"] = ""

    for label_id, group in result.groupby("label_id", sort=True):
        indices = group.index.to_numpy(copy=True)
        n = len(indices)
        if n < 3:
            label_name = str(group.iloc[0]["label_name"])
            raise ValueError(
                f"Class '{label_name}' has {n} recordings. At least 3 are required "
                "to place one recording in train, validation, and test."
            )

        rng.shuffle(indices)
        n_val = max(1, int(round(n * val_fraction)))
        n_test = max(1, int(round(n * test_fraction)))
        while n_val + n_test > n - 1:
            if n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break

        val_idx = indices[:n_val]
        test_idx = indices[n_val : n_val + n_test]
        train_idx = indices[n_val + n_test :]
        result.loc[train_idx, "split"] = "train"
        result.loc[val_idx, "split"] = "val"
        result.loc[test_idx, "split"] = "test"

    if (result["split"] == "").any():
        raise RuntimeError("Internal split error: some recordings were not assigned.")

    counts = result.groupby(["label_name", "split"]).size().unstack(fill_value=0)
    required = {"train", "val", "test"}
    if not required.issubset(counts.columns):
        raise RuntimeError("Internal split error: a split is missing.")
    if (counts[list(required)] < 1).any().any():
        raise RuntimeError("Internal split error: a class is absent from a split.")

    return result


def split_counts(manifest: pd.DataFrame) -> pd.DataFrame:
    return (
        manifest.groupby(["dataset", "label_name", "split"], observed=True)
        .size()
        .rename("num_recordings")
        .reset_index()
        .sort_values(["dataset", "label_name", "split"])
    )
