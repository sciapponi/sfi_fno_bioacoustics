from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    silhouette_score,
)
from torch import nn
from torch.utils.data import DataLoader


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result = dict(batch)
    for key in [
        "waveform", "lengths", "sample_rate", "original_sample_rate",
        "label", "recording_id", "crop_index", "usable_bandwidth_hz",
    ]:
        if key in result and isinstance(result[key], torch.Tensor):
            result[key] = result[key].to(device, non_blocking=True)
    if "views" in result:
        moved_views = []
        for view in result["views"]:
            moved = dict(view)
            for key in ["waveform", "lengths", "sample_rate", "available"]:
                moved[key] = moved[key].to(device, non_blocking=True)
            moved_views.append(moved)
        result["views"] = moved_views
    return result


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        warnings.filterwarnings(
            "ignore", message="The number of unique classes is greater than 50%.*"
        )
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        }


def embedding_geometry(embeddings: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(embeddings) < 3 or len(np.unique(labels)) < 2:
        return {}
    normalized = embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12
    )
    centroids = []
    centroid_labels = []
    within = []
    for label in np.unique(labels):
        group = normalized[labels == label]
        centroid = group.mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-12)
        centroids.append(centroid)
        centroid_labels.append(label)
        within.extend((1.0 - group @ centroid).tolist())
    centroid_matrix = np.stack(centroids)
    nearest = np.asarray(centroid_labels)[(normalized @ centroid_matrix.T).argmax(axis=1)]
    centred = embeddings - embeddings.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centred, compute_uv=False)
    energy = np.square(singular)
    probabilities = energy / max(float(energy.sum()), 1e-12)
    result = {
        "embedding_within_class_cosine_distance": float(np.mean(within)),
        "embedding_nearest_centroid_accuracy": float(np.mean(nearest == labels)),
        "embedding_effective_rank": float(
            np.exp(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())
        ),
    }
    try:
        result["embedding_silhouette_cosine"] = float(
            silhouette_score(normalized, labels, metric="cosine")
        )
    except ValueError:
        result["embedding_silhouette_cosine"] = float("nan")
    return result


@torch.no_grad()
def collect_recording_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, float]]:
    model.eval()
    records: dict[int, dict[str, Any]] = {}
    total_loss = 0.0
    total_crops = 0
    total_forward_seconds = 0.0

    for batch in loader:
        batch = move_batch(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        output = model(
            batch["waveform"], batch["lengths"], batch["sample_rate"]
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_forward_seconds += time.perf_counter() - start
        logits = output["logits"]
        total_loss += float(criterion(logits, batch["label"]).item()) * len(logits)
        total_crops += len(logits)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        embeddings = output["embedding"].cpu().numpy()

        for index in range(len(logits)):
            recording_id = int(batch["recording_id"][index])
            item = records.setdefault(
                recording_id,
                {
                    "probabilities": [], "embeddings": [],
                    "label": int(batch["label"][index]),
                    "dataset": batch["dataset"][index],
                    "label_name": batch["label_name"][index],
                    "original_sample_rate": int(batch["original_sample_rate"][index]),
                },
            )
            item["probabilities"].append(probabilities[index])
            item["embeddings"].append(embeddings[index])

    rows = []
    embedding_rows = []
    for recording_id, item in sorted(records.items()):
        probability = np.mean(np.stack(item["probabilities"]), axis=0)
        embedding = np.mean(np.stack(item["embeddings"]), axis=0)
        rows.append(
            {
                "recording_id": recording_id,
                "dataset": item["dataset"],
                "label_id": item["label"],
                "label_name": item["label_name"],
                "predicted_label_id": int(probability.argmax()),
                "confidence": float(probability.max()),
                "original_sample_rate": item["original_sample_rate"],
            }
        )
        embedding_rows.append(embedding)
    predictions = pd.DataFrame(rows)
    embedding_array = np.stack(embedding_rows)
    metrics = classification_metrics(
        predictions["label_id"].to_numpy(),
        predictions["predicted_label_id"].to_numpy(),
    )
    metrics.update(
        loss=total_loss / max(1, total_crops),
        num_recordings=len(predictions),
        num_crops=total_crops,
        frontend_and_model_ms_per_crop=1000.0 * total_forward_seconds / max(1, total_crops),
    )
    metrics.update(
        embedding_geometry(embedding_array, predictions["label_id"].to_numpy())
    )
    return predictions, embedding_array, metrics


def per_dataset_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in predictions.groupby("dataset", sort=True):
        rows.append(
            {
                "dataset": dataset,
                "num_recordings": len(group),
                **classification_metrics(
                    group["label_id"].to_numpy(),
                    group["predicted_label_id"].to_numpy(),
                ),
            }
        )
    return pd.DataFrame(rows)


def per_class_metrics(predictions: pd.DataFrame, label_map: pd.DataFrame) -> pd.DataFrame:
    labels = label_map["label_id"].to_numpy(dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        predictions["label_id"].to_numpy(),
        predictions["predicted_label_id"].to_numpy(),
        labels=labels,
        zero_division=0,
    )
    result = label_map.copy()
    result["precision"] = precision
    result["recall"] = recall
    result["f1"] = f1
    result["support"] = support
    return result
