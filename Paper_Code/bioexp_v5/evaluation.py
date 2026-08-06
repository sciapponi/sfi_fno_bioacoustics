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


def plot_tsne_by_class(
    embeddings: np.ndarray,
    predictions: pd.DataFrame,
    output_path,
    seed: int = 42,
    maximum_points: int = 5000,
    perplexity: float = 30.0,
) -> pd.DataFrame:
    """Save one test-embedding t-SNE plot coloured by class.

    This is intended for fold 1 diagnostics only. The plot deliberately omits a
    legend when many classes are present; the returned colour table is saved as a
    CSV by the runner.
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    values = np.asarray(embeddings, dtype=np.float32)
    frame = predictions.reset_index(drop=True).copy()
    if len(values) != len(frame):
        raise ValueError("Embedding and prediction row counts do not match.")
    if len(values) < 3:
        raise ValueError("At least three recordings are required for t-SNE.")

    rng = np.random.default_rng(int(seed))
    if len(values) > int(maximum_points):
        selected: list[int] = []
        groups = list(frame.groupby("label_id", sort=True).groups.items())
        quota = max(1, int(maximum_points) // max(1, len(groups)))
        for _, indices in groups:
            candidates = np.asarray(list(indices), dtype=np.int64)
            take = min(len(candidates), quota)
            selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
        if len(selected) < int(maximum_points):
            remaining = np.setdiff1d(np.arange(len(values)), np.asarray(selected))
            extra = min(int(maximum_points) - len(selected), len(remaining))
            if extra:
                selected.extend(rng.choice(remaining, size=extra, replace=False).tolist())
        selected = sorted(set(selected))[: int(maximum_points)]
        values = values[selected]
        frame = frame.iloc[selected].reset_index(drop=True)

    use_perplexity = min(float(perplexity), max(2.0, (len(values) - 1) / 3.0))
    projection = TSNE(
        n_components=2,
        perplexity=use_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=int(seed),
        metric="cosine",
    ).fit_transform(values)

    labels = sorted(frame["label_id"].unique().tolist())
    colour_index = {label: index for index, label in enumerate(labels)}
    colours = np.asarray([colour_index[value] for value in frame["label_id"]])

    figure, axis = plt.subplots(figsize=(10, 8))
    axis.scatter(
        projection[:, 0], projection[:, 1], c=colours,
        cmap="gist_ncar", s=12, alpha=0.75, linewidths=0,
    )
    axis.set_title("Test embeddings by class (fold 1)")
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    axis.grid(alpha=0.15)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    lookup = (
        frame[["label_id", "label_name"]]
        .drop_duplicates()
        .sort_values("label_id")
        .reset_index(drop=True)
    )
    lookup["colour_index"] = lookup["label_id"].map(colour_index)
    return lookup
