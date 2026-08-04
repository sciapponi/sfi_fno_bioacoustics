from __future__ import annotations

from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class DatasetBalancedPKBatchSampler(Sampler[list[int]]):
    """Build P × K batches balanced over datasets and classes.

    For each class slot, a dataset is selected uniformly and then one class is
    selected uniformly inside that dataset. K recordings are sampled from that
    class. Sampling with replacement is allowed for very small classes; each
    repeated index still receives a separately randomized waveform crop.

    The same sampler is used for cross-entropy and focal/contrastive runs so the
    objective comparison does not also change the batch composition.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        samples_per_class: int,
        seed: int,
        num_samples: int | None = None,
    ) -> None:
        if getattr(dataset, "eval_crops", 1) != 1:
            raise ValueError("The P x K sampler requires training eval_crops=1.")
        if samples_per_class < 2:
            raise ValueError("samples_per_class must be at least 2.")
        if batch_size % samples_per_class:
            raise ValueError("batch_size must be divisible by samples_per_class.")

        rows = dataset.rows.reset_index(drop=True)
        nested: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, row in rows.iterrows():
            nested[str(row["dataset"])][int(row["label_id"])].append(int(index))
        if not nested:
            raise ValueError("Cannot sample an empty training dataset.")

        self.datasets = sorted(nested)
        self.classes = {name: sorted(nested[name]) for name in self.datasets}
        self.indices = {
            name: {
                label: np.asarray(nested[name][label], dtype=np.int64)
                for label in self.classes[name]
            }
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
            for _slot in range(self.classes_per_batch):
                for _attempt in range(100):
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
