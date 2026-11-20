"""Balanced exemplar replay buffer + dataset mixing."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Sequence

import numpy as np
from torch.utils.data import Dataset

from ..data.records import ImageRecord


def build_replay_buffer(records: Sequence[ImageRecord], classes: Sequence[str], per_class: int, max_images: int,
                        seed: int = 0) -> List[ImageRecord]:
    """Pick images so that every class in ``classes`` has about ``per_class`` labelled instances (rarest classes first,
    seeded shuffles, capped at ``max_images``). Only the labels of ``classes`` are kept; other labels are dropped
    (those objects become unlabelled - the teacher pseudo-labels cover them when they are old classes)."""
    rng = np.random.default_rng(seed)
    cls_set = set(classes)
    by_class: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(records):
        for c in set(r.labels) & cls_set:
            by_class[c].append(i)
    chosen: List[int] = []
    chosen_set = set()
    counts: Counter = Counter()

    def add(i: int) -> None:
        chosen.append(i)
        chosen_set.add(i)
        counts.update(l for l in records[i].labels if l in cls_set)

    for c in sorted(by_class, key=lambda k: (len(by_class[k]), k)):
        cand = list(by_class[c])
        rng.shuffle(cand)
        for i in cand:
            if counts[c] >= per_class or len(chosen) >= max_images:
                break
            if i not in chosen_set:
                add(i)
    return [records[i].relabel({l: (l if l in cls_set else None) for l in set(records[i].labels)}) for i in sorted(chosen)]


class MixedDataset(Dataset):
    """Concatenation of datasets that forwards ``set_epoch``."""

    def __init__(self, datasets: Sequence[Dataset]):
        self.datasets = list(datasets)
        self.cum = np.cumsum([len(d) for d in self.datasets])

    def __len__(self) -> int:
        return int(self.cum[-1]) if len(self.cum) else 0

    def set_epoch(self, e: int) -> None:
        for d in self.datasets:
            if hasattr(d, "set_epoch"):
                d.set_epoch(e)

    def __getitem__(self, i: int):
        k = int(np.searchsorted(self.cum, i, side="right"))
        base = 0 if k == 0 else int(self.cum[k - 1])
        return self.datasets[k][i - base]

    def weights(self, fractions: Sequence[float]) -> List[float]:
        """Per-sample weights so dataset k contributes ``fractions[k]`` of each batch (in expectation)."""
        w: List[float] = []
        for d, f in zip(self.datasets, fractions):
            w.extend([f / max(len(d), 1)] * len(d))
        return w
