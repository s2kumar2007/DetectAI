"""Class-balanced sampling (Repeat Factor Sampling) and capped inverse-frequency class loss weights."""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Mapping, Sequence, Tuple

import torch

from ..data.records import ImageRecord


def compute_repeat_factors(
    records: Sequence[ImageRecord],
    class_to_slot: Mapping[str, int],
    t: float = 0.01,
    empty_weight: float = 0.25,
) -> Tuple[List[float], Dict[str, float]]:
    """Tile-level Repeat Factor Sampling (RFS).

    For each class c with tile fraction f_c = n_tiles(c) / N_tiles:
        r_c = max(1.0, sqrt(t / f_c))
    Each non-empty tile gets weight = max_{c in tile} r_c.
    Empty tiles (no mapped boxes) get ``empty_weight`` (default 0.25).
    """
    n_total = max(1, len(records))
    tile_counts: Counter = Counter()
    for r in records:
        present = {l for l in r.labels if l in class_to_slot}
        for l in present:
            tile_counts[l] += 1

    r_cls: Dict[str, float] = {}
    for c, cnt in tile_counts.items():
        f_c = cnt / n_total
        r_cls[c] = max(1.0, math.sqrt(max(t, 1e-8) / max(f_c, 1e-8)))

    weights: List[float] = []
    for r in records:
        present = [l for l in set(r.labels) if l in class_to_slot]
        if not present:
            weights.append(float(empty_weight))
        else:
            weights.append(max(r_cls[l] for l in present))
    return weights, r_cls


def compute_class_loss_weights(
    records: Sequence[ImageRecord],
    class_to_slot: Mapping[str, int],
    num_slots: int,
    cap: float = 5.0,
    power: float = 0.5,
) -> torch.Tensor:
    """Compute per-slot cross-entropy loss weights capped at ``cap``.

    Weight for slot s with N_s boxes is (median_N / N_s)^power, clamped to [1.0 / cap, cap].
    """
    weights = torch.ones(num_slots, dtype=torch.float32)
    counts: Counter = Counter()
    for r in records:
        for l in r.labels:
            if l in class_to_slot:
                counts[class_to_slot[l]] += 1
    if not counts:
        return weights
    vals = sorted(counts.values())
    med = float(vals[len(vals) // 2])
    for s, cnt in counts.items():
        w = (med / max(1.0, float(cnt))) ** float(power)
        weights[s] = float(max(1.0 / max(cap, 1.0), min(float(cap), w)))
    return weights
