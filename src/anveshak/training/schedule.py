"""LR schedule: linear warmup (iterations) x cosine decay (over progress, which can be time-aware)."""
from __future__ import annotations

import math


def lr_factor(iteration: int, warmup_iters: int, progress: float, min_frac: float) -> float:
    warm = min(1.0, (iteration + 1) / max(1, warmup_iters))
    progress = min(max(progress, 0.0), 1.0)
    cos = min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return warm * cos
