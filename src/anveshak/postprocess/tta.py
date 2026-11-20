"""Deterministic 4-view test-time augmentation on the square letterboxed tile: original, h-flip, v-flip, rot90."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

VIEWS = ("orig", "hflip", "vflip", "rot90")


def apply_view(img: torch.Tensor, view: str) -> torch.Tensor:
    """img: CHW (square). rot90 = torch.rot90(k=1, dims=(H, W)) (counter-clockwise)."""
    if view == "orig":
        return img
    if view == "hflip":
        return torch.flip(img, dims=(2,))
    if view == "vflip":
        return torch.flip(img, dims=(1,))
    if view == "rot90":
        return torch.rot90(img, 1, dims=(1, 2))
    raise ValueError(view)


def unmap_boxes(boxes: np.ndarray, view: str, size: int) -> np.ndarray:
    """Map xyxy boxes detected in the transformed view back to the original tile frame (size x size)."""
    b = np.asarray(boxes, np.float32).reshape(-1, 4)
    if len(b) == 0 or view == "orig":
        return b.copy()
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    if view == "hflip":
        return np.stack([size - x2, y1, size - x1, y2], 1)
    if view == "vflip":
        return np.stack([x1, size - y2, x2, size - y1], 1)
    if view == "rot90":  # forward map (x, y) -> (y, S - x)
        return np.stack([size - y2, x1, size - y1, x2], 1)
    raise ValueError(view)
