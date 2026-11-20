"""Torch box utilities (xyxy)."""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
from torch import Tensor

_BBOX_CLIP = math.log(1000.0 / 16)
DEFAULT_WEIGHTS: Tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0)


def encode_boxes(proposals: Tensor, gt: Tensor, weights: Sequence[float] = DEFAULT_WEIGHTS) -> Tensor:
    wx, wy, ww, wh = weights
    pw = (proposals[:, 2] - proposals[:, 0]).clamp(min=1e-3)
    ph = (proposals[:, 3] - proposals[:, 1]).clamp(min=1e-3)
    px, py = proposals[:, 0] + 0.5 * pw, proposals[:, 1] + 0.5 * ph
    gw = (gt[:, 2] - gt[:, 0]).clamp(min=1e-3)
    gh = (gt[:, 3] - gt[:, 1]).clamp(min=1e-3)
    gx, gy = gt[:, 0] + 0.5 * gw, gt[:, 1] + 0.5 * gh
    return torch.stack([wx * (gx - px) / pw, wy * (gy - py) / ph, ww * torch.log(gw / pw), wh * torch.log(gh / ph)], dim=1)


def decode_boxes(deltas: Tensor, boxes: Tensor, weights: Sequence[float] = DEFAULT_WEIGHTS) -> Tensor:
    wx, wy, ww, wh = weights
    w = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-3)
    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-3)
    cx, cy = boxes[:, 0] + 0.5 * w, boxes[:, 1] + 0.5 * h
    dx, dy = deltas[:, 0] / wx, deltas[:, 1] / wy
    dw = (deltas[:, 2] / ww).clamp(max=_BBOX_CLIP)
    dh = (deltas[:, 3] / wh).clamp(max=_BBOX_CLIP)
    pcx, pcy = dx * w + cx, dy * h + cy
    pw, ph = torch.exp(dw) * w, torch.exp(dh) * h
    return torch.stack([pcx - 0.5 * pw, pcy - 0.5 * ph, pcx + 0.5 * pw, pcy + 0.5 * ph], dim=1)


def clip_boxes_to(boxes: Tensor, size_hw: Tuple[int, int]) -> Tensor:
    h, w = size_hw
    return torch.stack(
        [boxes[:, 0].clamp(0, w), boxes[:, 1].clamp(0, h), boxes[:, 2].clamp(0, w), boxes[:, 3].clamp(0, h)], dim=1
    )


def elementwise_iou(a: Tensor, b: Tensor) -> Tensor:
    """IoU between matching rows of a and b, both (N,4)."""
    lt = torch.maximum(a[:, :2], b[:, :2])
    rb = torch.minimum(a[:, 2:], b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=1)
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(dim=1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(dim=1)
    return inter / (area_a + area_b - inter).clamp(min=1e-9)
