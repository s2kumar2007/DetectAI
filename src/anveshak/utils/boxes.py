"""Small numpy box helpers shared by data, eval and postprocess (xyxy convention)."""
from __future__ import annotations

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy arrays -> (N,M) float64."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def clip_boxes(boxes: np.ndarray, width: float, height: float) -> np.ndarray:
    out = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, width)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, height)
    return out


def nms_np(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS; returns kept indices sorted by descending score (stable tie-break by index)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((np.arange(len(scores)), -scores))
    keep = []
    suppressed = np.zeros(len(scores), dtype=bool)
    ious = iou_matrix(boxes, boxes) if len(boxes) else np.zeros((0, 0))
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed |= ious[i] > iou_thr
        suppressed[i] = True
    return np.asarray(keep, dtype=np.int64)
