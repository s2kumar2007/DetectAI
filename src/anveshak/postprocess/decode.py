"""Turn raw per-RoI model outputs into known-class detections (and, in ``unknown.py``, unknown detections)."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torchvision.ops import batched_nms


def known_detections(
    raw: Dict[str, torch.Tensor],
    min_score: float = 0.01,
    topk_classes: int = 2,
    nms_iou: float = 0.5,
    max_det: int = 300,
) -> Dict[str, np.ndarray]:
    """Known detections of one image: score = sqrt(S_obj * p_class); top-k classes per RoI; class-aware NMS.

    Returns numpy dict ``boxes (M,4)``, ``scores (M,)``, ``slots (M,)`` sorted by descending score.
    """
    probs, obj, boxes = raw["probs"].float().cpu(), raw["obj"].float().cpu(), raw["boxes"].float().cpu()
    n, S = probs.shape
    empty = {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros(0, np.float32), "slots": np.zeros(0, np.int64)}
    if n == 0:
        return empty
    sc = torch.sqrt(obj[:, None].clamp(min=0) * probs)
    k = min(int(topk_classes), S)
    top_s, top_c = sc.topk(k, dim=1)
    b = boxes[:, None, :].expand(n, k, 4).reshape(-1, 4)
    s, c = top_s.reshape(-1), top_c.reshape(-1)
    keep = s >= min_score
    b, s, c = b[keep], s[keep], c[keep]
    if b.shape[0] == 0:
        return empty
    wh_ok = ((b[:, 2] - b[:, 0]) > 0.5) & ((b[:, 3] - b[:, 1]) > 0.5)
    b, s, c = b[wh_ok], s[wh_ok], c[wh_ok]
    if b.shape[0] == 0:
        return empty
    idx = batched_nms(b, s, c, nms_iou)[: int(max_det)]
    return {"boxes": b[idx].numpy(), "scores": s[idx].numpy(), "slots": c[idx].numpy()}
