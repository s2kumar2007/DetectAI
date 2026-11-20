"""Frozen-teacher pseudo-labels for OLD classes so that old objects in new-class images are never treated as background."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torchvision.ops import batched_nms, box_iou


@torch.no_grad()
def pseudo_labels(raw: Dict[str, torch.Tensor], gt_boxes: torch.Tensor, cfg) -> Tuple[torch.Tensor, torch.Tensor]:
    """High-confidence teacher detections (score = sqrt(S_obj * p_class) >= score_thr), class-aware NMS, that do NOT
    overlap (IoU >= overlap_iou) any real GT box of the image. Returns (boxes (M,4), slot labels (M,))."""
    dev = raw["boxes"].device
    empty = (torch.zeros(0, 4, device=dev), torch.zeros(0, dtype=torch.long, device=dev))
    if raw["boxes"].shape[0] == 0:
        return empty
    pmax, cls = raw["probs"].max(1)
    score = torch.sqrt(raw["obj"].clamp(min=0) * pmax)
    boxes = raw["boxes"]
    ok = (score >= float(cfg.score_thr)) & ((boxes[:, 2] - boxes[:, 0]) > 2) & ((boxes[:, 3] - boxes[:, 1]) > 2)
    b, s, c = boxes[ok], score[ok], cls[ok]
    if b.shape[0] == 0:
        return empty
    keep = batched_nms(b, s, c, float(cfg.nms_iou))[: int(cfg.max_per_img)]
    b, c = b[keep], c[keep]
    if gt_boxes.shape[0] and b.shape[0]:
        b_ok = box_iou(b, gt_boxes.to(dev).float()).max(1).values < float(cfg.overlap_iou)
        b, c = b[b_ok], c[b_ok]
    return b, c
