"""In-house Weighted Boxes Fusion (deterministic, numpy only).

Boxes of the same label are clustered greedily in descending score order (ties: input order). A box joins the best
cluster (highest IoU with the cluster's fused box, must exceed ``iou_thr``) that has no box from the same view yet;
otherwise it opens a new cluster. Fused box = score-weighted mean of coordinates; fused score = mean score of the
members x min(#members, #views) / #views (so boxes seen by few views are down-weighted).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..utils.boxes import iou_matrix


def weighted_boxes_fusion(
    boxes: Sequence[np.ndarray],
    scores: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    n_views: int,
    iou_thr: float = 0.55,
    skip_thr: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``boxes[v]`` (N_v,4), ``scores[v]``, ``labels[v]`` are the detections of view v. Returns (boxes, scores, labels)."""
    rows = []  # (score, order, view, label, box)
    order = 0
    for v, (b, s, l) in enumerate(zip(boxes, scores, labels)):
        for bb, ss, ll in zip(np.asarray(b, np.float64).reshape(-1, 4), np.asarray(s, np.float64), np.asarray(l)):
            if ss >= skip_thr:
                rows.append((float(ss), order, v, int(ll), bb))
            order += 1
    if not rows:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    out_b: List[np.ndarray] = []
    out_s: List[float] = []
    out_l: List[int] = []
    for lab in sorted({r[3] for r in rows}):
        items = sorted((r for r in rows if r[3] == lab), key=lambda r: (-r[0], r[1]))
        clusters: List[dict] = []
        fused = np.zeros((0, 4))
        for score, _, view, _, box in items:
            best, best_iou = -1, iou_thr
            if clusters:
                ious = iou_matrix(box[None], fused)[0]
                for ci in np.argsort(-ious, kind="stable"):
                    if ious[ci] <= iou_thr:
                        break
                    if view not in clusters[ci]["views"]:
                        best, best_iou = int(ci), ious[ci]
                        break
            if best < 0:
                clusters.append({"boxes": [box], "scores": [score], "views": {view}})
                fused = np.vstack([fused, box[None]]) if len(fused) else box[None].copy()
            else:
                c = clusters[best]
                c["boxes"].append(box)
                c["scores"].append(score)
                c["views"].add(view)
                w = np.asarray(c["scores"])
                fused[best] = (np.asarray(c["boxes"]) * w[:, None]).sum(0) / w.sum()
        for ci, c in enumerate(clusters):
            out_b.append(fused[ci])
            out_s.append(float(np.mean(c["scores"])) * min(len(c["scores"]), n_views) / n_views)
            out_l.append(lab)
    idx = np.lexsort((np.arange(len(out_s)), -np.asarray(out_s)))
    return (np.asarray(out_b, np.float32)[idx], np.asarray(out_s, np.float32)[idx], np.asarray(out_l, np.int64)[idx])
