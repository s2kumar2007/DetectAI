"""Run a model over annotated records and score known classes (used for validation and retention reports)."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data.dataset import DetDataset, collate
from ..data.records import ImageRecord
from ..eval.io import records_to_gt, rows_to_df
from ..eval.local_scorer import map_known
from ..postprocess.decode import known_detections


@torch.no_grad()
def predict_known(model, records: Sequence[ImageRecord], slot_to_name: Dict[int, str], size: int, device,
                  batch_size: int = 4, min_score: float = 0.01, topk_classes: int = 2, nms_iou: float = 0.5,
                  max_det: int = 300) -> pd.DataFrame:
    """Single-view known-class predictions in ORIGINAL image pixel coordinates."""
    ds = DetDataset(records, {}, size, train=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    was = model.training
    model.eval()
    rows: List[list] = []
    for images, targets in dl:
        raws = model.predict_raw([im.to(device) for im in images])
        for raw, t in zip(raws, targets):
            det = known_detections(raw, min_score, topk_classes, nms_iou, max_det)
            w, h = t["orig_size"]
            b = det["boxes"] / t["scale"]
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
            b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
            for bb, s, c in zip(b, det["scores"], det["slots"]):
                if int(c) in slot_to_name and bb[2] > bb[0] and bb[3] > bb[1]:
                    rows.append([t["image_id"], slot_to_name[int(c)], float(s), *map(float, bb)])
    model.train(was)
    return rows_to_df(rows)


def known_map(model, records: Sequence[ImageRecord], slot_to_name: Dict[int, str], classes: Iterable[str], size: int,
              device, **kw) -> Tuple[float, Dict[str, float], List[str]]:
    """mAP (VOC all-point, IoU 0.5) over ``classes`` on ``records``; GT of other classes is ignored."""
    classes = sorted(set(classes))
    preds = predict_known(model, records, {s: n for s, n in slot_to_name.items() if n in set(classes)}, size, device, **kw)
    gts = records_to_gt(records)
    gts = gts[gts["class_name"].isin(classes)]
    return map_known(preds, gts, classes)
