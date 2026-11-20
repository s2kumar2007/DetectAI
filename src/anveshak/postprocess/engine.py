"""Inference engine: letterbox -> (4-view TTA) -> known / unknown candidates -> WBF -> class-aware NMS ->
conflict-aware suppression -> unknown cap -> CSV rows. Fully deterministic (no random TTA)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torchvision.ops import batched_nms, nms

from ..data.dataset import letterbox
from ..data.records import ImageRecord, load_image
from ..eval.conflict import COLS, UNKNOWN, conflict_keep_mask
from .decode import known_detections
from .tta import VIEWS, apply_view, unmap_boxes
from .unknown import Calibration, unknown_detections
from .wbf import weighted_boxes_fusion

Raw = Dict[str, torch.Tensor]


class Predictor:
    def __init__(self, model, slot_to_name: Dict[int, str], size: int, infer_cfg, calib: Calibration, device="cpu", tta: Optional[bool] = None):
        self.model, self.slot_to_name, self.size = model.to(device).eval(), dict(slot_to_name), int(size)
        self.cfg, self.calib, self.device = infer_cfg, calib, device
        self.tta = bool(infer_cfg.tta if tta is None else tta)
        self.views = VIEWS if self.tta else ("orig",)
        self.topk_classes = int(infer_cfg.get("topk_classes", 2))

    # ------------------------------------------------------------------ model pass (cacheable)
    @torch.no_grad()
    def compute_raws(self, img: torch.Tensor) -> List[Raw]:
        """Raw per-RoI outputs for every view of one letterboxed uint8 CHW tile (moved to CPU)."""
        imgs = [apply_view(img, v).to(self.device) for v in self.views]
        raws = self.model.predict_raw(imgs)
        return [{k: t.detach().cpu() for k, t in r.items()} for r in raws]

    # ------------------------------------------------------------------ candidates
    def known_part(self, raws: Sequence[Raw]) -> Dict[str, np.ndarray]:
        c = self.cfg
        per = []
        for v, raw in zip(self.views, raws):
            d = known_detections(raw, float(c.min_known_score), self.topk_classes, float(c.known_nms_iou), int(c.topk_known))
            per.append({"boxes": unmap_boxes(d["boxes"], v, self.size), "scores": d["scores"], "slots": d["slots"]})
        if len(per) == 1:
            return per[0]
        b, s, l = weighted_boxes_fusion([p["boxes"] for p in per], [p["scores"] for p in per], [p["slots"] for p in per],
                                        len(per), float(c.wbf_iou))
        if len(b):  # final class-aware NMS
            keep = batched_nms(torch.from_numpy(b), torch.from_numpy(s), torch.from_numpy(l), float(c.known_nms_iou))[: int(c.topk_known)]
            b, s, l = b[keep.numpy()], s[keep.numpy()], l[keep.numpy()]
        return {"boxes": b, "scores": s, "slots": l}

    def unknown_part(self, raws: Sequence[Raw], calib: Optional[Calibration] = None) -> Dict[str, np.ndarray]:
        calib, c = calib or self.calib, self.cfg
        per = []
        for v, raw in zip(self.views, raws):
            d = unknown_detections(raw, calib, float(c.unk_nms_iou))
            per.append({"boxes": unmap_boxes(d["boxes"], v, self.size), "scores": d["scores"]})
        if len(per) == 1:
            return per[0]
        b, s, _ = weighted_boxes_fusion([p["boxes"] for p in per], [p["scores"] for p in per],
                                        [np.zeros(len(p["scores"]), np.int64) for p in per], len(per), float(c.wbf_iou))
        if len(b):
            keep = nms(torch.from_numpy(b), torch.from_numpy(s), float(c.unk_nms_iou)).numpy()
            b, s = b[keep], s[keep]
        return {"boxes": b, "scores": s}

    # ------------------------------------------------------------------ finalisation
    def finalize(self, image_id: str, known: Dict[str, np.ndarray], unknown: Dict[str, np.ndarray], scale: float,
                 width: int, height: int, calib: Optional[Calibration] = None) -> List[list]:
        """Rows [image_id, class_name, score, xmin, ymin, xmax, ymax] after rounding, conflict suppression and caps."""
        calib, c = calib or self.calib, self.cfg
        rows: List[list] = []
        for part, names in ((known, None), (unknown, UNKNOWN)):
            if len(part["scores"]) == 0:
                continue
            b = part["boxes"].astype(np.float64) / scale
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, width)
            b[:, [1, 3]] = b[:, [1, 3]].clip(0, height)
            b = np.round(b, 2)
            s = np.clip(np.round(part["scores"].astype(np.float64), 6), 0.0, 1.0)
            for i in range(len(s)):
                if b[i, 2] <= b[i, 0] or b[i, 3] <= b[i, 1]:
                    continue
                name = names if names is not None else self.slot_to_name.get(int(part["slots"][i]))
                if name is None:
                    continue
                rows.append([image_id, name, float(s[i]), *map(float, b[i])])
        if not rows:
            return []
        rows.sort(key=lambda r: -r[2])  # stable: known rows precede unknown on ties
        arr = np.asarray([r[3:] for r in rows])
        sc = np.asarray([r[2] for r in rows])
        is_unk = np.asarray([r[1] == UNKNOWN for r in rows])
        keep = conflict_keep_mask(arr, sc, is_unk, np.arange(len(rows)), float(c.conflict_iou))
        rows = [r for r, k in zip(rows, keep) if k]
        n_unk = 0
        capped = []
        for r in rows:  # descending score order: keep the best `unk_cap` unknowns
            if r[1] == UNKNOWN:
                n_unk += 1
                if n_unk > calib.unk_cap:
                    continue
            capped.append(r)
        seen, out = set(), []
        for r in capped:
            key = tuple(r)
            if key not in seen:
                seen.add(key)
                out.append(r)
        return out[: int(c.max_rows_per_image)]

    # ------------------------------------------------------------------ public API
    def load_tile(self, rec: ImageRecord) -> Tuple[torch.Tensor, float]:
        img, _, s = letterbox(load_image(rec.path), np.zeros((0, 4), np.float32), self.size)
        return torch.from_numpy(img).permute(2, 0, 1).contiguous(), s

    def predict_record(self, rec: ImageRecord) -> List[list]:
        tile, s = self.load_tile(rec)
        raws = self.compute_raws(tile)
        return self.finalize(rec.image_id, self.known_part(raws), self.unknown_part(raws), s, rec.width, rec.height)

    def predict_records(self, records: Sequence[ImageRecord], progress: bool = False) -> pd.DataFrame:
        rows: List[list] = []
        it = records
        if progress:
            from tqdm import tqdm

            it = tqdm(records, desc="infer")
        for rec in it:
            rows.extend(self.predict_record(rec))
        return pd.DataFrame(rows, columns=COLS)
