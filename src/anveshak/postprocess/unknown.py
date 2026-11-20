"""Unknown gate: unknown_score = S_obj * (1 - S_known_max) * alpha.

``S_obj``      RoI objectness / IoU-quality in [0,1].
``S_known_max`` max cosine similarity to the ACTIVE prototypes, calibrated as clamp(cos_max / cos_ref, 0, 1) where
               ``cos_ref`` is the similarity at/above which a RoI counts as "fully known" (set by calibrate_unknown.py).
A RoI is an unknown candidate when unknown_score > tau_unk; the conflict rule against known boxes is applied later.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import json

import numpy as np
import torch
from torchvision.ops import nms


@dataclass
class Calibration:
    alpha: float = 1.0
    tau_unk: float = 0.3
    unk_cap: int = 20
    cos_ref: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


def load_calibration(cfg, path: Optional[str | Path] = None) -> Calibration:
    """Config defaults (``calibration:``), overridden by a calibration.json written by calibrate_unknown.py."""
    d = {k: cfg.calibration[k] for k in ("alpha", "tau_unk", "unk_cap", "cos_ref")}
    if path:
        j = json.loads(Path(path).read_text(encoding="utf-8"))
        for k in d:
            if k in j:
                d[k] = j[k]
    return Calibration(float(d["alpha"]), float(d["tau_unk"]), int(d["unk_cap"]), float(d["cos_ref"]))


def unknown_scores(obj: torch.Tensor, cos_max: torch.Tensor, calib: Calibration) -> torch.Tensor:
    s_known = (cos_max.float() / max(calib.cos_ref, 1e-6)).clamp(0.0, 1.0)
    return (obj.float() * (1.0 - s_known) * calib.alpha).clamp(0.0, 1.0)


def unknown_detections(raw: Dict[str, torch.Tensor], calib: Calibration, nms_iou: float = 0.5) -> Dict[str, np.ndarray]:
    """Unknown candidates of one image (before the conflict rule / cap): boxes (M,4), scores (M,), NMS'd."""
    obj, cos, boxes = raw["obj"].float().cpu(), raw["cos_max"].float().cpu(), raw["boxes"].float().cpu()
    empty = {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros(0, np.float32)}
    if boxes.shape[0] == 0:
        return empty
    sc = unknown_scores(obj, cos, calib)
    ok = (sc > calib.tau_unk) & ((boxes[:, 2] - boxes[:, 0]) > 0.5) & ((boxes[:, 3] - boxes[:, 1]) > 0.5)
    b, s = boxes[ok], sc[ok]
    if b.shape[0] == 0:
        return empty
    keep = nms(b, s, nms_iou)
    return {"boxes": b[keep].numpy(), "scores": s[keep].numpy()}
