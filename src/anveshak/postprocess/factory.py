"""Build a Predictor from a checkpoint and run it over the (unlabelled) test images."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from ..data.build import load_test_images
from ..data.records import ImageRecord
from ..data.slots import SlotMap
from ..training.checkpoint import model_from_checkpoint
from ..utils.hashing import file_sha256
from ..utils.log import get_logger
from ..utils.offline import enforce_offline
from ..utils.seed import seed_everything
from .engine import Predictor
from .unknown import Calibration, load_calibration

log = get_logger("anveshak.infer")


def build_predictor(checkpoint, cfg, calibration: Optional[str] = None, device="cpu", tta: Optional[bool] = None,
                    size: Optional[int] = None) -> Tuple[Predictor, SlotMap, dict]:
    """Deterministic setup (seeds, deterministic kernels, cudnn.benchmark off, no network) + EMA weights from ``checkpoint``."""
    enforce_offline()
    seed_everything(int(cfg.seed), deterministic=True)
    model, meta, slotmap = model_from_checkpoint(checkpoint, device, use="ema")
    size = int(size or meta["config"]["input"]["size"])
    calib = load_calibration(cfg, calibration)
    pred = Predictor(model, slotmap.slot_to_name(), size, cfg.infer, calib, device, tta)
    log.info("predictor: ckpt=%s stage=%s size=%d tta=%s classes=%d calib=%s", checkpoint, meta.get("stage"), size, pred.tta,
             len(slotmap), calib.to_dict())
    return pred, slotmap, meta


def predict_test_set(pred: Predictor, records, progress: bool = False) -> pd.DataFrame:
    """Predict EVERY record; asserts that each test image id was processed (rows may legitimately be empty)."""
    df = pred.predict_records(records, progress)
    ids = {r.image_id for r in records}
    stray = set(df["image_id"]) - ids
    if stray:
        raise RuntimeError(f"predictions for unknown image ids: {sorted(stray)[:5]}")
    return df
