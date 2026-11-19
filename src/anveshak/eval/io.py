"""CSV <-> DataFrame helpers for predictions and ground truth."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ..data.records import ImageRecord
from .conflict import COLS, UNKNOWN

GT_COLS = [c for c in COLS if c != "score"]


def read_boxes_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"image_id": str, "class_name": str}, encoding="utf-8")
    missing = [c for c in GT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


def records_to_gt(records: Iterable[ImageRecord], rename: Optional[dict] = None) -> pd.DataFrame:
    """Ground-truth DataFrame from records. ``rename`` maps class -> new name (e.g. held-out -> 'unknown')."""
    rows = []
    for r in records:
        for b, l in zip(r.boxes, r.labels):
            rows.append([r.image_id, (rename or {}).get(l, l), *[float(v) for v in b]])
    return pd.DataFrame(rows, columns=GT_COLS)


def rows_to_df(rows: Sequence[Sequence]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=COLS)
