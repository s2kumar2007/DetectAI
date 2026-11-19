"""Known/unknown conflict rule (applied BEFORE scoring).

If a known prediction and an unknown prediction overlap at IoU >= 0.5 (same image), only the higher-scoring one
survives; ties go to the earlier row. Known-known and unknown-unknown overlaps are untouched.

Interpretation (ASSUMPTIONS.md A12): the rule is evaluated pairwise on the ORIGINAL prediction set - a prediction is
removed if it loses to ANY overlapping opposite-type prediction, even if that opponent is itself removed by a third box.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from ..utils.boxes import iou_matrix

UNKNOWN = "unknown"
COLS = ["image_id", "class_name", "score", "xmin", "ymin", "xmax", "ymax"]
_EPS = 1e-12


def conflict_keep_mask(
    boxes: np.ndarray, scores: np.ndarray, is_unknown: np.ndarray, order: np.ndarray, iou_thr: float = 0.5
) -> np.ndarray:
    """Boolean keep-mask for one image. ``order`` = global row order (smaller = earlier)."""
    keep = np.ones(len(boxes), dtype=bool)
    k_idx = np.flatnonzero(~is_unknown)
    u_idx = np.flatnonzero(is_unknown)
    if len(k_idx) == 0 or len(u_idx) == 0:
        return keep
    iou = iou_matrix(boxes[k_idx], boxes[u_idx])
    conflict = iou >= iou_thr - _EPS
    if not conflict.any():
        return keep
    sk, su = scores[k_idx][:, None], scores[u_idx][None, :]
    rk, ru = order[k_idx][:, None], order[u_idx][None, :]
    unk_wins = (su > sk) | ((su == sk) & (ru < rk))
    keep[k_idx[(conflict & unk_wins).any(axis=1)]] = False
    keep[u_idx[(conflict & ~unk_wins).any(axis=0)]] = False
    return keep


def apply_conflict_rule(preds: pd.DataFrame, iou_thr: float = 0.5, unknown_name: str = UNKNOWN) -> pd.DataFrame:
    """Return the predictions that survive the conflict rule, in original row order."""
    if len(preds) == 0:
        return preds.copy()
    df = preds.reset_index(drop=True)
    keep = np.ones(len(df), dtype=bool)
    order_all = np.arange(len(df))
    is_unk_all = (df["class_name"].to_numpy() == unknown_name)
    boxes_all = df[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=np.float64)
    scores_all = df["score"].to_numpy(dtype=np.float64)
    for _, idx in df.groupby("image_id", sort=False).indices.items():
        idx = np.asarray(idx)
        k = conflict_keep_mask(boxes_all[idx], scores_all[idx], is_unk_all[idx], order_all[idx], iou_thr)
        keep[idx] = k
    return df[keep].reset_index(drop=True)


def split_known_unknown(preds: pd.DataFrame, unknown_name: str = UNKNOWN) -> Tuple[pd.DataFrame, pd.DataFrame]:
    m = preds["class_name"] == unknown_name
    return preds[~m], preds[m]
