"""Local re-implementation of the competition metric (OE-Score).

* IoU 0.5 matching; each prediction matches at most one GT; predictions processed in descending score order.
* mAP_K = mean over known classes (that have GT) of VOC all-point AP over all tiles. Known predictions that land on
  unknown GT are false positives. Known classes without GT are skipped and listed.
* U-Rec = matched unknown GT / unknown GT ; U-Prec = matched unknown preds / unknown preds ; U-F1 = harmonic mean.
* Conflict rule is applied BEFORE scoring (see :mod:`anveshak.eval.conflict`).
* OE-Score = 100 * 2 * mAP_K * U-F1 / (mAP_K + U-F1).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Collection, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.boxes import iou_matrix
from .conflict import COLS, UNKNOWN, apply_conflict_rule

BOX = ["xmin", "ymin", "xmax", "ymax"]


@dataclass
class ScoreResult:
    map_k: float
    ap_per_class: Dict[str, float]
    skipped_classes: List[str]
    u_rec: float
    u_prec: float
    u_f1: float
    oe_score: float
    n_unknown_gt: int = 0
    n_unknown_pred: int = 0
    n_unknown_matched_gt: int = 0
    n_unknown_matched_pred: int = 0
    n_pred_before_conflict: int = 0
    n_pred_after_conflict: int = 0
    ignored_pred_classes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"mAP_K={self.map_k:.4f}  U-Rec={self.u_rec:.4f}  U-Prec={self.u_prec:.4f}  U-F1={self.u_f1:.4f}  "
            f"OE={self.oe_score:.2f}  (classes scored={len(self.ap_per_class)}, skipped={len(self.skipped_classes)})"
        )


def voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """VOC all-point interpolated AP."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def oe_score(map_k: float, u_f1: float) -> float:
    denom = map_k + u_f1
    return 0.0 if denom <= 0 else 100.0 * 2.0 * map_k * u_f1 / denom


def _group_boxes(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for img, sub in df.groupby("image_id", sort=False):
        out[str(img)] = sub[BOX].to_numpy(dtype=np.float64)
    return out


def _match(pred_df: pd.DataFrame, gt_by_img: Dict[str, np.ndarray], iou_thr: float, mode: str) -> np.ndarray:
    """Return tp flags aligned with ``pred_df`` rows after sorting by (-score, row order)."""
    n = len(pred_df)
    scores = pred_df["score"].to_numpy(dtype=np.float64)
    order = np.lexsort((np.arange(n), -scores))
    imgs = pred_df["image_id"].astype(str).to_numpy()
    boxes = pred_df[BOX].to_numpy(dtype=np.float64)
    taken = {k: np.zeros(len(v), dtype=bool) for k, v in gt_by_img.items()}
    tp = np.zeros(n, dtype=bool)
    for i in order:
        g = gt_by_img.get(imgs[i])
        if g is None or len(g) == 0:
            continue
        ious = iou_matrix(boxes[i : i + 1], g)[0]
        if mode == "voc":  # best-IoU GT; if already taken -> false positive
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr - 1e-12 and not taken[imgs[i]][j]:
                taken[imgs[i]][j] = True
                tp[i] = True
        else:  # greedy: best IoU among still-free GT
            ious = np.where(taken[imgs[i]], -1.0, ious)
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr - 1e-12:
                taken[imgs[i]][j] = True
                tp[i] = True
    return tp, order  # type: ignore[return-value]


def class_ap(pred_df: pd.DataFrame, gt_df: pd.DataFrame, iou_thr: float = 0.5, mode: str = "voc") -> float:
    n_gt = len(gt_df)
    if n_gt == 0:
        raise ValueError("class_ap needs at least one GT")
    if len(pred_df) == 0:
        return 0.0
    tp, order = _match(pred_df, _group_boxes(gt_df), iou_thr, mode)
    tp_sorted = tp[order].astype(np.float64)
    ctp = np.cumsum(tp_sorted)
    cfp = np.cumsum(1.0 - tp_sorted)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    return voc_ap(recall, precision)


def _clean(df: Optional[pd.DataFrame], need_score: bool) -> pd.DataFrame:
    if df is None or len(df) == 0:
        cols = COLS if need_score else [c for c in COLS if c != "score"]
        return pd.DataFrame(columns=cols)
    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)
    df["class_name"] = df["class_name"].astype(str)
    return df.reset_index(drop=True)


def map_known(
    preds: pd.DataFrame,
    gts: pd.DataFrame,
    classes: Collection[str],
    iou_thr: float = 0.5,
    mode: str = "voc",
) -> tuple[float, Dict[str, float], List[str]]:
    """mAP over ``classes`` (those with GT). Predictions of other classes are ignored."""
    preds, gts = _clean(preds, True), _clean(gts, False)
    aps: Dict[str, float] = {}
    skipped: List[str] = []
    for c in sorted(classes):
        g = gts[gts["class_name"] == c]
        if len(g) == 0:
            skipped.append(c)
            continue
        aps[c] = class_ap(preds[preds["class_name"] == c], g, iou_thr, mode)
    return (float(np.mean(list(aps.values()))) if aps else 0.0), aps, skipped


def score_predictions(
    preds: pd.DataFrame,
    gts: pd.DataFrame,
    known_classes: Collection[str],
    iou_thr: float = 0.5,
    apply_conflict: bool = True,
    unknown_name: str = UNKNOWN,
    match_mode: str = "voc",
) -> ScoreResult:
    """Score ``preds`` against ``gts`` (columns per brief). GT whose class is not in ``known_classes`` is unknown."""
    known = set(known_classes)
    preds, gts = _clean(preds, True), _clean(gts, False)
    n_before = len(preds)
    if apply_conflict:
        preds = apply_conflict_rule(preds, iou_thr, unknown_name)
    ignored = sorted(set(preds["class_name"]) - known - {unknown_name})
    gt_unknown = gts[(gts["class_name"] == unknown_name) | (~gts["class_name"].isin(known))]
    gt_known = gts[gts["class_name"].isin(known)]

    map_k, aps, skipped = map_known(preds, gt_known, known, iou_thr, match_mode)

    p_unk = preds[preds["class_name"] == unknown_name].reset_index(drop=True)
    n_gt_u, n_p_u = len(gt_unknown), len(p_unk)
    matched = 0
    if n_gt_u and n_p_u:
        tp, _ = _match(p_unk, _group_boxes(gt_unknown), iou_thr, match_mode)
        matched = int(tp.sum())
    u_rec = matched / n_gt_u if n_gt_u else 0.0
    u_prec = matched / n_p_u if n_p_u else 0.0
    u_f1 = 0.0 if (u_rec + u_prec) == 0 else 2 * u_rec * u_prec / (u_rec + u_prec)
    return ScoreResult(
        map_k=map_k, ap_per_class=aps, skipped_classes=skipped,
        u_rec=u_rec, u_prec=u_prec, u_f1=u_f1, oe_score=oe_score(map_k, u_f1),
        n_unknown_gt=n_gt_u, n_unknown_pred=n_p_u, n_unknown_matched_gt=matched, n_unknown_matched_pred=matched,
        n_pred_before_conflict=n_before, n_pred_after_conflict=len(preds), ignored_pred_classes=ignored,
    )
