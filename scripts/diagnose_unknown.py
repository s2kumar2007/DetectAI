#!/usr/bin/env python
"""Diagnose unknown objectness and open-world metrics for a checkpoint.

Prints one table split by known vs unknown GT boxes:
- Mean objectness (S_obj)
- Mean cos_max
- Mean softmax pmax
- Fraction with cos_max < 0.3
- Proposal recall at IoU >= 0.5

Also prints local scorer metrics: mAP_K, U-Rec, U-Prec, U-F1, OE-Score.
Saves JSON and Markdown reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torchvision.ops import box_iou

import _bootstrap  # noqa: F401
from anveshak.data.build import load_class_names, load_set
from anveshak.data.split import load_split, val_view
from anveshak.eval.io import records_to_gt
from anveshak.eval.local_scorer import score_predictions
from anveshak.postprocess.factory import build_predictor
from anveshak.training.engine import resolve_device
from anveshak.utils.config import load_config
from anveshak.utils.log import get_logger

log = get_logger("anveshak.diagnose")


def diagnose_checkpoint(
    checkpoint_path: str | Path,
    split_path: str | Path,
    cfg,
    device_name: str = "auto",
    no_tta: bool = True,
    calib_path: str | Path | None = None,
) -> Dict[str, Any]:
    split = load_split(split_path)
    heldout = set(split.get("heldout_classes", []))
    device = resolve_device(device_name)
    pred, slotmap, meta = build_predictor(checkpoint_path, cfg, calib_path, device, tta=False if no_tta else None)

    names = load_class_names(cfg)
    recs = load_set(cfg, cfg.data.train_dir, names)
    val = val_view(recs, split)
    known_names = slotmap.names

    known_objs, known_cos, known_pmax, known_low_cos, known_recalled = [], [], [], [], []
    unk_objs, unk_cos, unk_pmax, unk_low_cos, unk_recalled = [], [], [], [], []
    bg_objs = []

    for rec in val:
        tile, s = pred.load_tile(rec)
        raws = pred.compute_raws(tile)
        raw = raws[0]  # view: orig

        prop_boxes = raw["boxes"].float()  # (P, 4) in letterbox coords
        prop_obj = raw["obj"].float()      # (P,)

        all_gt_boxes = []
        gt_t_list = []
        for b in rec.boxes:
            gb = torch.tensor(b, dtype=torch.float32).unsqueeze(0) * s
            all_gt_boxes.append(gb)
            gt_t_list.append(gb)

        if gt_t_list:
            x, _ = pred.model.preprocess([tile])
            feats = pred.model.extract(x)
            shapes = [tuple(x.shape[-2:])]
            all_gt_t = torch.cat(gt_t_list, dim=0)
            h = pred.model.roi_head.trunk(feats, [all_gt_t], shapes)
            gt_emb, _, gt_obj_l = pred.model.roi_head.outputs(h)
            gt_logits, gt_cos = pred.model.roi_head.cls(gt_emb)
            gt_active = pred.model.roi_head.cls.active
            gt_probs = torch.softmax(gt_logits, dim=1) if bool(gt_active.any()) else torch.zeros_like(gt_logits)
            gt_max_cos = gt_cos[:, gt_active].max(1).values if bool(gt_active.any()) else torch.zeros(gt_cos.shape[0])
            gt_obj = torch.sigmoid(gt_obj_l)

        for i, (b, l) in enumerate(zip(rec.boxes, rec.labels)):
            gb = all_gt_boxes[i]
            is_unk = (l == "unknown") or (l in heldout)
            obj_val = float(gt_obj[i])
            cos_val = float(gt_max_cos[i])
            pmax_val = float(gt_probs[i].max()) if gt_probs.shape[1] > 0 else 0.0
            low_cos = float(cos_val < 0.3)

            if prop_boxes.shape[0] > 0:
                ious = box_iou(gb, prop_boxes).squeeze(0)  # (P,)
                max_iou = float(ious.max())
                recalled = float(max_iou >= 0.5)
            else:
                recalled = 0.0

            if is_unk:
                unk_objs.append(obj_val)
                unk_cos.append(cos_val)
                unk_pmax.append(pmax_val)
                unk_low_cos.append(low_cos)
                unk_recalled.append(recalled)
            elif l in known_names:
                known_objs.append(obj_val)
                known_cos.append(cos_val)
                known_pmax.append(pmax_val)
                known_low_cos.append(low_cos)
                known_recalled.append(recalled)

        # Background proposals: max IoU < 0.1 to all GT in image
        if prop_boxes.shape[0] > 0:
            if all_gt_boxes:
                all_gt = torch.cat(all_gt_boxes, dim=0)
                max_gt_ious = box_iou(prop_boxes, all_gt).max(1).values
                bg_mask = max_gt_ious < 0.1
            else:
                bg_mask = torch.ones(prop_boxes.shape[0], dtype=torch.bool)
            if bg_mask.any():
                bg_objs.extend(prop_obj[bg_mask].tolist())

    def _mean(arr: List[float]) -> float:
        return float(np.mean(arr)) if arr else 0.0

    table = {
        "known": {
            "count": len(known_objs),
            "mean_obj": round(_mean(known_objs), 4),
            "mean_cos": round(_mean(known_cos), 4),
            "mean_pmax": round(_mean(known_pmax), 4),
            "frac_cos_lt_03": round(_mean(known_low_cos), 4),
            "proposal_recall_05": round(_mean(known_recalled), 4),
        },
        "unknown": {
            "count": len(unk_objs),
            "mean_obj": round(_mean(unk_objs), 4),
            "mean_cos": round(_mean(unk_cos), 4),
            "mean_pmax": round(_mean(unk_pmax), 4),
            "frac_cos_lt_03": round(_mean(unk_low_cos), 4),
            "proposal_recall_05": round(_mean(unk_recalled), 4),
        },
        "background": {
            "count": len(bg_objs),
            "mean_obj": round(_mean(bg_objs), 4),
            "mean_cos": None,
            "mean_pmax": None,
            "frac_cos_lt_03": None,
            "proposal_recall_05": None,
        },
    }

    # Run full predictor + scorer
    df = pred.predict_records(val)
    gt = records_to_gt(val)
    score_res = score_predictions(df, gt, known_names)

    metrics = {
        "map_k": round(float(score_res.map_k), 4),
        "u_rec": round(float(score_res.u_rec), 4),
        "u_prec": round(float(score_res.u_prec), 4),
        "u_f1": round(float(score_res.u_f1), 4),
        "oe_score": round(float(score_res.oe_score), 4),
        "n_unknown_pred": int(score_res.n_unknown_pred),
    }

    return {
        "checkpoint": str(checkpoint_path),
        "split": str(split_path),
        "heldout_classes": list(heldout),
        "n_val_images": len(val),
        "table": table,
        "metrics": metrics,
    }


def format_markdown(data: Dict[str, Any]) -> str:
    t = data["table"]
    m = data["metrics"]
    lines = [
        f"# Unknown Objectness Diagnostic ({Path(data['checkpoint']).name})",
        "",
        f"- **Checkpoint**: `{data['checkpoint']}`",
        f"- **Split**: `{data['split']}` (held-out fake unknowns: {data['heldout_classes']})",
        f"- **Validation images**: {data['n_val_images']}",
        "",
        "## GT Box Diagnostics",
        "",
        "| Group | Count | Mean Objectness | Mean Cos Max | Mean Softmax Pmax | Fraction Cos < 0.3 | Proposal Recall @ 0.5 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| Known GT | {t['known']['count']} | {t['known']['mean_obj']:.4f} | {t['known']['mean_cos']:.4f} | {t['known']['mean_pmax']:.4f} | {t['known']['frac_cos_lt_03']:.4f} | {t['known']['proposal_recall_05']:.4f} |",
        f"| Unknown GT | {t['unknown']['count']} | {t['unknown']['mean_obj']:.4f} | {t['unknown']['mean_cos']:.4f} | {t['unknown']['mean_pmax']:.4f} | {t['unknown']['frac_cos_lt_03']:.4f} | {t['unknown']['proposal_recall_05']:.4f} |",
        f"| Background | {t['background']['count']} | {t['background']['mean_obj']:.4f} | - | - | - | - |",
        "",
        "## Local Scorer Metrics",
        "",
        "| Metric | Value |",
        "| :--- | :---: |",
        f"| **mAP_K** | {m['map_k']:.4f} |",
        f"| **U-Rec** | {m['u_rec']:.4f} |",
        f"| **U-Prec** | {m['u_prec']:.4f} |",
        f"| **U-F1** | {m['u_f1']:.4f} |",
        f"| **OE-Score** | {m['oe_score']:.4f} |",
        f"| **Unknown Preds** | {m['n_unknown_pred']} |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt")
    ap.add_argument("--split", required=True, help="Path to split.json with held-out classes")
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL")
    ap.add_argument("--calibration", default=None, help="Optional calibration.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no_tta", action="store_true", default=True)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    data = diagnose_checkpoint(args.checkpoint, args.split, cfg, args.device, args.no_tta, args.calibration)
    md = format_markdown(data)

    print(md)

    ckpt_p = Path(args.checkpoint)
    out_json = Path(args.out_json) if args.out_json else ckpt_p.parent / "diagnose_unknown.json"
    out_md = Path(args.out_md) if args.out_md else ckpt_p.parent / "diagnose_unknown.md"

    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    out_md.write_text(md, encoding="utf-8")
    log.info("Saved diagnostic to %s and %s", out_json, out_md)


if __name__ == "__main__":
    main()
