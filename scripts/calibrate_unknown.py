#!/usr/bin/env python
"""Tune the unknown gate (alpha, tau_unk, unknown cap, cos_ref) to maximise OE-Score on the SIMULATED-unknown validation
split of the TRAINING data (never on test images), and write calibration.json for inference.

The checkpoint must come from a run whose training view excluded the held-out classes (see make_val_split.py):

    python scripts/calibrate_unknown.py --checkpoint runs/calib_stage1/checkpoint.pt --split splits/set_a_split.json \
        --set data.root=/path/to/data --out calibration.json
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from anveshak.data.build import load_class_names, load_set
from anveshak.data.split import load_split, val_view
from anveshak.eval.io import records_to_gt
from anveshak.eval.local_scorer import score_predictions
from anveshak.postprocess.factory import build_predictor
from anveshak.postprocess.unknown import Calibration
from anveshak.training.engine import resolve_device
from anveshak.utils.boxes import iou_matrix
from anveshak.utils.config import load_config
from anveshak.utils.hashing import file_sha256
from anveshak.utils.log import get_logger

log = get_logger("anveshak.calibrate")


def _floats(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def _trim_caps(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(df) == 0:
        return df
    unk = df["class_name"] == "unknown"
    rank = unk.groupby(df["image_id"]).cumsum()
    return df[~unk | (rank <= cap)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL")
    ap.add_argument("--split", default=None, help="split JSON from make_val_split.py (default: data.split_file)")
    ap.add_argument("--out", default="calibration.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no_tta", action="store_true", help="calibrate without TTA (faster; use the same setting at inference)")
    ap.add_argument("--max_images", type=int, default=0, help="limit validation images (0 = all)")
    ap.add_argument("--alphas", default="0.5,1,2")
    ap.add_argument("--taus", default="0.02,0.05,0.08,0.12,0.16,0.2,0.25,0.3,0.4,0.5")
    ap.add_argument("--caps", default="5,10,20,40")
    ap.add_argument("--cos_ref_pcts", default="10,25,50,75", help="percentiles of known-object max-cosine used as cos_ref (plus 1.0)")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    split_path = a.split or cfg.data.split_file
    if not split_path:
        raise SystemExit("a simulated-unknown split is required (--split or data.split_file); run scripts/make_val_split.py first")
    split = load_split(split_path)
    device = resolve_device(a.device)
    pred, slotmap, meta = build_predictor(a.checkpoint, cfg, None, device, tta=False if a.no_tta else None)
    leaked = sorted(set(split["heldout_classes"]) & set(slotmap.names))
    if leaked:
        raise SystemExit(f"checkpoint already knows held-out classes {leaked}: train the calibration run with this split (data.split_file)")
    names = load_class_names(cfg)
    recs = load_set(cfg, cfg.data.train_dir, names)
    val = val_view(recs, split)
    if a.max_images:
        val = val[: a.max_images]
    gt = records_to_gt(val)
    known = slotmap.names
    log.info("calibrating on %d validation images (%d GT boxes, held-out fake unknowns: %s)", len(val), len(gt), split["heldout_classes"])

    cache, cos_known = [], []
    for rec in val:
        tile, s = pred.load_tile(rec)
        raws = pred.compute_raws(tile)
        kn = pred.known_part(raws)
        cache.append((rec, s, raws, kn))
        kb = np.asarray([b for b, l in zip(rec.boxes, rec.labels) if l in set(known)], np.float32).reshape(-1, 4) * s
        if len(kb):
            r0 = raws[0]
            ious = iou_matrix(r0["boxes"].numpy(), kb)
            m = ious.max(1) >= 0.5
            cos_known.extend(r0["cos_max"].numpy()[m].tolist())
    cos_refs = sorted(set([1.0] + ([float(np.percentile(cos_known, p)) for p in _floats(a.cos_ref_pcts)] if cos_known else [])))
    alphas = sorted(set(_floats(a.alphas) + [float(cfg.calibration.alpha)]))
    taus = sorted(set(_floats(a.taus) + [float(cfg.calibration.tau_unk)]))
    caps = sorted(set([int(c) for c in _floats(a.caps)] + [int(cfg.calibration.unk_cap)]))
    log.info("grid: cos_ref=%s alpha=%s tau=%s caps=%s", [round(c, 3) for c in cos_refs], alphas, taus, caps)

    results = []
    for cos_ref, alpha, tau in itertools.product(cos_refs, alphas, taus):
        calib = Calibration(alpha, tau, max(caps), cos_ref)
        rows = []
        for rec, s, raws, kn in cache:
            rows.extend(pred.finalize(rec.image_id, kn, pred.unknown_part(raws, calib), s, rec.width, rec.height, calib))
        full = pd.DataFrame(rows, columns=["image_id", "class_name", "score", "xmin", "ymin", "xmax", "ymax"])
        for cap in caps:
            r = score_predictions(_trim_caps(full, cap), gt, known)
            results.append({"cos_ref": cos_ref, "alpha": alpha, "tau_unk": tau, "unk_cap": cap, "oe": r.oe_score, "map_k": r.map_k,
                            "u_f1": r.u_f1, "u_rec": r.u_rec, "u_prec": r.u_prec, "n_unknown_pred": r.n_unknown_pred})
    grid = pd.DataFrame(results).sort_values(["oe", "u_f1", "unk_cap", "tau_unk"], ascending=[False, False, True, False]).reset_index(drop=True)
    best = grid.iloc[0]
    d = Calibration(float(cfg.calibration.alpha), float(cfg.calibration.tau_unk), int(cfg.calibration.unk_cap), float(cfg.calibration.cos_ref))
    base = grid[(grid.alpha == d.alpha) & (grid.tau_unk == d.tau_unk) & (grid.unk_cap == d.unk_cap) & (grid.cos_ref == d.cos_ref)]
    out = {
        "alpha": float(best.alpha), "tau_unk": float(best.tau_unk), "unk_cap": int(best.unk_cap), "cos_ref": float(best.cos_ref),
        "val": {k: float(best[k]) for k in ("oe", "map_k", "u_f1", "u_rec", "u_prec")},
        "val_with_config_defaults": ({k: float(base.iloc[0][k]) for k in ("oe", "map_k", "u_f1")} if len(base) else None),
        "grid_points": int(len(grid)), "n_val_images": len(val), "heldout_classes": split["heldout_classes"], "split_seed": split.get("seed"),
        "tta": pred.tta, "checkpoint_sha256": file_sha256(a.checkpoint),
        "note": "Tuned ONLY on the simulated-unknown validation split of the training data. Not a real-data accuracy figure.",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    grid.to_csv(str(Path(a.out).with_suffix("")) + "_grid.csv", index=False)
    print(json.dumps({k: out[k] for k in ("alpha", "tau_unk", "unk_cap", "cos_ref", "val", "val_with_config_defaults")}, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
