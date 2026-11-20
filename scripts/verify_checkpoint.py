#!/usr/bin/env python
"""Re-run a checkpoint on the test images and compare with a submitted CSV: row counts, boxes within 1 px, scores within 1e-3.
Prints PASS/FAIL (exit code 0/1). On PASS writes <checkpoint>.verified.json, which Round 2 training requires.

    python scripts/verify_checkpoint.py --checkpoint runs/stage2/checkpoint.pt --csv submission/predictions.csv \
        --calibration calibration.json --set data.root=/path/to/data
Calibration / TTA default to what `<csv>.meta.json` (written by infer_and_validate.py) recorded.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from anveshak.data.build import load_test_images
from anveshak.eval.io import read_boxes_csv
from anveshak.postprocess.factory import build_predictor, predict_test_set
from anveshak.postprocess.writer import write_predictions_csv
from anveshak.training.engine import resolve_device
from anveshak.utils.config import load_config
from anveshak.utils.hashing import file_sha256, git_commit_hash

BOX = ["xmin", "ymin", "xmax", "ymax"]


def compare_predictions(new: pd.DataFrame, ref: pd.DataFrame, px_tol: float = 1.0, score_tol: float = 1e-3, max_show: int = 15):
    """Return a list of problems (empty = match). Rows are paired within (image, class) by smallest box distance."""
    problems = []
    if len(new) != len(ref):
        problems.append(f"row count differs: re-run {len(new)} vs submitted {len(ref)}")
    gn, gr = dict(tuple(new.groupby(["image_id", "class_name"]))), dict(tuple(ref.groupby(["image_id", "class_name"])))
    for key in sorted(set(gn) | set(gr)):
        a, b = gn.get(key), gr.get(key)
        na, nb = (0 if a is None else len(a)), (0 if b is None else len(b))
        if na != nb:
            problems.append(f"{key}: {na} re-run rows vs {nb} submitted rows")
            continue
        A, B = a[BOX].to_numpy(float), b[BOX].to_numpy(float)
        sa, sb = a["score"].to_numpy(float), b["score"].to_numpy(float)
        free = list(range(na))
        for j in range(nb):
            d = [np.abs(A[i] - B[j]).max() for i in free]
            k = free[int(np.argmin(d))]
            free.remove(k)
            if min(d) > px_tol:
                problems.append(f"{key}: submitted box {B[j].tolist()} has no re-run box within {px_tol}px (closest {min(d):.2f}px)")
            elif abs(sa[k] - sb[j]) > score_tol:
                problems.append(f"{key}: score {sb[j]:.6f} vs re-run {sa[k]:.6f}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--px_tol", type=float, default=1.0)
    ap.add_argument("--score_tol", type=float, default=1e-3)
    ap.add_argument("--no_write", action="store_true", help="do not write <checkpoint>.verified.json")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    meta_path = Path(a.csv + ".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    calib = a.calibration or meta.get("calibration_file")
    tta = False if a.no_tta else meta.get("tta", None)
    pred, slotmap, _ = build_predictor(a.checkpoint, cfg, calib, resolve_device(a.device), tta=tta)
    if meta.get("calibration") and not a.calibration and not calib:
        from anveshak.postprocess.unknown import Calibration
        pred.calib = Calibration(**meta["calibration"])
    records = load_test_images(cfg)
    new = predict_test_set(pred, records, progress=True)
    ref = read_boxes_csv(a.csv)
    ref["image_id"] = ref["image_id"].astype(str)
    problems = compare_predictions(new, ref, a.px_tol, a.score_tol)
    ids_missing = sorted({r.image_id for r in records} - set(ref["image_id"]))[:3]
    passed = not problems
    print(f"re-run rows: {len(new)}  submitted rows: {len(ref)}  images: {len(records)}  (images without rows: {len({r.image_id for r in records} - set(ref['image_id']))})")
    if passed:
        print("PASS - checkpoint reproduces the submitted CSV (boxes within %.1f px, scores within %g)" % (a.px_tol, a.score_tol))
        if not a.no_write:
            v = {"passed": True, "checkpoint": str(a.checkpoint), "checkpoint_sha256": file_sha256(a.checkpoint), "csv": str(a.csv),
                 "csv_sha256": file_sha256(a.csv), "n_rows": len(ref), "n_images": len(records), "tta": pred.tta,
                 "calibration": pred.calib.to_dict(), "git_commit": git_commit_hash()}
            Path(a.checkpoint + ".verified.json").write_text(json.dumps(v, indent=2), encoding="utf-8")
            print(f"wrote {a.checkpoint}.verified.json")
        return 0
    print(f"FAIL - {len(problems)} mismatches:")
    for p in problems[:15]:
        print("  ", p)
    return 1


if __name__ == "__main__":
    sys.exit(main())
