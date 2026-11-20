#!/usr/bin/env python
"""Inference on the (unlabelled) test images -> submission CSV -> automatic validation.

    python scripts/infer_and_validate.py --checkpoint runs/stage2/checkpoint.pt --calibration calibration.json \
        --set data.root=/path/to/data --out submission/predictions.csv

Deterministic: fixed seeds, deterministic kernels, 4-view TTA (original, h-flip, v-flip, rot90) fused with in-house WBF
(disable with --no_tta). The organisers' validate_predictions.py is run when `postprocess.validator_path` points to it;
our built-in checker (src/anveshak/postprocess/csv_check.py) ALWAYS runs. Any violation aborts with a non-zero exit code.
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from anveshak.data.build import load_test_images
from anveshak.eval.io import read_boxes_csv
from anveshak.eval.local_scorer import score_predictions
from anveshak.postprocess.factory import build_predictor, predict_test_set
from anveshak.postprocess.writer import validate_submission, write_predictions_csv
from anveshak.training.engine import resolve_device
from anveshak.utils.config import load_config
from anveshak.utils.hashing import file_sha256, git_commit_hash


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL")
    ap.add_argument("--calibration", default=None, help="calibration.json from calibrate_unknown.py (else config defaults)")
    ap.add_argument("--out", default="submission/predictions.csv")
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--gt", default=None, help="optional GT CSV (e.g. SYNTHETIC data) to run the local scorer")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    device = resolve_device(a.device)
    pred, slotmap, meta = build_predictor(a.checkpoint, cfg, a.calibration, device, tta=False if a.no_tta else None)
    records = load_test_images(cfg)
    if not records:
        raise SystemExit(f"no test images found under {cfg.data.root}/{cfg.data.test}")
    df = predict_test_set(pred, records, progress=True)
    out = write_predictions_csv(df, a.out)
    sizes = {r.image_id: (r.width, r.height) for r in records}
    info = validate_submission(out, slotmap.names, sizes, cfg.postprocess, int(cfg.infer.max_rows_per_image))
    n_unk = int((df["class_name"] == "unknown").sum())
    meta_out = {
        "checkpoint": str(a.checkpoint), "checkpoint_sha256": file_sha256(a.checkpoint), "calibration": pred.calib.to_dict(),
        "calibration_file": a.calibration, "tta": pred.tta, "n_images": len(records), "n_rows": len(df), "n_unknown_rows": n_unk,
        "csv_sha256": file_sha256(out), "git_commit": git_commit_hash(), "validation": info,
    }
    Path(str(out) + ".meta.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    print(f"wrote {out}: {len(df)} rows ({n_unk} unknown) for {len(records)} images - validation PASSED")
    if a.gt:
        res = score_predictions(read_boxes_csv(out).rename(columns=str), read_boxes_csv(a.gt), slotmap.names)
        print("local scorer (labelled data you supplied; never test labels from the organisers):")
        print(res.summary())


if __name__ == "__main__":
    main()
