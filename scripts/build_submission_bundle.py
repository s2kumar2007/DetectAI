#!/usr/bin/env python
"""Bundle everything needed to audit / reproduce a submission into one zip (no data, no images).

Contents: predictions.csv (+ .meta.json), class_slots.json, calibration.json, retention/summary/metrics files, resolved training
config, checkpoint hashes, verification record, MANIFEST.json (sha256 of every file, git commit, versions) and REPRODUCE.md.
The checkpoint itself is included only with --include_checkpoint (it can be large).

    python scripts/build_submission_bundle.py --run_dir runs/round2 --csv submission/predictions.csv \
        --calibration calibration.json --set data.root=/path/to/data --out submission/anveshak_round2_bundle.zip
"""
import argparse
import json
import platform
import sys
import zipfile
from pathlib import Path

import torch

import _bootstrap  # noqa: F401
from anveshak.data.build import load_test_images
from anveshak.postprocess.csv_check import check_csv
from anveshak.training.checkpoint import load_payload
from anveshak.utils.config import load_config
from anveshak.utils.hashing import file_sha256, git_commit_hash


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, help="directory holding checkpoint.pt of the submitted model")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL")
    ap.add_argument("--out", default="submission/bundle.zip")
    ap.add_argument("--include_checkpoint", action="store_true")
    ap.add_argument("--allow_unverified", action="store_true", help="bundle even if verify_checkpoint.py has not passed")
    a = ap.parse_args()
    run = Path(a.run_dir)
    ck = run / "checkpoint.pt"
    if not ck.exists():
        raise SystemExit(f"{ck} not found")
    verified = Path(str(ck) + ".verified.json")
    vj = json.loads(verified.read_text()) if verified.exists() else None
    if not (vj and vj.get("passed") and vj.get("checkpoint_sha256") == file_sha256(ck)) and not a.allow_unverified:
        raise SystemExit("checkpoint is not verified against the CSV: run scripts/verify_checkpoint.py first (or --allow_unverified)")
    pl = load_payload(ck)
    cfg = load_config(a.config, a.set)
    records = load_test_images(cfg)
    sizes = {r.image_id: (r.width, r.height) for r in records}
    res = check_csv(a.csv, set(pl["slot_map"]["class_to_slot"]) | {"unknown"}, sizes)
    res.raise_if_bad()

    files = {"predictions.csv": Path(a.csv), "class_slots.json": run / "class_slots.json"}
    for extra in (Path(a.csv + ".meta.json"), run / "retention_report.json", run / "train_summary.json", run / "metrics.jsonl",
                  run / "train.log", verified, run / "checkpoint.pt.sha256"):
        if extra.exists():
            files[extra.name if extra.name != "predictions.csv.meta.json" else "predictions.csv.meta.json"] = extra
    if a.calibration:
        files["calibration.json"] = Path(a.calibration)
    if a.include_checkpoint:
        files["checkpoint.pt"] = ck
    manifest = {
        "git_commit": git_commit_hash(), "python": platform.python_version(), "torch": torch.__version__,
        "stage": pl["stage"], "seed": pl["seed"], "checkpoint_sha256": file_sha256(ck), "init_hash": pl.get("init_hash"),
        "prev_sha256": pl.get("prev_sha256"), "checkpoint_git_commit": pl["git_commit"], "iterations": pl.get("iteration"),
        "n_test_images": len(records), "n_rows": res.n_rows, "checkpoint_included": a.include_checkpoint,
        "files": {n: file_sha256(p) for n, p in files.items()},
    }
    repro = (
        "# Reproduce\n\n```bash\n"
        f"git checkout {git_commit_hash()}\npip install -r requirements.txt\n"
        f"python scripts/infer_and_validate.py --checkpoint {ck} --calibration calibration.json --set data.root=<DATA> --out predictions.csv\n"
        f"python scripts/verify_checkpoint.py --checkpoint {ck} --csv predictions.csv --calibration calibration.json --set data.root=<DATA>\n```\n\n"
        f"Checkpoint SHA-256: `{manifest['checkpoint_sha256']}`\n"
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for n, p in files.items():
            z.write(p, n)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
        z.writestr("REPRODUCE.md", repro)
    print(f"wrote {a.out} ({len(files) + 2} files, checkpoint {'included' if a.include_checkpoint else 'referenced by hash only'})")


if __name__ == "__main__":
    main()
