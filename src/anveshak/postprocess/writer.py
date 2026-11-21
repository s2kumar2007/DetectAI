"""Submission CSV writer + validator integration."""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import pandas as pd

from ..utils.log import get_logger
from .csv_check import HEADER, CSVValidationError, check_csv

log = get_logger("anveshak.submit")


def write_predictions_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """UTF-8 (no BOM), exact header, LF line endings; coordinates 2 decimals, score 6 decimals."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(",".join(HEADER) + "\n")
        for r in df[HEADER].itertuples(index=False):
            fh.write(f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},{r[6]:.2f}\n")
    return path


def validate_submission(csv_path: str | Path, allowed_classes: Sequence[str], image_sizes: Mapping[str, Tuple[int, int]],
                        postprocess_cfg=None, max_rows_per_image: int = 1000) -> dict:
    """Run the organisers' validate_predictions.py when configured and present, and ALWAYS our own equivalent checker.
    Raises CSVValidationError on any violation."""
    info = {"external_validator": None, "external_ok": None}
    ext = getattr(postprocess_cfg, "validator_path", None) if postprocess_cfg is not None else None
    if ext and Path(ext).exists():
        cmd = str(postprocess_cfg.validator_cmd).format(python=sys.executable, validator=ext, csv=str(csv_path))
        log.info("running organisers' validator: %s", cmd)
        proc = subprocess.run(shlex.split(cmd, posix=(os.name != "nt")), capture_output=True, text=True)
        for line in (proc.stdout + proc.stderr).splitlines():
            log.info("[validate_predictions] %s", line)
        info.update(external_validator=str(ext), external_ok=proc.returncode == 0)
        if proc.returncode != 0:
            raise CSVValidationError(f"organisers' validator failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    else:
        log.info("organisers' validator not found (postprocess.validator_path=%s) - using built-in checker only", ext)
    res = check_csv(csv_path, set(allowed_classes) | {"unknown"}, image_sizes, max_rows_per_image)
    res.raise_if_bad()
    for w in res.warnings[:10]:
        log.warning("csv check: %s", w)
    info.update(rows=res.n_rows, images_with_rows=res.n_images_with_rows, images_without_rows=len(res.images_without_rows))
    return info
