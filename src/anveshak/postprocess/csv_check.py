"""Strict prediction-CSV checker (our equivalent of the organisers' validate_predictions.py).

Rules: UTF-8 (no BOM), exact header ``image_id,class_name,score,xmin,ymin,xmax,ymax``, case-sensitive class codes or
``unknown``, score finite in [0,1], finite numeric coords with xmin<xmax and ymin<ymax clipped inside the tile
[0,W]x[0,H], no duplicate rows, at most 1000 rows per image, and only known image ids.
"""
from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Collection, Dict, List, Mapping, Optional, Tuple

HEADER = ["image_id", "class_name", "score", "xmin", "ymin", "xmax", "ymax"]
UNKNOWN = "unknown"
MAX_ROWS_PER_IMAGE = 1000


class CSVValidationError(RuntimeError):
    pass


@dataclass
class CheckResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    n_rows: int = 0
    n_images_with_rows: int = 0
    images_without_rows: List[str] = field(default_factory=list)

    def raise_if_bad(self, max_show: int = 25) -> None:
        if not self.ok:
            shown = "\n  ".join(self.errors[:max_show])
            more = f"\n  ... and {len(self.errors) - max_show} more" if len(self.errors) > max_show else ""
            raise CSVValidationError(f"CSV validation FAILED ({len(self.errors)} problems):\n  {shown}{more}")


def check_csv(
    path: str | Path,
    allowed_classes: Collection[str],
    image_sizes: Optional[Mapping[str, Tuple[int, int]]] = None,
    max_rows_per_image: int = MAX_ROWS_PER_IMAGE,
    max_errors: int = 200,
) -> CheckResult:
    """Validate ``path``. ``image_sizes`` maps image_id -> (width, height); when given, unknown ids are errors."""
    errors: List[str] = []
    warnings: List[str] = []

    def err(msg: str) -> None:
        if len(errors) < max_errors:
            errors.append(msg)

    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        err("file starts with a UTF-8 BOM; write plain UTF-8")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return CheckResult(False, [f"file is not valid UTF-8: {exc}"])

    allowed = set(allowed_classes) | {UNKNOWN}
    lower_map = {c.lower(): c for c in allowed}
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        return CheckResult(False, ["file is empty (no header)"])
    if rows[0] != HEADER:
        err(f"header must be exactly {','.join(HEADER)}; got {','.join(rows[0])}")
    per_image: Counter = Counter()
    seen: set = set()
    n = 0
    for ln, row in enumerate(rows[1:], start=2):
        if not row:
            continue
        n += 1
        if len(row) != 7:
            err(f"line {ln}: expected 7 columns, got {len(row)}")
            continue
        img, cls, s_score, *s_box = row
        if img == "":
            err(f"line {ln}: empty image_id")
        elif image_sizes is not None and img not in image_sizes:
            err(f"line {ln}: image_id {img!r} is not a known test image")
        if cls not in allowed:
            hint = f" (did you mean {lower_map[cls.lower()]!r}? class codes are case-sensitive)" if cls.lower() in lower_map else ""
            err(f"line {ln}: class_name {cls!r} is not a valid class code or 'unknown'{hint}")
        try:
            score = float(s_score)
            box = [float(v) for v in s_box]
        except ValueError:
            err(f"line {ln}: non-numeric score/coordinate in {row[2:]}")
            continue
        if not math.isfinite(score) or not (0.0 <= score <= 1.0):
            err(f"line {ln}: score {s_score} outside [0,1] or not finite")
        if not all(math.isfinite(v) for v in box):
            err(f"line {ln}: non-finite coordinate")
            continue
        xmin, ymin, xmax, ymax = box
        if not xmin < xmax:
            err(f"line {ln}: xmin ({xmin}) must be < xmax ({xmax})")
        if not ymin < ymax:
            err(f"line {ln}: ymin ({ymin}) must be < ymax ({ymax})")
        if xmin < 0 or ymin < 0:
            err(f"line {ln}: negative coordinate ({xmin},{ymin})")
        if image_sizes is not None and img in image_sizes:
            w, h = image_sizes[img]
            if xmax > w + 1e-6 or ymax > h + 1e-6:
                err(f"line {ln}: box ({xmax},{ymax}) exceeds tile size {w}x{h}")
        key = (img, cls, round(score, 9), *[round(v, 6) for v in box])
        if key in seen:
            err(f"line {ln}: duplicate row {row}")
        seen.add(key)
        per_image[img] += 1
    for img, c in per_image.items():
        if c > max_rows_per_image:
            err(f"image {img!r} has {c} rows (max {max_rows_per_image})")
    missing: List[str] = []
    if image_sizes is not None:
        missing = sorted(set(image_sizes) - set(per_image))
        if missing:
            warnings.append(f"{len(missing)} test images have no prediction rows (e.g. {missing[:3]})")
    return CheckResult(not errors, errors, warnings, n, len(per_image), missing)


def assert_csv_valid(*args, **kwargs) -> CheckResult:
    res = check_csv(*args, **kwargs)
    res.raise_if_bad()
    return res
