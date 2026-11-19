#!/usr/bin/env python
"""Print class counts, box-size stats and image sizes for any dataset root.

Example:
    python scripts/inspect_data.py --root /path/to/anveshak_data/set_a --class_list /path/to/class_list.txt
"""
import argparse
import sys
from collections import Counter

import numpy as np

import _bootstrap  # noqa: F401
from anveshak.data.classes import load_class_list
from anveshak.data.formats import detect_format, load_annotations, load_image_dir


def describe(root: str, fmt: str = "auto", class_list_path: str | None = None, image_id_mode: str = "stem") -> dict:
    class_list = load_class_list(class_list_path) if class_list_path else None
    try:
        fmt_used = detect_format(root) if fmt == "auto" else fmt
        recs = load_annotations(root, fmt_used, class_list, image_id_mode)
    except ValueError:
        fmt_used, recs = "images-only", load_image_dir(root, image_id_mode)
    counts = Counter(l for r in recs for l in r.labels)
    boxes = np.concatenate([r.boxes for r in recs], axis=0) if any(len(r.boxes) for r in recs) else np.zeros((0, 4))
    w = boxes[:, 2] - boxes[:, 0] if len(boxes) else np.zeros(0)
    h = boxes[:, 3] - boxes[:, 1] if len(boxes) else np.zeros(0)
    sizes = Counter((r.width, r.height) for r in recs)
    info = {
        "format": fmt_used,
        "images": len(recs),
        "images_without_boxes": sum(1 for r in recs if len(r.boxes) == 0),
        "boxes": int(len(boxes)),
        "class_counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "image_sizes": dict(sizes.most_common(10)),
        "w": w, "h": h,
    }
    if class_list:
        info["classes_in_list_not_in_data"] = [c for c in class_list if c not in counts]
        info["classes_in_data_not_in_list"] = sorted(c for c in counts if c not in class_list)
    return info


def _pct(a: np.ndarray) -> str:
    if len(a) == 0:
        return "n/a"
    q = np.percentile(a, [0, 5, 50, 95, 100])
    return "min %.1f | p5 %.1f | median %.1f | p95 %.1f | max %.1f" % tuple(q)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--format", default="auto", choices=["auto", "coco", "yolo", "voc", "csv"])
    ap.add_argument("--class_list", default=None)
    ap.add_argument("--image_id_mode", default="stem", choices=["stem", "filename"])
    a = ap.parse_args()
    info = describe(a.root, a.format, a.class_list, a.image_id_mode)
    print(f"root:                 {a.root}")
    print(f"format:               {info['format']}")
    print(f"images:               {info['images']}  (without boxes: {info['images_without_boxes']})")
    print(f"boxes:                {info['boxes']}")
    print(f"image sizes (w,h):    {info['image_sizes']}")
    print(f"box width  (px):      {_pct(info['w'])}")
    print(f"box height (px):      {_pct(info['h'])}")
    if info["boxes"]:
        print(f"box area   (px^2):    {_pct(info['w'] * info['h'])}")
    print(f"classes with boxes:   {len(info['class_counts'])}")
    for name, n in info["class_counts"].items():
        print(f"  {name:<24s}{n:>8d}")
    if "classes_in_list_not_in_data" in info:
        print(f"in class list, absent from data: {info['classes_in_list_not_in_data']}")
        print(f"in data, absent from class list: {info['classes_in_data_not_in_list']}")


if __name__ == "__main__":
    sys.exit(main())
