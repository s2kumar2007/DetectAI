"""Annotation format auto-detection and parsers: COCO JSON, YOLO txt, Pascal VOC XML, CSV.

Everything is converted to :class:`ImageRecord` (xyxy pixel coordinates, string class codes).
Detection is by *inspecting files* under a directory; ``fmt`` can force a format.
"""
from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .records import IMG_EXTS, ImageRecord, image_size, list_images, make_image_id
from ..utils.log import get_logger

log = get_logger("anveshak.data")

FORMATS = ("coco", "yolo", "voc", "csv")
_CSV_ALIASES = {
    "image": ("image_id", "filename", "file_name", "image", "image_name", "img", "file", "path", "image_path"),
    "cls": ("class_name", "class", "label", "category", "class_code", "category_name", "name"),
    "xmin": ("xmin", "x_min", "x1", "left"),
    "ymin": ("ymin", "y_min", "y1", "top"),
    "xmax": ("xmax", "x_max", "x2", "right"),
    "ymax": ("ymax", "y_max", "y2", "bottom"),
    "x": ("x", "bbox_x"),
    "y": ("y", "bbox_y"),
    "w": ("w", "width", "bbox_w", "bbox_width"),
    "h": ("h", "height", "bbox_h", "bbox_height"),
    "img_w": ("image_width", "img_width", "img_w", "im_width"),
    "img_h": ("image_height", "img_height", "img_h", "im_height"),
}


# ----------------------------------------------------------------------------- helpers
class _ImageIndex:
    """Lookup of images under a root by relative path, basename or stem."""

    def __init__(self, root: Path):
        self.root = root
        self.paths = list_images(root)
        self.by_rel: Dict[str, Path] = {p.relative_to(root).as_posix(): p for p in self.paths} if root.is_dir() else {}
        self.by_name: Dict[str, List[Path]] = defaultdict(list)
        self.by_stem: Dict[str, List[Path]] = defaultdict(list)
        for p in self.paths:
            self.by_name[p.name].append(p)
            self.by_stem[p.stem].append(p)

    def find(self, ref: str, hint_dir: Optional[Path] = None) -> Optional[Path]:
        ref = str(ref).replace("\\", "/")
        if ref in self.by_rel:
            return self.by_rel[ref]
        cand = self.root / ref
        if cand.is_file():
            return cand
        if hint_dir is not None and (hint_dir / ref).is_file():
            return hint_dir / ref
        name = Path(ref).name
        hits = self.by_name.get(name) or self.by_stem.get(Path(ref).stem) or []
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            if hint_dir is not None:
                near = [h for h in hits if hint_dir in h.parents]
                if near:
                    return sorted(near)[0]
            return sorted(hits)[0]
        return None


def _map_class(name: object, class_list: Optional[Sequence[str]]) -> str:
    s = str(name).strip()
    if class_list and s not in class_list and s.lstrip("-").isdigit():
        i = int(s)
        if 0 <= i < len(class_list):
            return class_list[i]
    return s


def _sniff_json(path: Path) -> Optional[dict]:
    try:
        if path.stat().st_size > 2_000_000_000:
            return None
        with open(path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
    except Exception:
        return None
    if isinstance(obj, dict) and "images" in obj and "annotations" in obj:
        return obj
    return None


def _xml_is_voc(path: Path) -> bool:
    try:
        return ET.parse(path).getroot().tag == "annotation"
    except Exception:
        return False


def _csv_columns(path: Path) -> Optional[Dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            header = next(csv.reader(fh), None)
    except Exception:
        return None
    if not header:
        return None
    low = {h.strip().lower(): h for h in header}
    cols: Dict[str, str] = {}
    for key, aliases in _CSV_ALIASES.items():
        for a in aliases:
            if a in low and low[a] not in cols.values():
                cols[key] = low[a]
                break
    has_corners = all(k in cols for k in ("xmin", "ymin", "xmax", "ymax"))
    has_xywh = all(k in cols for k in ("x", "y", "w", "h"))
    if "image" in cols and "cls" in cols and (has_corners or has_xywh):
        if not has_corners:
            for k in ("xmin", "ymin", "xmax", "ymax"):
                cols.pop(k, None)
        return cols
    return None


def _yolo_label_files(root: Path) -> List[Path]:
    out = []
    for p in sorted(root.rglob("*.txt")):
        if p.name.lower() in ("classes.txt", "obj.names", "readme.txt", "class_list.txt"):
            continue
        try:
            with open(p, "r", encoding="utf-8") as fh:
                first = ""
                for line in fh:
                    if line.strip():
                        first = line.strip()
                        break
        except Exception:
            continue
        if first == "":
            out.append(p)  # empty label file = image without objects
            continue
        toks = first.split()
        if len(toks) >= 5:
            try:
                [float(t) for t in toks]
                out.append(p)
            except ValueError:
                continue
    return out


def detect_format(root: str | Path) -> str:
    """Inspect files under ``root`` and return one of coco | voc | csv | yolo. Raises if none match."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    for p in sorted(root.rglob("*.json")):
        if _sniff_json(p) is not None:
            return "coco"
    for p in sorted(root.rglob("*.xml")):
        if _xml_is_voc(p):
            return "voc"
    for p in sorted(root.rglob("*.csv")):
        if _csv_columns(p) is not None:
            return "csv"
    if _yolo_label_files(root):
        return "yolo"
    raise ValueError(f"could not detect an annotation format under {root} (looked for COCO/VOC/CSV/YOLO)")


# ----------------------------------------------------------------------------- parsers
def load_coco(root: Path, class_list: Optional[Sequence[str]], image_id_mode: str) -> List[ImageRecord]:
    index = _ImageIndex(root)
    recs: Dict[str, ImageRecord] = {}
    for jp in sorted(root.rglob("*.json")):
        obj = _sniff_json(jp)
        if obj is None:
            continue
        cats = {c["id"]: str(c.get("name", c["id"])) for c in obj.get("categories", [])}
        per_img: Dict[object, list] = defaultdict(list)
        for ann in obj["annotations"]:
            if ann.get("iscrowd", 0) and ann.get("ignore", 0):
                continue
            per_img[ann["image_id"]].append(ann)
        for im in obj["images"]:
            path = index.find(im["file_name"], hint_dir=jp.parent)
            if path is None:
                log.warning("COCO image not found on disk: %s", im["file_name"])
                continue
            w = int(im.get("width") or 0)
            h = int(im.get("height") or 0)
            if not w or not h:
                w, h = image_size(path)
            boxes, labels = [], []
            for ann in per_img.get(im["id"], []):
                x, y, bw, bh = [float(v) for v in ann["bbox"]]
                if bw <= 0 or bh <= 0:
                    continue
                boxes.append([x, y, x + bw, y + bh])
                labels.append(_map_class(cats.get(ann["category_id"], ann["category_id"]), class_list))
            rec = ImageRecord(make_image_id(path, image_id_mode), str(path), w, h, np.asarray(boxes, dtype=np.float32), labels)
            recs[str(path)] = rec
    return list(recs.values())


def _yolo_names(root: Path) -> Optional[List[str]]:
    for name in ("classes.txt", "obj.names"):
        for p in root.rglob(name):
            lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                return lines
    try:
        import yaml

        for p in root.rglob("*.yaml"):
            d = yaml.safe_load(p.read_text(encoding="utf-8"))
            if isinstance(d, dict) and "names" in d:
                names = d["names"]
                if isinstance(names, dict):
                    return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
                if isinstance(names, list):
                    return [str(n) for n in names]
    except Exception:
        pass
    return None


def load_yolo(root: Path, class_list: Optional[Sequence[str]], image_id_mode: str) -> List[ImageRecord]:
    index = _ImageIndex(root)
    names = list(class_list) if class_list else _yolo_names(root)
    labels_by_stem: Dict[str, List[Path]] = defaultdict(list)
    for lp in _yolo_label_files(root):
        labels_by_stem[lp.stem].append(lp)
    recs: List[ImageRecord] = []
    for img in index.paths:
        cands = labels_by_stem.get(img.stem, [])
        label_path: Optional[Path] = None
        if len(cands) == 1:
            label_path = cands[0]
        elif len(cands) > 1:
            twin = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
            label_path = twin if twin in cands else sorted(cands)[0]
        w, h = image_size(img)
        boxes, labels = [], []
        if label_path is not None:
            for line in label_path.read_text(encoding="utf-8").splitlines():
                t = line.split()
                if len(t) < 5:
                    continue
                cls_tok = t[0]
                vals = [float(v) for v in t[1:]]
                if len(vals) == 4:
                    cx, cy, bw, bh = vals
                    x1, y1, x2, y2 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
                elif len(vals) >= 6 and len(vals) % 2 == 0:  # polygon / OBB -> axis-aligned box
                    xs, ys = np.asarray(vals[0::2]) * w, np.asarray(vals[1::2]) * h
                    x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
                elif len(vals) == 5:  # trailing confidence
                    cx, cy, bw, bh = vals[:4]
                    x1, y1, x2, y2 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
                else:
                    continue
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    continue
                idx = int(float(cls_tok))
                name = names[idx] if names and 0 <= idx < len(names) else f"class_{idx}"
                boxes.append([x1, y1, x2, y2])
                labels.append(name)
        recs.append(ImageRecord(make_image_id(img, image_id_mode), str(img), w, h, np.asarray(boxes, dtype=np.float32), labels))
    return recs


def load_voc(root: Path, class_list: Optional[Sequence[str]], image_id_mode: str) -> List[ImageRecord]:
    index = _ImageIndex(root)
    recs: Dict[str, ImageRecord] = {}
    for xp in sorted(root.rglob("*.xml")):
        try:
            tree = ET.parse(xp).getroot()
        except ET.ParseError:
            continue
        if tree.tag != "annotation":
            continue
        fname = tree.findtext("filename") or xp.stem
        path = index.find(fname, hint_dir=xp.parent) or index.find(xp.stem, hint_dir=xp.parent)
        if path is None:
            twin = [index.by_stem.get(xp.stem, [])]
            path = twin[0][0] if twin[0] else None
        if path is None:
            log.warning("VOC image not found for %s", xp)
            continue
        size = tree.find("size")
        w = int(float(size.findtext("width", "0"))) if size is not None else 0
        h = int(float(size.findtext("height", "0"))) if size is not None else 0
        if not w or not h:
            w, h = image_size(path)
        boxes, labels = [], []
        for obj in tree.findall("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            x1, y1, x2, y2 = (float(bb.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax"))
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(_map_class(obj.findtext("name", "unknown"), class_list))
        recs[str(path)] = ImageRecord(make_image_id(path, image_id_mode), str(path), w, h, np.asarray(boxes, dtype=np.float32), labels)
    return list(recs.values())


def load_csv(root: Path, class_list: Optional[Sequence[str]], image_id_mode: str) -> List[ImageRecord]:
    index = _ImageIndex(root)
    grouped: Dict[str, Dict[str, list]] = {}
    for cp in sorted(root.rglob("*.csv")):
        cols = _csv_columns(cp)
        if cols is None:
            continue
        with open(cp, "r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                path = index.find(row[cols["image"]], hint_dir=cp.parent)
                if path is None:
                    for ext in IMG_EXTS:
                        path = index.find(str(row[cols["image"]]) + ext, hint_dir=cp.parent)
                        if path is not None:
                            break
                if path is None:
                    log.warning("CSV image not found: %s", row[cols["image"]])
                    continue
                if "xmin" in cols:
                    x1, y1, x2, y2 = (float(row[cols[k]]) for k in ("xmin", "ymin", "xmax", "ymax"))
                else:
                    x, y, bw, bh = (float(row[cols[k]]) for k in ("x", "y", "w", "h"))
                    x1, y1, x2, y2 = x, y, x + bw, y + bh
                g = grouped.setdefault(str(path), {"boxes": [], "labels": []})
                if x2 - x1 > 0 and y2 - y1 > 0:
                    g["boxes"].append([x1, y1, x2, y2])
                    g["labels"].append(_map_class(row[cols["cls"]], class_list))
    recs = []
    for p, g in grouped.items():
        w, h = image_size(p)
        recs.append(ImageRecord(make_image_id(p, image_id_mode), p, w, h, np.asarray(g["boxes"], dtype=np.float32), g["labels"]))
    return recs


_LOADERS = {"coco": load_coco, "yolo": load_yolo, "voc": load_voc, "csv": load_csv}


def load_annotations(
    root: str | Path,
    fmt: str = "auto",
    class_list: Optional[Sequence[str]] = None,
    image_id_mode: str = "stem",
) -> List[ImageRecord]:
    """Load every annotated image under ``root`` as a list of ImageRecord sorted by image_id."""
    root = Path(root)
    if fmt == "auto":
        fmt = detect_format(root)
        log.info("detected annotation format %s under %s", fmt, root)
    if fmt not in _LOADERS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")
    recs = _LOADERS[fmt](root, class_list, image_id_mode)
    ids = [r.image_id for r in recs]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})[:5]
        raise ValueError(f"duplicate image ids under {root} (e.g. {dup}); use data.image_id_mode=filename or fix the data")
    recs.sort(key=lambda r: r.image_id)
    return recs


def load_image_dir(root: str | Path, image_id_mode: str = "stem") -> List[ImageRecord]:
    """Unlabelled images (e.g. the test set): records with no boxes."""
    recs = []
    for p in list_images(root):
        w, h = image_size(p)
        recs.append(ImageRecord(make_image_id(p, image_id_mode), str(p), w, h))
    ids = [r.image_id for r in recs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate image ids under {root}; use data.image_id_mode=filename")
    recs.sort(key=lambda r: r.image_id)
    return recs
