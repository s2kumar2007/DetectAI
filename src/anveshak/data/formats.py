"""Annotation format auto-detection and parsers: Tile-JSON, COCO JSON, YOLO txt, Pascal VOC XML, CSV.

Everything is converted to :class:`ImageRecord` (xyxy pixel coordinates, string class codes).
Detection is by *inspecting files* under a directory; ``fmt`` can force a format.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .records import IMG_EXTS, ImageRecord, image_size, list_images, make_image_id
from ..utils.log import get_logger

log = get_logger("anveshak.data")

FORMATS = ("tilejson", "coco", "yolo", "voc", "csv")
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


def _sniff_tilejson(path: Path) -> Optional[dict]:
    try:
        if path.stat().st_size > 50_000_000:
            return None
        with open(path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
    except Exception:
        return None
    if isinstance(obj, dict) and "tile_id" in obj and "annotations" in obj:
        return obj
    return None


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
    """Inspect files under ``root`` and return one of tilejson | coco | voc | csv | yolo. Raises if none match."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    labels_dir = root / "labels"
    if labels_dir.is_dir():
        for p in sorted(labels_dir.glob("*.json")):
            if _sniff_tilejson(p) is not None:
                return "tilejson"
    for p in sorted(root.rglob("*.json")):
        if _sniff_tilejson(p) is not None:
            return "tilejson"
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
    raise ValueError(f"could not detect an annotation format under {root} (looked for Tile-JSON/COCO/VOC/CSV/YOLO)")


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


def _read_tiles_csv(root: Path) -> Dict[str, tuple[int, int]]:
    tc = root / "tiles.csv"
    if not tc.is_file():
        return {}
    out: Dict[str, tuple[int, int]] = {}
    try:
        with open(tc, "r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                tid = (row.get("tile_id") or "").strip()
                if tid:
                    try:
                        out[tid] = (int(float(row.get("width", 0) or 0)), int(float(row.get("height", 0) or 0)))
                    except ValueError:
                        pass
    except Exception:
        pass
    return out


def _tilejson_cache_path(root: Path) -> Path:
    cand = root / ".tilejson_cache.pkl"
    try:
        if cand.exists() or os_writable(root):
            return cand
    except Exception:
        pass
    h = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:16]
    p = Path("runs") / "cache" / f"tilejson_{h}.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def os_writable(d: Path) -> bool:
    import os
    return os.access(str(d), os.W_OK)


def load_tilejson(
    root: Path,
    class_list: Optional[Sequence[str]],
    image_id_mode: str,
    class_id_map: Optional[Dict[str, int]] = None,
) -> List[ImageRecord]:
    """Load per-tile JSON annotations (<root>/labels/<tile_id>.json) matched by stem to <root>/images/<tile_id>.*."""
    img_root = root / "images" if (root / "images").is_dir() else root
    img_paths = list_images(img_root)
    tiles_meta = _read_tiles_csv(root)

    labels_dir = root / "labels"
    json_files = sorted(labels_dir.glob("*.json")) if labels_dir.is_dir() else sorted(root.rglob("*.json"))

    cache_path = _tilejson_cache_path(root)
    entries: Dict[str, tuple] = {}
    if cache_path.is_file():
        try:
            with open(cache_path, "rb") as fh:
                blob = pickle.load(fh)
            if isinstance(blob, dict) and blob.get("version") == 1:
                entries = blob.get("entries", {})
        except Exception:
            entries = {}

    dirty = False
    by_tile_id: Dict[str, tuple] = {}
    for jp in json_files:
        st = jp.stat()
        sig = (st.st_mtime_ns, st.st_size)
        key = str(jp.relative_to(root))
        cached = entries.get(key)
        if cached is not None and cached[:2] == sig:
            _, _, tile_id, w, h, boxes_raw, raw_labels, json_cids = cached
        else:
            obj = _sniff_tilejson(jp)
            if obj is None:
                continue
            tile_id = str(obj.get("tile_id") or jp.stem)
            w = int(obj.get("width") or 0)
            h = int(obj.get("height") or 0)
            bbox_fmt = str(obj.get("bbox_format") or "xyxy").lower()
            if bbox_fmt not in ("xyxy", "xywh"):
                raise ValueError(f"unsupported bbox_format {bbox_fmt!r} in {jp}")
            b_list, l_list, cid_list = [], [], []
            for ann in obj.get("annotations", []):
                bb = ann.get("bbox")
                if not bb or len(bb) < 4:
                    continue
                x1, y1, v3, v4 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
                if bbox_fmt == "xywh":
                    x2, y2 = x1 + v3, y1 + v4
                else:
                    x2, y2 = v3, v4
                lab = str(ann.get("label", "unknown")).strip()
                cid = ann.get("class_id")
                cid_int = int(cid) if cid is not None else None
                b_list.append([x1, y1, x2, y2])
                l_list.append(lab)
                cid_list.append(cid_int)
            boxes_raw = np.asarray(b_list, dtype=np.float32).reshape(-1, 4) if b_list else np.zeros((0, 4), dtype=np.float32)
            raw_labels = l_list
            json_cids = cid_list
            entries[key] = (st.st_mtime_ns, st.st_size, tile_id, w, h, boxes_raw, raw_labels, json_cids)
            dirty = True
        by_tile_id[tile_id] = (w, h, boxes_raw, raw_labels, json_cids)

    if dirty:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as fh:
                pickle.dump({"version": 1, "entries": entries}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            log.warning("could not write tilejson cache %s: %s", cache_path, exc)

    expected_id_map = dict(class_id_map) if class_id_map is not None else ({c: i for i, c in enumerate(class_list)} if class_list else {})
    class_set = set(class_list) if class_list is not None else None
    missing_labels: set[str] = set()
    id_mismatches: List[str] = []

    recs: List[ImageRecord] = []
    for img in img_paths:
        stem = img.stem
        entry = by_tile_id.get(stem)
        if entry is not None:
            w, h, boxes_raw, raw_labels, json_cids = entry
        else:
            w, h = tiles_meta.get(stem, (0, 0))
            boxes_raw = np.zeros((0, 4), dtype=np.float32)
            raw_labels, json_cids = [], []
        if w <= 0 or h <= 0:
            w, h = tiles_meta.get(stem, (0, 0))
        if w <= 0 or h <= 0:
            w, h = image_size(img)

        if len(boxes_raw):
            b = boxes_raw.copy()
            b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0.0, float(w))
            b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0.0, float(h))
            keep_mask = ((b[:, 2] - b[:, 0]) >= 1.0) & ((b[:, 3] - b[:, 1]) >= 1.0)
            b = b[keep_mask]
            labs: List[str] = []
            for keep, lab, cid in zip(keep_mask, raw_labels, json_cids):
                if not keep:
                    continue
                if lab.lower() == "unknown":
                    labs.append("unknown")
                    continue
                if class_set is not None and lab not in class_set:
                    missing_labels.add(lab)
                if cid is not None and lab in expected_id_map and cid != expected_id_map[lab]:
                    if len(id_mismatches) < 10:
                        id_mismatches.append(f"{stem}:{lab}(json={cid},csv={expected_id_map[lab]})")
                labs.append(lab)
        else:
            b = np.zeros((0, 4), dtype=np.float32)
            labs = []
        recs.append(ImageRecord(make_image_id(img, image_id_mode), str(img), int(w), int(h), b, labs))

    if missing_labels:
        log.warning("tilejson labels missing from class_list (%d): %s", len(missing_labels), sorted(missing_labels)[:20])
    if id_mismatches:
        log.warning("tilejson class_id mismatch vs class_list: %s", id_mismatches)
    return recs


_LOADERS = {"tilejson": load_tilejson, "coco": load_coco, "yolo": load_yolo, "voc": load_voc, "csv": load_csv}


def load_annotations(
    root: str | Path,
    fmt: str = "auto",
    class_list: Optional[Sequence[str]] = None,
    image_id_mode: str = "stem",
    class_id_map: Optional[Dict[str, int]] = None,
) -> List[ImageRecord]:
    """Load every annotated image under ``root`` as a list of ImageRecord sorted by image_id."""
    root = Path(root)
    if fmt == "auto":
        fmt = detect_format(root)
        log.info("detected annotation format %s under %s", fmt, root)
    if fmt not in _LOADERS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")
    if fmt == "tilejson":
        recs = load_tilejson(root, class_list, image_id_mode, class_id_map=class_id_map)
    else:
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
