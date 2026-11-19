"""SYNTHETIC remote-sensing-like data generator (coloured shapes on textured backgrounds).

Used for ALL tests and smoke runs - the real Anveshak data is never available at build time.

Layout written under ``out_dir``::

    class_list.txt             known class codes: base (A) + few-shot B + Round-2 classes (NOT the unknown ones)
    set_a/                     COCO   (images/ + annotations.json)   base classes labelled
    set_b/                     YOLO   (images/ + labels/)            few-shot classes labelled, base objects UNLABELLED
    round2/                    VOC    (images/ + annotations/*.xml)  round-2 classes labelled, base/B objects UNLABELLED
    test/images/               unlabelled test tiles
    test_gt_round1.csv         hidden GT for local scoring (known = A+B; everything else "unknown")
    test_gt_round2.csv         hidden GT (known = A+B+R2; held-back shapes are "unknown")
    meta.json                  roles of every class

Some classes are *held back as unlabelled unknowns*: they appear in images but in no annotation file
and not in the class list.
"""
from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

SHAPES = ("rect", "ellipse", "triangle", "cross", "diamond", "ring")
COLORS = [(225, 40, 40), (40, 70, 235), (40, 205, 60), (245, 225, 40), (225, 50, 205), (40, 215, 225)]


def class_code(i: int) -> str:
    return f"SYN{i + 1:02d}"


@dataclass
class SyntheticSpec:
    size: int = 128
    n_a: int = 96
    n_b: int = 12
    n_r2: int = 16
    n_test: int = 40
    base_ids: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    b_ids: Tuple[int, ...] = (6, 7)
    r2_ids: Tuple[int, ...] = (8, 9)
    unknown_ids: Tuple[int, ...] = (10, 11)
    fmt_a: str = "coco"
    fmt_b: str = "yolo"
    fmt_r2: str = "voc"
    seed: int = 0
    max_objects: int = 4

    @property
    def n_classes(self) -> int:
        return len(self.base_ids) + len(self.b_ids) + len(self.r2_ids) + len(self.unknown_ids)


def _class_style(i: int) -> Tuple[str, Tuple[int, int, int]]:
    shape = SHAPES[i % 6]
    color = COLORS[i % 6] if i < 6 else COLORS[(i + 3) % 6]
    return shape, color


def make_background(rng: np.random.Generator, size: int) -> np.ndarray:
    base = rng.integers(80, 140, size=3).astype(np.float32)
    low = rng.normal(0, 14, size=(8, 8, 3)).astype(np.float32)
    low = np.asarray(Image.fromarray(np.clip(low + 128, 0, 255).astype(np.uint8)).resize((size, size), Image.BICUBIC), dtype=np.float32) - 128
    grad = np.linspace(-10, 10, size, dtype=np.float32)[None, :, None] * rng.choice([-1, 1])
    noise = rng.normal(0, 6, size=(size, size, 3)).astype(np.float32)
    img = base[None, None, :] + low + grad + noise
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_object(draw: ImageDraw.ImageDraw, shape: str, color: Tuple[int, int, int], box: Tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    if shape == "rect":
        draw.rectangle([x1, y1, x2 - 1, y2 - 1], fill=color)
    elif shape == "ellipse":
        draw.ellipse([x1, y1, x2 - 1, y2 - 1], fill=color)
    elif shape == "triangle":
        draw.polygon([(cx, y1), (x1, y2 - 1), (x2 - 1, y2 - 1)], fill=color)
    elif shape == "diamond":
        draw.polygon([(cx, y1), (x2 - 1, cy), (cx, y2 - 1), (x1, cy)], fill=color)
    elif shape == "cross":
        t = max((x2 - x1) // 3, 2)
        draw.rectangle([cx - t // 2, y1, cx + t // 2, y2 - 1], fill=color)
        draw.rectangle([x1, cy - t // 2, x2 - 1, cy + t // 2], fill=color)
    elif shape == "ring":
        w = max((x2 - x1) // 4, 2)
        draw.ellipse([x1, y1, x2 - 1, y2 - 1], outline=color, width=w)
    else:  # pragma: no cover
        raise ValueError(shape)


def _place(rng: np.random.Generator, size: int, taken: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    lo, hi = max(int(size * 0.16), 10), max(int(size * 0.34), 16)
    for _ in range(40):
        w = int(rng.integers(lo, hi + 1))
        h = int(np.clip(w * rng.uniform(0.7, 1.3), lo, hi))
        x1 = int(rng.integers(2, size - w - 2))
        y1 = int(rng.integers(2, size - h - 2))
        box = (x1, y1, x1 + w, y1 + h)
        if all(box[2] + 3 < t[0] or box[0] > t[2] + 3 or box[3] + 3 < t[1] or box[1] > t[3] + 3 for t in taken):
            return box
    return None


def render_scene(
    rng: np.random.Generator, size: int, class_pool: Sequence[int], n_objects: int
) -> Tuple[np.ndarray, List[Tuple[int, Tuple[int, int, int, int]]]]:
    img = Image.fromarray(make_background(rng, size))
    draw = ImageDraw.Draw(img)
    taken: List[Tuple[int, int, int, int]] = []
    objs: List[Tuple[int, Tuple[int, int, int, int]]] = []
    for _ in range(n_objects):
        box = _place(rng, size, taken)
        if box is None:
            break
        cid = int(rng.choice(class_pool))
        shape, color = _class_style(cid)
        jitter = rng.integers(-18, 19, size=3)
        col = tuple(int(np.clip(c + j, 0, 255)) for c, j in zip(color, jitter))
        draw_object(draw, shape, col, box)
        taken.append(box)
        objs.append((cid, box))
    return np.asarray(img), objs


# ----------------------------------------------------------------------------- writers
def _write_coco(out: Path, items: List[Tuple[str, np.ndarray, List[Tuple[str, Tuple[int, int, int, int]]]]], names: Sequence[str]) -> None:
    (out / "images").mkdir(parents=True, exist_ok=True)
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
    cid = {n: i + 1 for i, n in enumerate(names)}
    images, anns = [], []
    for k, (stem, arr, objs) in enumerate(items):
        Image.fromarray(arr).save(out / "images" / f"{stem}.png")
        images.append({"id": k + 1, "file_name": f"images/{stem}.png", "width": arr.shape[1], "height": arr.shape[0]})
        for name, (x1, y1, x2, y2) in objs:
            anns.append({"id": len(anns) + 1, "image_id": k + 1, "category_id": cid[name], "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
    (out / "annotations.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}), encoding="utf-8")


def _write_yolo(out: Path, items, names: Sequence[str]) -> None:
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    idx = {n: i for i, n in enumerate(names)}
    for stem, arr, objs in items:
        Image.fromarray(arr).save(out / "images" / f"{stem}.png")
        h, w = arr.shape[:2]
        lines = []
        for name, (x1, y1, x2, y2) in objs:
            lines.append(f"{idx[name]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_voc(out: Path, items, names: Sequence[str]) -> None:
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    for stem, arr, objs in items:
        Image.fromarray(arr).save(out / "images" / f"{stem}.png")
        h, w = arr.shape[:2]
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = f"{stem}.png"
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(w)
        ET.SubElement(size, "height").text = str(h)
        ET.SubElement(size, "depth").text = "3"
        for name, (x1, y1, x2, y2) in objs:
            o = ET.SubElement(root, "object")
            ET.SubElement(o, "name").text = name
            bb = ET.SubElement(o, "bndbox")
            for k, v in zip(("xmin", "ymin", "xmax", "ymax"), (x1, y1, x2, y2)):
                ET.SubElement(bb, k).text = str(int(v))
        ET.ElementTree(root).write(out / "annotations" / f"{stem}.xml", encoding="utf-8")


def _write_csv(out: Path, items, names: Sequence[str]) -> None:
    (out / "images").mkdir(parents=True, exist_ok=True)
    with open(out / "annotations.csv", "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["image_id", "class_name", "xmin", "ymin", "xmax", "ymax"])
        for stem, arr, objs in items:
            Image.fromarray(arr).save(out / "images" / f"{stem}.png")
            for name, (x1, y1, x2, y2) in objs:
                wr.writerow([stem, name, x1, y1, x2, y2])


_WRITERS = {"coco": _write_coco, "yolo": _write_yolo, "voc": _write_voc, "csv": _write_csv}


# ----------------------------------------------------------------------------- main entry
def generate_dataset(out_dir: str | Path, spec: Optional[SyntheticSpec] = None) -> Dict[str, object]:
    """Write the synthetic dataset and return a dict with class roles."""
    spec = spec or SyntheticSpec()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec.seed)
    S = spec.size
    base, bset, r2, unk = list(spec.base_ids), list(spec.b_ids), list(spec.r2_ids), list(spec.unknown_ids)
    known_order = base + bset + r2
    class_list = [class_code(i) for i in known_order]
    (out / "class_list.txt").write_text("\n".join(class_list) + "\n", encoding="utf-8")

    def labelled_scene(label_ids: Sequence[int], unlabelled_pool: Sequence[int], p_unl: float):
        n_obj = int(rng.integers(1, spec.max_objects + 1))
        pool = list(label_ids)
        # draw objects: mostly labelled classes, sometimes unlabelled ones (old / unknown)
        ids_mask = rng.random(n_obj) < p_unl
        classes = [int(rng.choice(unlabelled_pool)) if (m and unlabelled_pool) else int(rng.choice(pool)) for m in ids_mask]
        img = Image.fromarray(make_background(rng, S))
        draw = ImageDraw.Draw(img)
        taken, labelled, drawn = [], [], []
        for cid in classes:
            box = _place(rng, S, taken)
            if box is None:
                break
            shape, color = _class_style(cid)
            col = tuple(int(np.clip(c + j, 0, 255)) for c, j in zip(color, rng.integers(-18, 19, size=3)))
            draw_object(draw, shape, col, box)
            taken.append(box)
            drawn.append((cid, box))
            if cid in label_ids:
                labelled.append((class_code(cid), box))
        return np.asarray(img), labelled, drawn

    # ---- Set A: base classes labelled; unknown shapes (and B/R2 shapes) appear unlabelled
    items_a = []
    for k in range(spec.n_a):
        arr, lab, _ = labelled_scene(base, unk + bset + r2, p_unl=0.18)
        items_a.append((f"a_{k:04d}", arr, lab))
    _WRITERS[spec.fmt_a](out / "set_a", items_a, [class_code(i) for i in base])

    # ---- Set B: few-shot B classes labelled; base + unknown objects present but UNLABELLED
    items_b = []
    for k in range(spec.n_b):
        arr, lab, _ = labelled_scene(bset, base + unk, p_unl=0.35)
        if not lab:  # guarantee at least one labelled few-shot object
            arr, lab, _ = labelled_scene(bset, [], p_unl=0.0)
        items_b.append((f"b_{k:04d}", arr, lab))
    _WRITERS[spec.fmt_b](out / "set_b", items_b, class_list)

    # ---- Round 2: R2 classes labelled; base + B + unknown objects present but UNLABELLED
    items_r = []
    for k in range(spec.n_r2):
        arr, lab, _ = labelled_scene(r2, base + bset + unk, p_unl=0.40)
        if not lab:
            arr, lab, _ = labelled_scene(r2, [], p_unl=0.0)
        items_r.append((f"r_{k:04d}", arr, lab))
    _WRITERS[spec.fmt_r2](out / "round2", items_r, class_list)

    # ---- test tiles + hidden GT
    (out / "test" / "images").mkdir(parents=True, exist_ok=True)
    rows1, rows2 = [], []
    everything = base + bset + r2 + unk
    for k in range(spec.n_test):
        arr, objs = render_scene(rng, S, everything, int(rng.integers(1, spec.max_objects + 1)))
        stem = f"t_{k:04d}"
        Image.fromarray(arr).save(out / "test" / "images" / f"{stem}.png")
        for cid, (x1, y1, x2, y2) in objs:
            code = class_code(cid)
            n1 = code if cid in base + bset else "unknown"
            n2 = code if cid in base + bset + r2 else "unknown"
            rows1.append([stem, n1, x1, y1, x2, y2])
            rows2.append([stem, n2, x1, y1, x2, y2])
    for name, rows in (("test_gt_round1.csv", rows1), ("test_gt_round2.csv", rows2)):
        with open(out / name, "w", encoding="utf-8", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["image_id", "class_name", "xmin", "ymin", "xmax", "ymax"])
            wr.writerows(rows)

    meta = {
        "size": S,
        "base": [class_code(i) for i in base],
        "few_shot_b": [class_code(i) for i in bset],
        "round2": [class_code(i) for i in r2],
        "unknown_heldback": [class_code(i) for i in unk],
        "class_list": class_list,
        "formats": {"set_a": spec.fmt_a, "set_b": spec.fmt_b, "round2": spec.fmt_r2},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
