"""Annotation record types and image IO shared by all annotation formats."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class ImageRecord:
    """One image with (possibly partial) xyxy pixel-coordinate annotations."""

    image_id: str
    path: str
    width: int
    height: int
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    labels: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)
        if len(self.labels) != len(self.boxes):
            raise ValueError(f"{self.image_id}: {len(self.labels)} labels vs {len(self.boxes)} boxes")

    def subset(self, keep: Sequence[bool]) -> "ImageRecord":
        keep_arr = np.asarray(keep, dtype=bool)
        return ImageRecord(
            self.image_id,
            self.path,
            self.width,
            self.height,
            self.boxes[keep_arr],
            [l for l, k in zip(self.labels, keep_arr) if k],
        )

    def relabel(self, mapping: Dict[str, Optional[str]]) -> "ImageRecord":
        """Rename labels via mapping (missing keys unchanged; value None drops the box)."""
        keep, new = [], []
        for lab in self.labels:
            tgt = mapping.get(lab, lab)
            keep.append(tgt is not None)
            if tgt is not None:
                new.append(tgt)
        return ImageRecord(self.image_id, self.path, self.width, self.height, self.boxes[np.asarray(keep, dtype=bool)], new)


def image_size(path: str | Path) -> tuple[int, int]:
    """(width, height) without decoding pixels."""
    with Image.open(path) as im:
        return int(im.size[0]), int(im.size[1])


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as uint8 HxWx3 RGB (handles 16-bit / float / palette / gray / RGBA / multi-band TIFF)."""
    with Image.open(path) as im:
        mode = im.mode
        if mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            arr = np.asarray(im, dtype=np.float32)
            lo, hi = float(arr.min()), float(arr.max())
            arr = (arr - lo) / max(hi - lo, 1e-6) * 255.0
            arr = np.repeat(arr.astype(np.uint8)[..., None], 3, axis=2)
            return arr
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def list_images(root: str | Path) -> List[Path]:
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() in IMG_EXTS else []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def make_image_id(path: str | Path, mode: str = "stem") -> str:
    p = Path(path)
    return p.name if mode == "filename" else p.stem
