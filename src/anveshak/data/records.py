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


def _percentile_stretch_u8(arr: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    """Scale a 2D or HxWxC numeric array to uint8 using a robust percentile stretch (fallback to min/max)."""
    f = arr.astype(np.float32)
    if f.ndim == 2:
        lo, hi = np.percentile(f, (lo_pct, hi_pct))
        if hi - lo < 1e-6:
            lo, hi = float(f.min()), float(f.max())
        if hi - lo < 1e-6:
            return np.zeros(f.shape, dtype=np.uint8)
        return (np.clip((f - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)
    out = np.empty(f.shape, dtype=np.uint8)
    for c in range(f.shape[2]):
        out[:, :, c] = _percentile_stretch_u8(f[:, :, c], lo_pct, hi_pct)
    return out


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as uint8 HxWx3 RGB (handles 16-bit / float / palette / gray / RGBA / multi-band TIFF)."""
    with Image.open(path) as im:
        if im.mode in ("P", "1"):
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
        arr = np.asarray(im)
        if arr.ndim == 2 and getattr(im, "n_frames", 1) >= 3:
            bands = []
            for fi in range(3):
                im.seek(fi)
                bands.append(np.asarray(im))
            arr = np.stack(bands, axis=2)
        if arr.ndim == 2:
            if arr.dtype != np.uint8:
                arr = _percentile_stretch_u8(arr)
            return np.repeat(arr[..., None], 3, axis=2)
        if arr.ndim == 3:
            if arr.shape[2] in (1, 2):
                ch = arr[:, :, 0]
                if ch.dtype != np.uint8:
                    ch = _percentile_stretch_u8(ch)
                return np.repeat(ch[..., None], 3, axis=2)
            rgb = arr[:, :, :3]
            if rgb.dtype != np.uint8:
                rgb = _percentile_stretch_u8(rgb)
            return np.ascontiguousarray(rgb, dtype=np.uint8)
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def list_images(root: str | Path) -> List[Path]:
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() in IMG_EXTS else []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def make_image_id(path: str | Path, mode: str = "stem") -> str:
    p = Path(path)
    return p.name if mode == "filename" else p.stem
