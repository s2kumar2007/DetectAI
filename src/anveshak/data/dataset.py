"""Detection dataset: letterbox to a square canvas, deterministic per-(seed, epoch, index) augmentation.

Training augmentations: horizontal / vertical flips, 90/180/270 rotations, scale jitter, mosaic and colour jitter.
Targets use *slot ids* as labels; objects whose class has no slot (held-out / unlabelled) are simply absent, so they are
unlabelled objects - never background labels.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .records import ImageRecord, load_image


# --------------------------------------------------------------------------- geometry helpers (HWC uint8 + xyxy)
def letterbox(img: np.ndarray, boxes: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Resize so the long side == size, pad bottom/right with zeros to size x size. Returns (img, boxes, scale)."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    res = cv2.resize(img, (nw, nh), interpolation=interp) if (nh, nw) != (h, w) else img
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:nh, :nw] = res
    return canvas, boxes.astype(np.float32) * np.float32(s), s


def hflip(img, boxes):
    w = img.shape[1]
    b = boxes.copy()
    b[:, [0, 2]] = w - boxes[:, [2, 0]]
    return np.ascontiguousarray(img[:, ::-1]), b


def vflip(img, boxes):
    h = img.shape[0]
    b = boxes.copy()
    b[:, [1, 3]] = h - boxes[:, [3, 1]]
    return np.ascontiguousarray(img[::-1]), b


def rot90(img, boxes, k: int):
    """Rotate counter-clockwise k*90 degrees (same convention as torch.rot90 dims=(H,W))."""
    for _ in range(k % 4):
        w = img.shape[1]
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        boxes = np.stack([y1, w - x2, y2, w - x1], axis=1).astype(np.float32) if len(boxes) else boxes
        img = np.ascontiguousarray(np.rot90(img, 1))
    return img, boxes


def _clip_filter(boxes, labels, area0, size_hw, min_vis=0.4, min_side=2.0):
    h, w = size_hw
    if len(boxes) == 0:
        return boxes, labels
    b = boxes.copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    keep = (bw >= min_side) & (bh >= min_side) & (bw * bh >= min_vis * area0)
    return b[keep], labels[keep]


def scale_jitter(img, boxes, labels, f: float, rng: np.random.Generator, min_vis: float = 0.4):
    """Scale the canvas by f then random crop / pad back to the original square size."""
    S = img.shape[0]
    n = max(1, int(round(S * f)))
    res = cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
    area0 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * (n / S) ** 2 if len(boxes) else np.zeros(0)
    b = boxes * (n / S)
    canvas = np.zeros_like(img)
    if n >= S:
        ox, oy = int(rng.integers(0, n - S + 1)), int(rng.integers(0, n - S + 1))
        canvas[:] = res[oy : oy + S, ox : ox + S]
        b = b - np.array([ox, oy, ox, oy], dtype=np.float32)
    else:
        ox, oy = int(rng.integers(0, S - n + 1)), int(rng.integers(0, S - n + 1))
        canvas[oy : oy + n, ox : ox + n] = res
        b = b + np.array([ox, oy, ox, oy], dtype=np.float32)
    b, l = _clip_filter(b, labels, area0, (S, S), min_vis=min_vis)
    return canvas, b.astype(np.float32), l


def native_crop(
    img: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    crop_size: int,
    rng: np.random.Generator,
    min_vis: float = 0.6,
    slot_rarity: Optional[Mapping[int, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random crop at native pixel resolution (no downscaling), biased toward rare-class boxes."""
    h, w = img.shape[:2]
    cs = int(crop_size)
    if len(boxes) > 0 and rng.random() < 0.75:
        if slot_rarity:
            bw = np.array([float(slot_rarity.get(int(lb), 1.0)) for lb in labels], dtype=np.float64)
            bw = bw / bw.sum()
            bi = int(rng.choice(len(boxes), p=bw))
        else:
            bi = int(rng.integers(0, len(boxes)))
        cx = float((boxes[bi, 0] + boxes[bi, 2]) * 0.5 + rng.uniform(-cs * 0.3, cs * 0.3))
        cy = float((boxes[bi, 1] + boxes[bi, 3]) * 0.5 + rng.uniform(-cs * 0.3, cs * 0.3))
        ox = int(round(max(0.0, min(float(max(0, w - cs)), cx - cs * 0.5))))
        oy = int(round(max(0.0, min(float(max(0, h - cs)), cy - cs * 0.5))))
    else:
        ox = int(rng.integers(0, max(1, w - cs + 1))) if w > cs else 0
        oy = int(rng.integers(0, max(1, h - cs + 1))) if h > cs else 0
    canvas = np.zeros((cs, cs, 3), dtype=np.uint8)
    cw, ch = min(cs, w - ox), min(cs, h - oy)
    canvas[:ch, :cw] = img[oy : oy + ch, ox : ox + cw]
    if len(boxes):
        area0 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        b = boxes.astype(np.float32) - np.array([ox, oy, ox, oy], dtype=np.float32)
        b, l = _clip_filter(b, labels, area0, (cs, cs), min_vis=min_vis)
    else:
        b, l = np.zeros((0, 4), dtype=np.float32), labels
    return canvas, b.astype(np.float32), l


def color_jitter(img: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    x = img.astype(np.float32)
    x = x * (1 + rng.uniform(-strength, strength)) + 255 * rng.uniform(-strength, strength) * 0.25  # brightness
    m = x.mean()
    x = (x - m) * (1 + rng.uniform(-strength, strength)) + m  # contrast
    gray = x.mean(axis=2, keepdims=True)
    x = gray + (x - gray) * (1 + rng.uniform(-strength, strength))  # saturation
    x = x * (1 + rng.uniform(-strength / 2, strength / 2, size=3))  # per-channel gain
    return np.clip(x, 0, 255).astype(np.uint8)


def mosaic4(parts, rng: np.random.Generator):
    """parts: 4 x (img SxS, boxes, labels). Place in a 2S canvas and crop a random SxS window around the centre."""
    S = parts[0][0].shape[0]
    canvas = np.zeros((2 * S, 2 * S, 3), dtype=np.uint8)
    all_b, all_l, all_a = [], [], []
    for q, (im, b, l) in enumerate(parts):
        ox, oy = (q % 2) * S, (q // 2) * S
        canvas[oy : oy + S, ox : ox + S] = im
        shift = np.array([ox, oy, ox, oy], dtype=np.float32)
        all_b.append(b + shift)
        all_l.append(l)
        all_a.append((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) if len(b) else np.zeros(0))
    wx, wy = int(rng.integers(S // 4, 3 * S // 4 + 1)), int(rng.integers(S // 4, 3 * S // 4 + 1))
    x0, y0 = S - wx, S - wy
    win = canvas[y0 : y0 + S, x0 : x0 + S].copy()
    b = np.concatenate(all_b) - np.array([x0, y0, x0, y0], dtype=np.float32)
    l = np.concatenate(all_l)
    a = np.concatenate(all_a)
    b, l = _clip_filter(b, l, a, (S, S), min_vis=0.3)
    return win, b.astype(np.float32), l


def _default_aug(aug) -> dict:
    d = {"hflip": 0.5, "vflip": 0.5, "rot90": True, "scale_jitter": [0.75, 1.25], "min_vis": 0.4, "mosaic": 0.0, "color_jitter": 0.2, "color_prob": 0.8}
    if aug:
        d.update({k: aug[k] for k in aug})
    return d


class DetDataset(Dataset):
    """Items: (uint8 CHW tensor SxS, target dict). ``labels`` are slot ids; unmapped classes are dropped."""

    def __init__(
        self,
        records: Sequence[ImageRecord],
        class_to_slot: Mapping[str, int],
        size: int,
        train: bool = False,
        aug=None,
        seed: int = 0,
        repeat: int = 1,
        input_mode: str = "letterbox",
        crop_size: Optional[int] = None,
        slot_rarity: Optional[Mapping[int, float]] = None,
    ):
        self.records = list(records)
        self.class_to_slot = dict(class_to_slot)
        self.size, self.train, self.seed, self.repeat = int(size), bool(train), int(seed), max(1, int(repeat))
        self.aug = _default_aug(aug)
        self.input_mode = str(input_mode)
        self.crop_size = int(crop_size) if crop_size else None
        self.slot_rarity = dict(slot_rarity) if slot_rarity else None
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records) * self.repeat

    def _load(self, rec: ImageRecord, rng: Optional[np.random.Generator] = None):
        img = load_image(rec.path)
        keep = [i for i, l in enumerate(rec.labels) if l in self.class_to_slot]
        boxes = rec.boxes[keep] if keep else np.zeros((0, 4), np.float32)
        labels = np.array([self.class_to_slot[rec.labels[i]] for i in keep], dtype=np.int64)
        if self.train and self.input_mode == "crop" and self.crop_size:
            r = rng if rng is not None else np.random.default_rng(self.seed)
            img, boxes, labels = native_crop(
                img, boxes, labels, self.crop_size, r,
                min_vis=float(self.aug.get("min_vis", 0.6)),
                slot_rarity=self.slot_rarity,
            )
            return img, boxes, labels, 1.0
        img, boxes, s = letterbox(img, boxes, self.size)
        return img, boxes, labels, s

    def _augment(self, img, boxes, labels, rng):
        a = self.aug
        if rng.random() < a["hflip"]:
            img, boxes = hflip(img, boxes)
        if rng.random() < a["vflip"]:
            img, boxes = vflip(img, boxes)
        if a["rot90"]:
            img, boxes = rot90(img, boxes, int(rng.integers(0, 4)))
        lo, hi = a["scale_jitter"]
        if hi > lo or lo != 1.0:
            img, boxes, labels = scale_jitter(img, boxes, labels, float(rng.uniform(lo, hi)), rng, min_vis=float(a.get("min_vis", 0.4)))
        if a["color_jitter"] > 0 and rng.random() < a["color_prob"]:
            img = color_jitter(img, float(a["color_jitter"]), rng)
        return img, boxes, labels

    def __getitem__(self, index: int):
        idx = index % len(self.records)
        rec = self.records[idx]
        if self.train:
            rng = np.random.default_rng([self.seed, self.epoch, index])
            img, boxes, labels, s = self._load(rec, rng)
            if self.aug["mosaic"] > 0 and rng.random() < self.aug["mosaic"] and len(self.records) >= 4:
                parts = [(img, boxes, labels)]
                for j in rng.integers(0, len(self.records), size=3):
                    im2, b2, l2, _ = self._load(self.records[int(j)], rng)
                    parts.append((im2, b2, l2))
                img, boxes, labels = mosaic4(parts, rng)
            img, boxes, labels = self._augment(img, boxes, labels, rng)
        else:
            img, boxes, labels, s = self._load(rec)
        target = {
            "boxes": torch.from_numpy(np.ascontiguousarray(boxes, dtype=np.float32)).reshape(-1, 4),
            "labels": torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64)),
            "image_id": rec.image_id,
            "scale": float(s),
            "orig_size": (rec.width, rec.height),
            "index": int(index),
        }
        return torch.from_numpy(img).permute(2, 0, 1).contiguous(), target


def collate(batch):
    imgs, tgts = zip(*batch)
    return list(imgs), list(tgts)
