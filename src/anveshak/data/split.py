"""Simulated-unknown validation split.

``make_split`` holds out N known classes as FAKE unknowns: their labels are removed from the training view (so the
objects stay in the images, unlabelled - exactly like real unknowns) and kept in the validation view relabelled as
"unknown". It also selects a validation image subset. Thresholds are tuned on this split only (hard rule 4).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .records import ImageRecord

UNKNOWN = "unknown"


def make_split(
    records: Sequence[ImageRecord],
    n_holdout: int = 6,
    val_frac: float = 0.15,
    seed: int = 0,
    class_list: Optional[Sequence[str]] = None,
    min_val_images_per_heldout: int = 2,
) -> dict:
    rng = np.random.default_rng(seed)
    counts = Counter(l for r in records for l in r.labels)
    img_count = Counter(l for r in records for l in set(r.labels))
    pool = [c for c in (class_list or sorted(counts)) if counts.get(c, 0) > 0]
    if n_holdout > len(pool) - 1 and n_holdout > 0:
        raise ValueError(f"cannot hold out {n_holdout} of {len(pool)} classes (need at least one known class)")
    # prefer classes with enough images so both train and val keep instances
    eligible = [c for c in pool if img_count[c] >= 3] or pool
    if len(eligible) < n_holdout:
        eligible = pool
    heldout = sorted(rng.choice(eligible, size=n_holdout, replace=False).tolist()) if n_holdout > 0 else []

    ids = [r.image_id for r in records]
    n_val = max(1, int(round(val_frac * len(ids))))
    by_class: Dict[str, List[str]] = defaultdict(list)
    for r in records:
        for l in set(r.labels):
            by_class[l].append(r.image_id)
    val: set = set()
    for c in heldout:  # guarantee held-out classes are measurable in validation
        cand = [i for i in by_class[c] if i not in val]
        rng.shuffle(cand)
        need = min(min_val_images_per_heldout, max(len(by_class[c]) - 1, 1))
        val.update(cand[:need])
    for c in sorted(set(pool) - set(heldout)):  # and every other class with >=2 images appears at least once
        if len(by_class[c]) >= 2 and not (set(by_class[c]) & val) and len(val) < n_val:
            val.add(str(rng.choice(by_class[c])))
    rest = [i for i in ids if i not in val]
    rng.shuffle(rest)
    for i in rest:
        if len(val) >= n_val:
            break
        val.add(i)
    val_ids = sorted(val)
    train_ids = sorted(set(ids) - val)
    return {
        "seed": seed,
        "n_holdout": n_holdout,
        "val_frac": val_frac,
        "heldout_classes": heldout,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "class_counts": dict(counts),
    }


def save_split(split: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(split, indent=2), encoding="utf-8")


def load_split(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def train_view(records: Sequence[ImageRecord], split: dict) -> List[ImageRecord]:
    """Training images with held-out class labels REMOVED (objects stay in the image, unlabelled)."""
    ids, held = set(split["train_ids"]), set(split["heldout_classes"])
    return [r.relabel({h: None for h in held}) for r in records if r.image_id in ids]


def val_view(records: Sequence[ImageRecord], split: dict) -> List[ImageRecord]:
    """Validation images with held-out class labels renamed to 'unknown'."""
    ids, held = set(split["val_ids"]), set(split["heldout_classes"])
    return [r.relabel({h: UNKNOWN for h in held}) for r in records if r.image_id in ids]
