"""Assemble stage data: class names, annotation records, validation split, slot allocation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..utils.log import get_logger
from .classes import load_class_list
from .formats import load_annotations, load_image_dir
from .records import ImageRecord
from .slots import SlotMap
from .split import load_split, make_split, train_view, val_view

log = get_logger("anveshak.data")


def resolve(cfg, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else Path(cfg.data.root) / q


def load_class_names(cfg) -> Optional[List[str]]:
    path = resolve(cfg, cfg.data.class_list)
    if not path.exists():
        log.warning("class list %s not found - classes will be derived from annotations (sorted)", path)
        return None
    return load_class_list(path)


def load_set(cfg, sub: str, class_names: Optional[Sequence[str]]) -> List[ImageRecord]:
    root = resolve(cfg, sub)
    recs = load_annotations(root, cfg.data.format, class_names, cfg.data.image_id_mode)
    log.info("loaded %d images from %s", len(recs), root)
    return recs


def load_test_images(cfg) -> List[ImageRecord]:
    return load_image_dir(resolve(cfg, cfg.data.test), cfg.data.image_id_mode)


def ordered_classes(records: Sequence[ImageRecord], class_names: Optional[Sequence[str]]) -> List[str]:
    """Classes present in ``records`` in class-list order; classes absent from the list are appended (sorted) with a warning."""
    present = sorted({l for r in records for l in r.labels})
    if class_names is None:
        return present
    listed = [c for c in class_names if c in set(present)]
    extra = [c for c in present if c not in set(class_names)]
    if extra:
        log.warning("annotation classes not in the class list (kept, appended): %s", extra)
    return listed + extra


def get_split(cfg, records_a: Sequence[ImageRecord], recorded: Optional[dict] = None) -> dict:
    """Priority: explicit split file > split recorded in the previous checkpoint > deterministic automatic image split."""
    if cfg.data.split_file:
        sp = load_split(cfg.data.split_file)
        log.info("using split file %s (held-out fake unknowns: %s)", cfg.data.split_file, sp["heldout_classes"])
        return sp
    if recorded is not None:
        return recorded
    sp = make_split(records_a, n_holdout=0, val_frac=float(cfg.eval.val_frac), seed=int(cfg.seed))
    log.info("auto validation split: %d train / %d val images", len(sp["train_ids"]), len(sp["val_ids"]))
    return sp


def drop_heldout(records: Sequence[ImageRecord], split: dict) -> List[ImageRecord]:
    held = {h: None for h in split["heldout_classes"]}
    return [r.relabel(held) if held else r for r in records]
