"""Build the StageState for each stage (data, slots, model, hooks)."""
from __future__ import annotations

from typing import Dict

import torch

from ..data.build import get_split, load_class_names, load_set, ordered_classes
from ..data.dataset import DetDataset
from ..data.slots import SlotMap
from ..data.split import train_view, val_view
from ..models.assertions import assert_stage1_random_init
from ..models.detector import build_model
from ..utils.log import get_logger
from ..utils.offline import enforce_offline
from ..utils.seed import seed_everything
from .engine import StageState
from .evaluate import known_map
from .sampling import compute_class_loss_weights, compute_repeat_factors

log = get_logger("anveshak.stages")


def stage1_report(state: StageState):
    def fn(model, device) -> Dict:
        names = state.slotmap.names
        if not state.val_records:
            return {"note": "no validation images", "old_classes": []}
        m, aps, skipped = known_map(model, state.val_records, state.slotmap.slot_to_name(), names, int(state.cfg.input.size), device)
        return {
            "old_classes": [], "note": "Stage 1 has no previous classes; reporting validation mAP of the new base classes (EMA weights).",
            "val_map_known": m, "val_ap_per_class": aps, "val_skipped_classes": skipped, "n_val_images": len(state.val_records),
        }
    return fn


def prepare_stage1(cfg) -> StageState:
    """Stage 1: RANDOM INIT only. No checkpoint may be configured; the initial weights hash is printed."""
    enforce_offline()
    seed_everything(int(cfg.seed), deterministic=False)
    class_names = load_class_names(cfg)
    recs = load_set(cfg, cfg.data.train_dir, class_names)
    split = get_split(cfg, recs)
    train_recs, val_recs = train_view(recs, split), val_view(recs, split)
    names = ordered_classes(train_recs, class_names)
    if not names:
        raise ValueError("no labelled classes found in the Stage 1 training set")
    slotmap = SlotMap()
    slotmap.assign(names, "stage1", start=cfg.slots.start)
    assert all(0 <= s < 40 for s in slotmap.active_slots), f"Stage 1 slots must be in [0, 40), got {slotmap.active_slots}"
    log.info("Stage 1: %d images (%d val), %d classes -> slots %s..%s", len(train_recs), len(val_recs), len(names),
             slotmap.active_slots[0], slotmap.active_slots[-1])

    seed_everything(int(cfg.seed), deterministic=False)
    model = build_model(cfg)
    init_hash = assert_stage1_random_init(cfg, model, build_model, int(cfg.seed))  # must run BEFORE slots are activated
    model.activate_slots(slotmap.active_slots)

    sample_weights = None
    slot_rarity = None
    if bool(cfg.train.get("rfs_enable", True)):
        sample_weights, r_cls = compute_repeat_factors(
            train_recs,
            slotmap.class_to_slot,
            t=float(cfg.train.get("rfs_threshold", 0.01)),
            empty_weight=float(cfg.train.get("empty_tile_weight", 0.25)),
        )
        slot_rarity = {slotmap.class_to_slot[c]: float(r) for c, r in r_cls.items() if c in slotmap.class_to_slot}
        cls_w = compute_class_loss_weights(
            train_recs,
            slotmap.class_to_slot,
            int(cfg.model.roi_head.num_slots),
            cap=float(cfg.train.get("class_weight_cap", 5.0)),
            power=float(cfg.train.get("class_weight_power", 0.5)),
        )
        model.roi_head.class_weights.copy_(cls_w)
        log.info("Stage 1 RFS enabled (t=%.3f, empty_w=%.2f, max_r=%.2f, max_cls_w=%.2f)",
                 float(cfg.train.get("rfs_threshold", 0.01)),
                 float(cfg.train.get("empty_tile_weight", 0.25)),
                 max(sample_weights) if sample_weights else 1.0,
                 float(cls_w.max()))
    seed_everything(int(cfg.seed), deterministic=False)

    ds = DetDataset(
        train_recs,
        slotmap.class_to_slot,
        int(cfg.input.size),
        train=True,
        aug=cfg.aug,
        seed=int(cfg.seed),
        input_mode=str(cfg.input.get("mode", "letterbox")),
        crop_size=cfg.input.get("crop_size", None),
        slot_rarity=slot_rarity,
    )
    state = StageState(cfg=cfg, stage="stage1", model=model, slotmap=slotmap, train_dataset=ds, val_records=val_recs,
                       split=split, init_hash=init_hash, sample_weights=sample_weights)
    state.report_fn = stage1_report(state)
    return state
