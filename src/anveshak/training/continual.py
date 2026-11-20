"""Stage 2 (Set B) and Stage 3 (Round 2): continue the SAME model without forgetting.

Ingredients: previous-checkpoint validation (our own checkpoint only), frozen teacher, weight imprinting of the new
classes, teacher pseudo-labels for old classes, balanced exemplar replay, feature (L2 on FPN) + logit (KL over old-class
RoI logits) distillation, frozen stem / early stages, frozen old prototypes, retention report (old-class mAP before/after).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.build import drop_heldout, load_class_names, load_set, ordered_classes, resolve
from ..data.dataset import DetDataset, collate
from ..data.records import ImageRecord
from ..data.slots import SlotMap
from ..data.split import val_view
from ..models.assertions import PretrainedWeightError, assert_prev_checkpoint_ok
from ..models.detector import OWODDetector, build_model
from ..models.imprint import imprint_new_classes
from ..utils.hashing import file_sha256
from ..utils.log import get_logger
from ..utils.offline import enforce_offline
from ..utils.seed import seed_everything
from .checkpoint import load_payload, payload_meta
from .engine import StageState
from .evaluate import known_map
from .pseudo import pseudo_labels
from .replay import MixedDataset, build_replay_buffer

log = get_logger("anveshak.continual")


def verified_ok(ckpt_path: str | Path) -> bool:
    """True when ``<ckpt>.verified.json`` (written by scripts/verify_checkpoint.py) says PASS for this exact file."""
    vf = Path(str(ckpt_path) + ".verified.json")
    if not vf.exists():
        return False
    j = json.loads(vf.read_text(encoding="utf-8"))
    return bool(j.get("passed")) and j.get("checkpoint_sha256") == file_sha256(ckpt_path)


def kl_old_logits(student: torch.Tensor, teacher: torch.Tensor, old_slots: Sequence[int], T: float) -> torch.Tensor:
    """KL(teacher || student) over the OLD-class slots only, temperature T, mean over RoIs."""
    idx = torch.as_tensor(list(old_slots), device=student.device)
    s = F.log_softmax(student[:, idx].float() / T, dim=1)
    t = F.softmax(teacher[:, idx].float() / T, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


def feature_l2(student_feats: Sequence[torch.Tensor], teacher_feats: Sequence[torch.Tensor]) -> torch.Tensor:
    return sum(F.mse_loss(s.float(), t.float()) for s, t in zip(student_feats, teacher_feats)) / len(student_feats)


def make_loss_fn(teacher: OWODDetector, cfg, old_slots: List[int]):
    pcfg, dcfg = cfg.pseudo, cfg.distill

    def fn(model: OWODDetector, images, targets):
        with torch.no_grad():
            x, sizes = teacher.preprocess(images)
            t_feats = teacher.extract(x)
            shapes = [tuple(x.shape[-2:])] * x.shape[0]
            props, _ = teacher.rpn(t_feats, sizes)
            raws = teacher.roi_head.predict(t_feats, props, sizes, shapes) if bool(pcfg.enable) else None
        dev = x.device
        tg = []
        for i, t in enumerate(targets):
            boxes, labels = t["boxes"].to(dev).float(), t["labels"].to(dev).long()
            if raws is not None:
                pb, pl = pseudo_labels(raws[i], boxes, pcfg)
                boxes, labels = torch.cat([boxes, pb]), torch.cat([labels, pl])
            tg.append({"boxes": boxes, "labels": labels})
        losses, aux = model.forward_train(images, tg)
        if bool(dcfg.enable):
            losses["kd_feat"] = float(dcfg.feat_weight) * feature_l2(aux["feats"], t_feats)
            with torch.no_grad():
                t_logits = teacher.roi_head.logits_for(t_feats, aux["rois"], shapes)
            losses["kd_logit"] = float(dcfg.logit_weight) * kl_old_logits(aux["logits"], t_logits, old_slots, float(dcfg.temperature))
        return losses

    return fn


def prepare_continual(cfg, device) -> StageState:
    stage = cfg.stage
    allowed = ["stage1"] if stage == "stage2" else ["stage2"]
    ck = cfg.init.checkpoint
    if not ck:
        raise PretrainedWeightError(f"{stage} needs init.checkpoint = our previous-stage checkpoint ({allowed[0]})")
    enforce_offline()
    payload = load_payload(ck)
    assert_prev_checkpoint_ok(payload_meta(payload), cfg, allowed, str(ck))
    if bool(cfg.init.get("require_verified", False)) and not verified_ok(ck):
        raise PretrainedWeightError(f"{ck} has no passing {Path(str(ck)).name}.verified.json - run scripts/verify_checkpoint.py first")
    prev_sha = file_sha256(ck)
    seed_everything(int(cfg.seed), deterministic=False)
    model = build_model(cfg)
    model.load_state_dict(payload["ema"])  # continue from the EMA weights of the previous stage
    prev_map = SlotMap.from_dict(payload["slot_map"])
    old_slots = prev_map.active_slots
    teacher = copy.deepcopy(model).eval().to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    split = payload.get("split") or {"train_ids": [], "val_ids": [], "heldout_classes": []}
    val_ids = set(split["val_ids"])

    class_names = load_class_names(cfg)
    new_all = drop_heldout(load_set(cfg, cfg.data.train_dir, class_names), split)
    new_recs = [r for r in new_all if r.image_id not in val_ids and len(r.boxes)]
    new_classes = [c for c in ordered_classes(new_recs, class_names) if c not in prev_map]
    if not new_classes:
        raise ValueError(f"no NEW classes found in {cfg.data.train_dir} (all labelled classes already have slots)")
    slotmap = prev_map.copy()
    assigned = slotmap.assign(new_classes, stage, start=cfg.slots.start)
    model.activate_slots(list(assigned.values()))
    log.info("%s: %d new classes -> slots %s; %d old slots frozen/distilled", stage, len(assigned), sorted(assigned.values()), len(old_slots))

    size = int(cfg.input.size)
    parts = [DetDataset(new_recs, slotmap.class_to_slot, size, True, cfg.aug, int(cfg.seed), int(cfg.new_data.repeat))]
    fracs = [1.0]
    replay: List[ImageRecord] = []
    val_recs: List[ImageRecord] = []
    set_a = None
    if bool(cfg.replay.enable):
        for src in cfg.replay.sources:
            recs = drop_heldout(load_set(cfg, str(cfg.data.get(src, src)), class_names), split)
            if src == "set_a":
                set_a = recs
            replay += [r for r in recs if r.image_id not in val_ids]
        replay = build_replay_buffer(replay, prev_map.names, int(cfg.replay.per_class), int(cfg.replay.max_images), int(cfg.seed))
        log.info("replay buffer: %d images covering %d old classes", len(replay), len({l for r in replay for l in r.labels}))
    if replay:
        parts.append(DetDataset(replay, prev_map.class_to_slot, size, True, cfg.aug, int(cfg.seed) + 1))
        fracs = [1.0 - float(cfg.replay.ratio), float(cfg.replay.ratio)]
    mixed = MixedDataset(parts)
    if set_a is None and resolve(cfg, str(cfg.data.set_a)).exists():
        set_a = load_set(cfg, str(cfg.data.set_a), class_names)
    if set_a is not None:
        val_recs = val_view(set_a, split)

    state = StageState(cfg=cfg, stage=stage, model=model, slotmap=slotmap, train_dataset=mixed, val_records=val_recs, split=split,
                       init_hash=str(payload.get("init_hash", "")), sample_weights=mixed.weights(fracs) if len(parts) > 1 else None,
                       old_slots=old_slots, compute_losses=make_loss_fn(teacher, cfg, old_slots),
                       extra_meta={"prev_checkpoint": str(ck), "prev_sha256": prev_sha, "prev_stage": payload["stage"]})

    def setup(m: OWODDetector) -> None:
        if bool(cfg.imprint.enable):
            ds = DetDataset(new_recs, assigned, size, train=False)
            imprint_new_classes(m, DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate), list(assigned.values()))

    def report(m_after: OWODDetector, dev) -> Dict:
        old_names, s2n = prev_map.names, prev_map.slot_to_name()
        rep: Dict = {"old_classes": old_names, "new_classes": list(assigned), "new_slots": assigned}
        if val_recs:
            b, b_aps, _ = known_map(teacher, val_recs, s2n, old_names, size, dev)
            a, a_aps, _ = known_map(m_after, val_recs, s2n, old_names, size, dev)
            rep.update(old_map_before=b, old_map_after=a, retention_ratio=(a / b if b > 0 else None), n_val_images=len(val_recs),
                       old_ap_before=b_aps, old_ap_after=a_aps,
                       note="old-class mAP on held-out validation images of the old data; 'before' = previous-stage checkpoint (teacher)")
        else:
            rep["note"] = "no validation images available for retention"
        n2s = {s: n for n, s in assigned.items()}
        nm, naps, _ = known_map(m_after, new_recs[:200], n2s, list(assigned), size, dev)
        rep.update(new_map_on_training_images=nm, new_ap_on_training_images=naps,
                   new_map_note="measured on the few-shot TRAINING images (optimistic; no held-out few-shot data exists)")
        return rep

    state.setup_fn, state.report_fn = setup, report
    return state
