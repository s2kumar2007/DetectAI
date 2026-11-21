"""Tests for Phase 10: Unknown objectness fix and negative sampling hygiene."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torchvision.ops import box_iou

from anveshak.data.build import load_class_names, load_set
from anveshak.data.split import make_split, save_split, val_view
from anveshak.data.synthetic import SyntheticSpec, generate_dataset
from anveshak.models.detector import build_model
from anveshak.training.run import run_stage
from anveshak.utils.config import REPO_ROOT, load_config
from anveshak.utils.seed import seed_everything


def test_unlabelled_object_gets_no_background_target():
    """Test 4(a): An unlabelled object in a synthetic batch gets no background target at its location."""
    seed_everything(42, deterministic=False)
    cfg = load_config(REPO_ROOT / "configs" / "smoke.yaml")
    cfg.model.roi_head.ignore_high_obj_negatives = True
    cfg.model.roi_head.roi_neg_weight = 0.1
    m = build_model(cfg).train()
    m.activate_slots([0, 1])

    # Image with 2 objects: Object 1 is labelled GT; Object 2 is UNLABELLED.
    img = torch.zeros(1, 3, 128, 128, dtype=torch.uint8)
    box1 = torch.tensor([[10.0, 10.0, 40.0, 40.0]])  # Labelled GT
    box2 = torch.tensor([[70.0, 70.0, 100.0, 100.0]])  # Unlabelled object

    targets = [{"boxes": box1, "labels": torch.tensor([0])}]

    # Proposals: one on box1, one on box2 (unlabelled object), one far in background
    proposals = [torch.cat([box1, box2, torch.tensor([[100.0, 10.0, 120.0, 30.0]])], dim=0)]
    scores = [torch.tensor([0.9, 0.85, 0.1])]

    feats = m.extract(m.preprocess([img[0]])[0])
    shapes = [(128, 128)]

    # Run RoI loss with mining_on=True (active post-warmup)
    losses, aux = m.roi_head.loss(feats, proposals, scores, targets, shapes, mining_on=True)

    rois = aux["rois"][0]
    obj_t = aux["obj_t"]
    obj_w = aux["obj_w"]

    # Identify proposals near box 2 (the unlabelled object)
    ious_to_box2 = box_iou(rois, box2).squeeze(1)
    near_unlabelled = ious_to_box2 >= 0.5

    assert near_unlabelled.any(), "Proposal overlapping unlabelled object must be sampled"
    # An unlabelled object must NEVER get a positive-weight background target (obj_t == 0 with obj_w > 0)
    has_background_target = (obj_t == 0.0) & (obj_w > 0.0) & near_unlabelled
    assert not has_background_target.any(), (
        f"Unlabelled object proposal received background target: obj_t={obj_t[near_unlabelled]}, obj_w={obj_w[near_unlabelled]}"
    )


def test_unknown_objectness_above_background_after_smoke_train(tmp_path):
    """Test 4(b): After short smoke train, mean objectness on held-out unknown boxes is > 0.05 and > background."""
    syn_root = tmp_path / "syn_data"
    generate_dataset(syn_root, SyntheticSpec(size=128, n_a=48, n_b=8, n_r2=8, n_test=12, seed=123))

    cfg_base = load_config([str(REPO_ROOT / "configs/stage1_setA.yaml"), str(REPO_ROOT / "configs/smoke.yaml")],
                           [f"data.root={syn_root}"])
    names = load_class_names(cfg_base)
    recs = load_set(cfg_base, "set_a", names)
    split = make_split(recs, n_holdout=2, val_frac=0.25, seed=123, class_list=names)
    split_path = tmp_path / "split.json"
    save_split(split, split_path)

    out_dir = tmp_path / "smoke_train"
    train_cfg = load_config(
        [str(REPO_ROOT / "configs/stage1_setA.yaml"), str(REPO_ROOT / "configs/smoke.yaml")],
        [
            f"data.root={syn_root}",
            f"out_dir={out_dir}",
            "train.max_iters=150",
            "train.max_hours=null",
            f"data.split_file={split_path}",
            "train.log_every=25",
            "train.ckpt_every=0",
            "train.eval_at_end=false",
            "model.roi_head.roi_neg_weight=0.25",
            "model.roi_head.ignore_high_obj_negatives=true",
        ],
    )
    run_stage(train_cfg)

    # Evaluate via diagnose_unknown
    import sys
    if str(REPO_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from diagnose_unknown import diagnose_checkpoint

    diag = diagnose_checkpoint(out_dir / "checkpoint.pt", split_path, train_cfg, device_name="cpu")
    t = diag["table"]
    mean_unk_obj = t["unknown"]["mean_obj"]
    mean_bg_obj = t["background"]["mean_obj"]

    print(f"\n[Test 4(b)] mean unknown objectness: {mean_unk_obj:.4f}, mean background objectness: {mean_bg_obj:.4f}")
    assert mean_unk_obj > 0.05, f"Mean unknown objectness {mean_unk_obj:.4f} must be > 0.05"
    assert mean_unk_obj > mean_bg_obj, f"Unknown objectness ({mean_unk_obj:.4f}) must be higher than background ({mean_bg_obj:.4f})"
