"""Unit and smoke tests for Phase 13: RFS, native crops, 1500-box density, split policy, and resume."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from anveshak.data.dataset import DetDataset
from anveshak.data.records import ImageRecord
from anveshak.data.split import make_split
from anveshak.models.rpn import OLNRPN
from anveshak.postprocess.engine import Predictor
from anveshak.postprocess.unknown import Calibration
from anveshak.training.run import run_stage
from anveshak.training.sampling import compute_class_loss_weights, compute_repeat_factors
from anveshak.utils.config import REPO_ROOT, load_config


def test_rfs_rare_classes_sampled_more_often():
    c2s = {"DOM29": 29, "DOM28": 28, "RARE0": 0}
    recs = []
    for i in range(50):
        recs.append(ImageRecord(f"dom_{i}", f"dom_{i}.tif", 128, 128, np.array([[10, 10, 30, 30]], np.float32), ["DOM29"]))
    for i in range(40):
        recs.append(ImageRecord(f"dom2_{i}", f"dom2_{i}.tif", 128, 128, np.array([[10, 10, 30, 30]], np.float32), ["DOM28"]))
    for i in range(2):
        recs.append(ImageRecord(f"rare_{i}", f"rare_{i}.tif", 128, 128, np.array([[10, 10, 30, 30]], np.float32), ["RARE0"]))
    recs.append(ImageRecord("empty_0", "empty_0.tif", 128, 128, np.zeros((0, 4), np.float32), []))

    weights, r_cls = compute_repeat_factors(recs, c2s, t=0.2, empty_weight=0.25)
    assert r_cls["RARE0"] > r_cls["DOM29"]
    assert weights[-1] == 0.25
    assert weights[-2] > weights[0]

    # Weighted sampling draws rare tiles much more frequently per tile than dominant tiles
    probs = np.array(weights, dtype=np.float64)
    probs /= probs.sum()
    assert probs[-2] > probs[0] * 2.5

    cw = compute_class_loss_weights(recs, c2s, num_slots=64, cap=5.0, power=0.5)
    assert cw[0] > cw[29]
    assert float(cw.max()) <= 5.0


def test_native_crop_preserves_resolution(tmp_path):
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[100:140, 100:140] = 200
    p = tmp_path / "tile.tif"
    Image.fromarray(img).save(p)
    rec = ImageRecord("tile", str(p), 256, 256, np.array([[100, 100, 140, 140]], np.float32), ["A"])
    ds = DetDataset(
        [rec], {"A": 0}, size=256, train=True,
        aug={"hflip": 0.0, "vflip": 0.0, "rot90": False, "scale_jitter": [1.0, 1.0], "min_vis": 0.6, "mosaic": 0.0, "color_jitter": 0.0},
        seed=42, input_mode="crop", crop_size=128, slot_rarity={0: 2.0},
    )
    t_img, tgt = ds[0]
    assert tuple(t_img.shape) == (3, 128, 128)
    assert tgt["scale"] == 1.0


def test_density_1500_boxes_bounded_and_capped():
    cfg = load_config()
    rpn = OLNRPN(64, [4, 8, 16], cfg.model.rpn, groups=16)
    # 1500 synthetic GT boxes on a 512x512 feature grid
    pts = torch.rand(2000, 2) * 512.0
    lvl = torch.zeros(2000, dtype=torch.long)
    rng = np.random.default_rng(0)
    x1 = rng.uniform(0, 480, size=1500)
    y1 = rng.uniform(0, 480, size=1500)
    gt = torch.tensor(np.stack([x1, y1, x1 + 20.0, y1 + 20.0], axis=1), dtype=torch.float32)
    pos, gt_idx, neg = rpn._assign(pts, lvl, gt, n_level0=1000)
    assert pos.any() and neg.any()

    # Predictor finalize enforces 1,000-row cap
    s2n = {i: f"C{i}" for i in range(10)}
    pred = Predictor(torch.nn.Identity(), s2n, 1024, cfg.infer, Calibration(unk_cap=20), device="cpu", tta=False)
    n_k = 1400
    kx = np.linspace(10, 900, n_k)
    known = {
        "boxes": np.stack([kx, kx * 0.5, kx + 20, kx * 0.5 + 20], axis=1).astype(np.float32),
        "scores": np.linspace(0.99, 0.1, n_k, dtype=np.float32),
        "slots": np.arange(n_k, dtype=np.int64) % 10,
    }
    unknown = {
        "boxes": np.array([[50, 50, 80, 80]], dtype=np.float32),
        "scores": np.array([0.85], dtype=np.float32),
    }
    rows = pred.finalize("tile_dense", known, unknown, scale=1.0, width=1024, height=1024)
    assert len(rows) == 1000
    assert rows[0][2] >= rows[-1][2]


def test_make_split_holdout_policy():
    names = [f"C{i}" for i in range(8)]
    # C0, C1: 6000 boxes (top-2); C2..C5: 1500 boxes (mid band); C6, C7: 100 boxes (< 500)
    counts = [6000, 6000, 1500, 1500, 1500, 1500, 100, 100]
    recs = []
    idx = 0
    for cname, cnt in zip(names, counts):
        per_tile = max(1, cnt // 10)
        for _ in range(10):
            boxes = np.tile(np.array([[10, 10, 30, 30]], np.float32), (per_tile, 1))
            recs.append(ImageRecord(f"im_{idx}", f"im_{idx}.tif", 128, 128, boxes, [cname] * per_tile))
            idx += 1

    sp_def = make_split(recs, n_holdout=2, val_frac=0.25, seed=7, class_list=names)
    assert set(sp_def["heldout_classes"]).issubset({"C2", "C3", "C4", "C5"})

    sp_exp = make_split(recs, n_holdout=2, val_frac=0.25, seed=7, class_list=names, holdout_classes=["2", "4"])
    assert sp_exp["heldout_classes"] == ["C2", "C4"]


def test_resume_from_last_checkpoint(syn_root, tmp_path):
    out = tmp_path / "resume_run"
    base_ov = [
        f"data.root={syn_root}",
        f"out_dir={out}",
        "train.max_hours=null",
        "train.log_every=2",
        "train.ckpt_every=2",
        "train.eval_at_end=false",
    ]
    cfg1 = load_config([str(REPO_ROOT / "configs/stage1_setA.yaml"), str(REPO_ROOT / "configs/smoke.yaml")], base_ov + ["train.max_iters=4"])
    r1 = run_stage(cfg1, resume=True)
    assert r1["summary"]["iterations"] == 4
    assert (out / "last.pt").is_file()

    cfg2 = load_config([str(REPO_ROOT / "configs/stage1_setA.yaml"), str(REPO_ROOT / "configs/smoke.yaml")], base_ov + ["train.max_iters=7"])
    r2 = run_stage(cfg2, resume=True)
    assert r2["summary"]["iterations"] == 7
    log_txt = (out / "train.log").read_text(encoding="utf-8")
    assert "RESUMED from" in log_txt and "at iteration 4" in log_txt
