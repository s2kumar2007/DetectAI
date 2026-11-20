"""Phase 4: letterbox / augmentation geometry, slot map, dataset targets (synthetic data only)."""
import numpy as np
import pytest
import torch

from anveshak.data.build import load_class_names, load_set
from anveshak.data.dataset import DetDataset, collate, hflip, letterbox, mosaic4, rot90, scale_jitter, vflip
from anveshak.data.slots import SlotMap
from anveshak.utils.config import load_config


def _marked(h=40, w=60, box=(10, 5, 30, 20)):
    img = np.zeros((h, w, 3), np.uint8)
    x1, y1, x2, y2 = box
    img[y1:y2, x1:x2] = 255
    return img, np.array([box], np.float32)


def _content_box(img):
    ys, xs = np.nonzero(img[..., 0] > 127)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


@pytest.mark.parametrize("fn", [hflip, vflip, lambda i, b: rot90(i, b, 1), lambda i, b: rot90(i, b, 2), lambda i, b: rot90(i, b, 3)])
def test_geometry_transforms_keep_boxes_on_content(fn):
    img, b = _marked()
    img2, b2 = fn(img, b)
    assert np.allclose(b2[0], _content_box(img2), atol=1)


def test_letterbox_scale_and_pad():
    img, b = _marked(40, 60, (0, 0, 60, 40))
    can, b2, s = letterbox(img, b, 120)
    assert can.shape == (120, 120, 3) and abs(s - 2.0) < 1e-6
    assert np.allclose(b2[0], [0, 0, 120, 80]) and can[80:].sum() == 0


def test_scale_jitter_and_mosaic_boxes_valid():
    rng = np.random.default_rng(0)
    img, b = _marked(64, 64, (20, 20, 40, 40))
    lab = np.array([3])
    for f in (0.6, 1.0, 1.5):
        im2, b2, l2 = scale_jitter(img, b, lab, f, rng)
        assert im2.shape == img.shape and len(b2) == len(l2)
        if len(b2):
            assert (b2[:, :2] >= 0).all() and (b2[:, 2:] <= 64).all() and np.allclose(b2[0], _content_box(im2), atol=2)
    win, bm, lm = mosaic4([(img, b, lab)] * 4, rng)
    assert win.shape == img.shape and ((bm[:, 2:] <= 64) & (bm[:, :2] >= 0)).all() and len(bm) == len(lm)


def test_slotmap_allocation_and_json(tmp_path):
    sm = SlotMap()
    assert sm.assign(["A", "B", "C"], "stage1") == {"A": 0, "B": 1, "C": 2}
    assert sm.assign(["B", "D"], "stage2") == {"D": 3}  # existing classes keep their slot
    r2 = sm.assign(["X", "Y"], "stage3", start=40)
    assert r2 == {"X": 40, "Y": 41}
    with pytest.raises(ValueError):
        sm.assign(["Z"], "stage3", start=3)  # inside the Round 1 range
    with pytest.raises(ValueError):
        sm.assign(["Q"], "stage3", start=41)  # occupied
    sm.save(tmp_path / "class_slots.json")
    sm2 = SlotMap.load(tmp_path / "class_slots.json")
    assert sm2.class_to_slot == sm.class_to_slot and sm2.slot_stage == sm.slot_stage and sm2.active_slots == [0, 1, 2, 3, 40, 41]


def test_dataset_targets_slots_and_unlabelled_dropped(syn_root):
    cfg = load_config(overrides=[f"data.root={syn_root}"])
    names = load_class_names(cfg)
    recs = load_set(cfg, "set_a", names)
    mapping = {names[0]: 7, names[1]: 9}  # only two classes have slots -> others are unlabelled
    ds = DetDataset(recs, mapping, 128, train=False)
    seen = set()
    for i in range(len(ds)):
        img, t = ds[i]
        assert img.shape == (3, 128, 128) and img.dtype == torch.uint8
        assert t["boxes"].shape[0] == t["labels"].shape[0]
        seen |= set(t["labels"].tolist())
    assert seen <= {7, 9} and seen


def test_train_dataset_is_deterministic_per_epoch_and_changes_across_epochs(syn_root):
    cfg = load_config(overrides=[f"data.root={syn_root}", "aug.mosaic=0.5"])
    names = load_class_names(cfg)
    recs = load_set(cfg, "set_a", names)
    m = {n: i for i, n in enumerate(names)}
    ds = DetDataset(recs, m, 128, train=True, aug=cfg.aug, seed=3)
    a1, t1 = ds[5]
    a2, t2 = ds[5]
    assert torch.equal(a1, a2) and torch.equal(t1["boxes"], t2["boxes"])
    ds.set_epoch(1)
    a3, _ = ds[5]
    assert not torch.equal(a1, a3)
    b = collate([ds[i] for i in range(3)])
    assert len(b[0]) == 3 and len(b[1]) == 3
