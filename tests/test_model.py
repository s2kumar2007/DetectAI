"""Phase 3: backbone / FPN / OLN-RPN / cosine RoI head / assertions / imprinting (random tensors, CPU, tiny config)."""
import copy

import pytest
import torch
import torch.nn as nn

from anveshak.models.assertions import PretrainedWeightError, assert_prev_checkpoint_ok, assert_stage1_random_init
from anveshak.models.backbone import build_backbone
from anveshak.models.detector import arch_signature, build_model
from anveshak.models.imprint import imprint_new_classes
from anveshak.utils.config import REPO_ROOT, load_config
from anveshak.utils.hashing import state_dict_hash
from anveshak.utils.seed import seed_everything


@pytest.fixture(scope="module")
def cfg():
    return load_config(REPO_ROOT / "configs" / "smoke.yaml")


def _batch(n=2, size=128, seed=0, n_boxes=2):
    g = torch.Generator().manual_seed(seed)
    imgs = [(torch.rand(3, size, size, generator=g) * 255).to(torch.uint8) for _ in range(n)]
    tg = []
    for i in range(n):
        xy = torch.rand(n_boxes, 2, generator=g) * (size * 0.5)
        wh = 16 + torch.rand(n_boxes, 2, generator=g) * 32
        tg.append({"boxes": torch.cat([xy, xy + wh], 1), "labels": torch.randint(0, 3, (n_boxes,), generator=g)})
    return imgs, tg


def _model(cfg, active=(0, 1, 2), seed=0):
    seed_everything(seed, deterministic=False)
    m = build_model(cfg)
    m.activate_slots(list(active))
    return m


def test_backbone_is_groupnorm_random_init():
    for name in ("resnet50", "resnet18"):
        b = build_backbone(name)
        assert not any(isinstance(m, nn.BatchNorm2d) for m in b.modules())
        assert any(isinstance(m, nn.GroupNorm) and m.num_groups == 32 for m in b.modules())
        x = torch.zeros(1, 3, 64, 64)
        outs = b(x)
        assert [o.shape[1] for o in outs] == b.out_channels
        assert [o.shape[-1] for o in outs] == [16, 8, 4, 2]


def test_backbone_freeze_stages():
    b = build_backbone("resnet18", layers=[1, 1, 1, 1])
    b.freeze_stages(3)
    assert not any(p.requires_grad for p in b.conv1.parameters())
    assert not any(p.requires_grad for p in b.layer1.parameters())
    assert not any(p.requires_grad for p in b.layer2.parameters())
    assert all(p.requires_grad for p in b.layer3.parameters())


def test_fpn_levels_with_and_without_p2(cfg):
    for use_p2, n in ((True, 5), (False, 4)):
        c = copy.deepcopy(cfg)
        c.model.fpn.use_p2 = use_p2
        m = build_model(c)
        feats = m.extract(torch.zeros(1, 3, 128, 128))
        assert len(feats) == n and m.fpn.num_levels == n
        assert feats[-1].shape[-1] == 2  # P6 = stride 64


def test_forward_backward_finite(cfg):
    m = _model(cfg).train()
    imgs, tg = _batch()
    losses, aux = m.forward_train(imgs, tg)
    assert set(losses) >= {"rpn_reg", "rpn_ctr", "rpn_iou", "roi_cls", "roi_box", "roi_obj", "roi_margin"}
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert m.roi_head.cls.prototypes.grad is not None


def test_forward_with_no_gt_and_with_mining(cfg):
    m = _model(cfg).train()
    m.mining_on = True
    imgs, _ = _batch()
    empty = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)} for _ in imgs]
    losses, _ = m.forward_train(imgs, empty)
    assert all(torch.isfinite(v) for v in losses.values())
    imgs, tg = _batch(seed=3)
    losses, aux = m.forward_train(imgs, tg)
    assert all(torch.isfinite(v) for v in losses.values())
    sum(losses.values()).backward()


def test_slot_masking_inactive_minus_inf(cfg):
    m = _model(cfg, active=(0, 5, 7)).eval()
    cls = m.roi_head.cls
    emb = torch.randn(6, cls.dim)
    logits, cos = cls(emb)
    inactive = [i for i in range(cls.num_slots) if i not in (0, 5, 7)]
    assert torch.isinf(logits[:, inactive]).all() and (logits[:, inactive] < 0).all()
    assert torch.isfinite(logits[:, [0, 5, 7]]).all()
    p = torch.softmax(logits, 1)
    assert torch.allclose(p[:, [0, 5, 7]].sum(1), torch.ones(6)) and (p[:, inactive] == 0).all()
    out = m.predict_raw(_batch(1)[0])[0]
    assert (out["probs"][:, inactive] == 0).all()
    assert out["probs"].shape[1] == cls.num_slots == 64
    assert m.active_slots == [0, 5, 7]


def test_loss_only_uses_active_slots(cfg):
    m = _model(cfg, active=(0, 1, 2)).train()
    imgs, tg = _batch()
    _, aux = m.forward_train(imgs, tg)
    assert torch.isinf(aux["logits"][:, 3:]).all()


def test_predict_is_deterministic(cfg):
    m = _model(cfg).eval()
    imgs, _ = _batch(2, seed=5)
    a, b = m.predict_raw(imgs), m.predict_raw(imgs)
    for x, y in zip(a, b):
        for k in x:
            assert torch.equal(x[k], y[k])
    for o in a:
        assert o["boxes"].shape[1] == 4 and (o["obj"] >= 0).all() and (o["obj"] <= 1).all()


def test_imprint_sets_prototype_to_normalised_mean_embedding(cfg):
    m = _model(cfg, active=(0,)).eval()
    imgs, tg = _batch(3, seed=2, n_boxes=3)
    for t in tg:
        t["labels"] = torch.tensor([5, 6, 5])
    before = m.roi_head.cls.prototypes.detach().clone()
    counts = imprint_new_classes(m, [(imgs, tg)], [5, 6])
    assert counts == {5: 6, 6: 3}
    emb = m.embed_gt_boxes(imgs, [t["boxes"] for t in tg])
    lab = torch.cat([t["labels"] for t in tg])
    for s in (5, 6):
        want = torch.nn.functional.normalize(emb[lab == s].mean(0), dim=0)
        assert torch.allclose(m.roi_head.cls.prototypes[s], want, atol=1e-5)
        assert abs(float(m.roi_head.cls.prototypes[s].norm()) - 1.0) < 1e-5
    assert torch.equal(m.roi_head.cls.prototypes[0], before[0])  # untouched slot
    assert m.active_slots == [0, 5, 6]
    with pytest.raises(ValueError):
        imprint_new_classes(m, [(imgs, tg)], [9])


def test_stage1_assertion_passes_on_fresh_random_init(cfg, capsys):
    seed_everything(7, deterministic=False)
    m = build_model(cfg)
    h = assert_stage1_random_init(cfg, m, build_model, 7)
    assert h == state_dict_hash(m.state_dict()) and h in capsys.readouterr().out


def test_stage1_assertion_rejects_checkpoint_and_modified_weights(cfg):
    seed_everything(7, deterministic=False)
    m = build_model(cfg)
    bad = copy.deepcopy(cfg)
    bad.init.checkpoint = "somewhere/model.pt"
    with pytest.raises(PretrainedWeightError):
        assert_stage1_random_init(bad, m, build_model, 7)
    with torch.no_grad():
        next(m.parameters()).add_(1e-3)  # simulates a loaded / altered weight
    with pytest.raises(PretrainedWeightError):
        assert_stage1_random_init(cfg, m, build_model, 7)


def test_prev_checkpoint_assertion(cfg):
    meta = {"created_by": "anveshak-owod-rs", "stage": "stage1", "arch": arch_signature(cfg)}
    assert_prev_checkpoint_ok(meta, cfg, ["stage1"], "ok.pt")
    with pytest.raises(PretrainedWeightError):
        assert_prev_checkpoint_ok({**meta, "created_by": "torchvision"}, cfg, ["stage1"], "x.pt")
    with pytest.raises(PretrainedWeightError):
        assert_prev_checkpoint_ok({**meta, "stage": "stage3"}, cfg, ["stage1"], "x.pt")
    with pytest.raises(PretrainedWeightError):
        assert_prev_checkpoint_ok({**meta, "arch": {**meta["arch"], "embed_dim": 999}}, cfg, ["stage1"], "x.pt")
    with pytest.raises(PretrainedWeightError):
        assert_prev_checkpoint_ok(meta, cfg, ["stage1"], "https://example.com/w.pt")


def test_rpn_negatives_never_inside_dilated_gt(cfg):
    m = _model(cfg).train()
    imgs, _ = _batch(1)
    x, sizes = m.preprocess(imgs)
    feats = m.extract(x)
    _, _, _, _, pts, lvl, slices = m.rpn._head(feats)
    gt = torch.tensor([[20.0, 20.0, 60.0, 50.0]])
    pos, idx, neg = m.rpn._assign(pts, lvl, gt, slices[0][1])
    assert pos.any() and neg.any() and not (pos & neg).any()
    cx, cy, w, h = 40.0, 35.0, 40.0, 30.0
    d = float(cfg.model.rpn.neg_dilate)
    inside = ((pts[:, 0] - cx).abs() < w * d / 2) & ((pts[:, 1] - cy).abs() < h * d / 2)
    assert not (neg & inside).any()          # unlabelled-object safety: ambiguous zone is ignored
    assert (inside & ~pos & ~neg).any()      # ... and that zone is non-empty


def test_mahalanobis_ablation_off_by_default(cfg):
    assert cfg.model.ablation.prob_mahalanobis is False
    assert build_model(cfg).roi_head.maha is None
