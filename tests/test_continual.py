"""Phase 6: Stage 2 on SYNTHETIC Set B - imprinting, pseudo-labels, replay, KD, frozen parts, retention."""
import json

import numpy as np
import pytest
import torch

from anveshak.data.records import ImageRecord
from anveshak.models.assertions import PretrainedWeightError
from anveshak.training.checkpoint import load_payload, model_from_checkpoint
from anveshak.training.continual import feature_l2, kl_old_logits
from anveshak.training.pseudo import pseudo_labels
from anveshak.training.replay import MixedDataset, build_replay_buffer
from anveshak.training.run import run_stage
from anveshak.utils.config import REPO_ROOT, load_config

S2 = [str(REPO_ROOT / "configs/stage2_setB.yaml"), str(REPO_ROOT / "configs/smoke.yaml")]


def _cfg2(syn_root, s1, out, iters, extra=()):
    return load_config(S2, [f"data.root={syn_root}", f"out_dir={out}", f"init.checkpoint={s1 / 'checkpoint.pt'}", f"train.max_iters={iters}",
                            "train.max_hours=null", "train.log_every=10", "train.ckpt_every=0", "train.lr=0.0003", *extra])


@pytest.fixture(scope="module")
def stage2(tiny_stage2):
    return tiny_stage2


def test_pseudo_labels_filter_by_score_and_gt_overlap():
    probs = torch.zeros(4, 8)
    probs[:, 2] = 1.0
    raw = {"boxes": torch.tensor([[0, 0, 20, 20.], [100, 100, 130, 130.], [50, 50, 80, 80.], [0, 0, 20, 20.]]),
           "obj": torch.tensor([0.9, 0.9, 0.2, 0.95]), "probs": probs}
    cfg = load_config().pseudo
    cfg.score_thr = 0.7
    b, l = pseudo_labels(raw, torch.zeros(0, 4), cfg)
    assert len(b) == 2 and (l == 2).all()                        # low-score box dropped, duplicate NMS'd
    b, l = pseudo_labels(raw, torch.tensor([[2, 2, 22, 22.]]), cfg)  # overlaps a real (new-class) GT -> not added
    assert len(b) == 1 and torch.allclose(b[0], torch.tensor([100, 100, 130, 130.]))


def test_replay_buffer_is_balanced_and_seeded():
    recs = []
    for i in range(60):
        c = "ABC"[i % 3] if i < 50 else "D"
        recs.append(ImageRecord(f"i{i}", "x", 10, 10, np.array([[0, 0, 5, 5.]]), [c]))
    buf = build_replay_buffer(recs, ["A", "B", "C", "D"], per_class=4, max_images=100, seed=1)
    cnt = {c: sum(l == c for r in buf for l in r.labels) for c in "ABCD"}
    assert all(v == 4 for v in cnt.values()), cnt
    assert [r.image_id for r in buf] == [r.image_id for r in build_replay_buffer(recs, list("ABCD"), 4, 100, 1)]
    assert len(build_replay_buffer(recs, list("ABCD"), 50, 7, 1)) == 7  # cap
    only_a = build_replay_buffer(recs, ["A"], 3, 100, 0)
    assert all(set(r.labels) == {"A"} for r in only_a)


def test_mixed_dataset_weights_and_distill_losses():
    class D(torch.utils.data.Dataset):
        def __init__(self, n): self.n = n
        def __len__(self): return self.n
        def __getitem__(self, i): return i
    m = MixedDataset([D(4), D(6)])
    assert len(m) == 10 and m[3] == 3 and m[4] == 0 and m[9] == 5
    w = m.weights([0.25, 0.75])
    assert sum(w[:4]) == pytest.approx(0.25) and sum(w[4:]) == pytest.approx(0.75)
    t = torch.randn(5, 64)
    assert kl_old_logits(t, t, [0, 1, 2], 2.0).item() == pytest.approx(0.0, abs=1e-6)
    assert kl_old_logits(torch.randn(5, 64), t, [0, 1, 2], 2.0) > 0
    f = [torch.randn(1, 4, 8, 8)]
    assert feature_l2(f, f) == 0


def test_stage2_slots_imprint_freeze_and_checkpoint(stage2, tiny_stage1, syn_root):
    out, res = stage2
    prev = load_payload(tiny_stage1 / "checkpoint.pt")
    pl = load_payload(out / "checkpoint.pt")
    slots = json.loads((out / "class_slots.json").read_text())["class_to_slot"]
    old = prev["slot_map"]["class_to_slot"]
    assert {k: v for k, v in slots.items() if k in old} == old                       # old classes keep their slots
    new = {k: v for k, v in slots.items() if k not in old}
    assert sorted(new.values()) == [4, 5] and set(new) == {"SYN07", "SYN08"}        # next free slots after Stage 1
    # old prototypes were frozen (EMA of unchanged weights == previous EMA prototypes)
    for s in old.values():
        assert torch.allclose(pl["model"]["roi_head.cls.prototypes"][s], prev["ema"]["roi_head.cls.prototypes"][s], atol=1e-6)
    assert pl["stage"] == "stage2" and pl["prev_sha256"] and pl["prev_stage"] == "stage1"
    assert pl["init_hash"] == prev["init_hash"]                                      # same lineage
    assert pl["arch"] == prev["arch"]
    # frozen early stages: stem + layer1 unchanged
    for k, v in pl["model"].items():
        if k.startswith(("backbone.conv1", "backbone.layer1.")):
            assert torch.equal(v, prev["ema"][k]), k
    losses = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines()]
    assert "kd_feat" in losses[0] and "kd_logit" in losses[0]


def test_retention_report_and_old_class_map_retained(stage2):
    out, _ = stage2
    rep = json.loads((out / "retention_report.json").read_text())
    b, a = rep["old_map_before"], rep["old_map_after"]
    print(f"[SMOKE, synthetic] old-class mAP before={b:.3f} after={a:.3f} retention={rep['retention_ratio']}")
    assert rep["new_slots"] == {"SYN07": 4, "SYN08": 5} and rep["n_val_images"] > 0
    assert b > 0.3, "stage-1 smoke model should have learned the easy synthetic classes"
    assert a >= 0.65 * b, f"old-class mAP collapsed: {b:.3f} -> {a:.3f}"
    assert rep["new_map_on_training_images"] >= 0


def test_stage2_model_detects_new_classes_and_keeps_old(stage2, syn_root):
    out, _ = stage2
    model, meta, sm = model_from_checkpoint(out / "checkpoint.pt")
    assert model.active_slots == [0, 1, 2, 3, 4, 5]
    raw = model.predict_raw([torch.zeros(3, 128, 128, dtype=torch.uint8)])[0]
    assert (raw["probs"][:, 6:] == 0).all()


def test_stage2_rejects_foreign_or_wrong_stage_checkpoint(syn_root, tiny_stage1, tmp_path):
    with pytest.raises(PretrainedWeightError):
        run_stage(_cfg2(syn_root, tiny_stage1, tmp_path / "a", 1, ["init.checkpoint=null"]))
    # a Stage 2 checkpoint cannot start Stage 2 again (only Stage 1 can)
    s2 = tmp_path / "s2"
    run_stage(_cfg2(syn_root, tiny_stage1, s2, 2))
    with pytest.raises(PretrainedWeightError):
        run_stage(_cfg2(syn_root, tiny_stage1, tmp_path / "b", 1, [f"init.checkpoint={s2 / 'checkpoint.pt'}"]))
    # architecture mismatch is refused (rule 3: same architecture)
    with pytest.raises(PretrainedWeightError):
        run_stage(_cfg2(syn_root, tiny_stage1, tmp_path / "c", 1, ["model.roi_head.embed_dim=32"]))
