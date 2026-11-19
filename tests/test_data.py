import json

import numpy as np
import pytest

from anveshak.data.classes import load_class_list
from anveshak.data.formats import detect_format, load_annotations, load_image_dir
from anveshak.data.synthetic import SyntheticSpec, class_code, generate_dataset
from anveshak.utils.config import load_config
from anveshak.utils.offline import NetworkBlockedError, enforce_offline


# ------------------------------------------------------------------ class list
@pytest.mark.parametrize(
    "name,content,expected",
    [
        ("a.txt", "# comment\nAAA\nBBB\n\nCCC\n", ["AAA", "BBB", "CCC"]),
        ("b.json", json.dumps(["x1", "x2"]), ["x1", "x2"]),
        ("c.json", json.dumps({"classes": ["p", "q"]}), ["p", "q"]),
        ("d.json", json.dumps({"1": "second", "0": "first"}), ["first", "second"]),
        ("e.json", json.dumps([{"class_code": "m"}, {"class_code": "n"}]), ["m", "n"]),
        ("f.csv", "id,class_code\n1,B\n0,A\n", ["A", "B"]),
        ("g.csv", "class\nK1\nK2\n", ["K1", "K2"]),
        ("h.txt", "X, Y, Z\n", ["X", "Y", "Z"]),
        ("i.txt", "﻿Q1\r\nQ2\r\n", ["Q1", "Q2"]),
    ],
)
def test_class_list_formats(tmp_path, name, content, expected):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    assert load_class_list(p) == expected


def test_class_list_rejects_duplicates_and_empty(tmp_path):
    p = tmp_path / "d.txt"
    p.write_text("A\nA\n")
    with pytest.raises(ValueError):
        load_class_list(p)
    p.write_text("\n\n")
    with pytest.raises(ValueError):
        load_class_list(p)
    with pytest.raises(FileNotFoundError):
        load_class_list(tmp_path / "missing.txt")


# ------------------------------------------------------------------ formats
def test_synthetic_default_formats_detected(syn_root):
    assert detect_format(syn_root / "set_a") == "coco"
    assert detect_format(syn_root / "set_b") == "yolo"
    assert detect_format(syn_root / "round2") == "voc"
    meta = json.loads((syn_root / "meta.json").read_text())
    names = load_class_list(syn_root / "class_list.txt")
    assert names == meta["class_list"]
    for sub, expected in (("set_a", meta["base"]), ("set_b", meta["few_shot_b"]), ("round2", meta["round2"])):
        recs = load_annotations(syn_root / sub, "auto", names)
        labels = {l for r in recs for l in r.labels}
        assert labels and labels <= set(expected), (sub, labels)
        for r in recs:
            if len(r.boxes):
                assert (r.boxes[:, 2] > r.boxes[:, 0]).all() and (r.boxes[:, 3] > r.boxes[:, 1]).all()
                assert r.boxes.max() <= max(r.width, r.height)


@pytest.mark.parametrize("fmt", ["coco", "yolo", "voc", "csv"])
def test_all_four_formats_roundtrip_same_boxes(tmp_path, fmt):
    spec = SyntheticSpec(size=96, n_a=6, n_b=3, n_r2=3, n_test=2, fmt_a=fmt, fmt_b=fmt, fmt_r2=fmt, seed=5)
    meta = generate_dataset(tmp_path, spec)
    names = load_class_list(tmp_path / "class_list.txt")
    assert detect_format(tmp_path / "set_a") == fmt
    recs = load_annotations(tmp_path / "set_a", "auto", names)
    assert len(recs) >= 1
    ref = generate_dataset(tmp_path / "ref", SyntheticSpec(size=96, n_a=6, n_b=3, n_r2=3, n_test=2, fmt_a="coco", seed=5))
    ref_recs = {r.image_id: r for r in load_annotations(tmp_path / "ref" / "set_a", "coco", names)}
    for r in recs:
        rr = ref_recs[r.image_id]
        assert sorted(r.labels) == sorted(rr.labels)
        a = r.boxes[np.lexsort(r.boxes.T)] if len(r.boxes) else r.boxes
        b = rr.boxes[np.lexsort(rr.boxes.T)] if len(rr.boxes) else rr.boxes
        np.testing.assert_allclose(a, b, atol=1.0)
    assert set(l for r in recs for l in r.labels) <= set(meta["base"])


def test_yolo_without_class_list_falls_back(tmp_path):
    generate_dataset(tmp_path, SyntheticSpec(size=64, n_a=2, n_b=3, n_r2=2, n_test=1, seed=1))
    recs = load_annotations(tmp_path / "set_b", "yolo", None)
    assert all(l.startswith("class_") for r in recs for l in r.labels)


def test_unlabelled_objects_exist_in_set_b(syn_root):
    """Set B images contain base objects that are NOT annotated (the continual-learning trap)."""
    names = load_class_list(syn_root / "class_list.txt")
    recs = load_annotations(syn_root / "set_b", "auto", names)
    assert all(set(r.labels) <= {"SYN07", "SYN08"} for r in recs)


def test_image_dir_and_id_mode(syn_root):
    recs = load_image_dir(syn_root / "test")
    assert len(recs) == 24 and recs[0].image_id == "t_0000"
    recs_fn = load_image_dir(syn_root / "test", "filename")
    assert recs_fn[0].image_id == "t_0000.png"


def test_detect_format_raises_on_empty(tmp_path):
    (tmp_path / "x.txt").write_text("hello")
    with pytest.raises(ValueError):
        detect_format(tmp_path)


def test_inspect_data_script(syn_root, capsys):
    import inspect_data

    info = inspect_data.describe(str(syn_root / "set_a"), "auto", str(syn_root / "class_list.txt"))
    assert info["format"] == "coco" and info["images"] == 96 and info["boxes"] > 0
    assert set(info["class_counts"]) <= {class_code(i) for i in range(6)}
    assert "SYN07" in info["classes_in_list_not_in_data"]


# ------------------------------------------------------------------ config / offline
def test_config_merge_and_override():
    cfg = load_config(["configs/smoke.yaml"], ["train.lr=0.5", "model.fpn.use_p2=false"])
    assert cfg.input.size == 128 and cfg.train.lr == 0.5 and cfg.model.fpn.use_p2 is False
    assert cfg.model.roi_head.num_slots == 64  # inherited from base.yaml
    assert cfg.model.backbone.layers == [1, 1, 1, 1]


def test_offline_guard_blocks_downloads_and_network():
    import socket

    import torch

    enforce_offline()
    with pytest.raises(NetworkBlockedError):
        torch.hub.load_state_dict_from_url("https://example.com/w.pth")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(NetworkBlockedError):
        s.connect(("93.184.216.34", 80))
    s.close()
