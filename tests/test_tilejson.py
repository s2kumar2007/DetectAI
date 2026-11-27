"""Unit and smoke tests for Phase 12b: tile-JSON loader, 16-bit/4-band TIFF handling, CSV class_list, inspect_data."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from anveshak.data.classes import load_class_id_map, load_class_list
from anveshak.data.formats import detect_format, load_annotations, load_image_dir
from anveshak.data.records import load_image
from anveshak.data.split import make_split, save_split
from anveshak.training.run import run_stage
from anveshak.utils.config import REPO_ROOT, load_config


def test_class_list_csv_excludes_unknown_and_maps_ids(tmp_path):
    csv_path = tmp_path / "class_list.csv"
    csv_path.write_text(
        "class_id,label\n"
        "2,cls_hash_c\n"
        "0,cls_hash_a\n"
        "1,cls_hash_b\n"
        "3,unknown\n",
        encoding="utf-8",
    )
    names = load_class_list(csv_path)
    assert names == ["cls_hash_a", "cls_hash_b", "cls_hash_c"]
    cid_map = load_class_id_map(csv_path)
    assert cid_map["cls_hash_a"] == 0
    assert cid_map["cls_hash_b"] == 1
    assert cid_map["cls_hash_c"] == 2


def test_load_image_16bit_and_4band_tif(tmp_path):
    # 16-bit single-band TIFF
    arr16 = np.linspace(100, 60000, 64 * 64, dtype=np.uint16).reshape(64, 64)
    p16 = tmp_path / "img16.tif"
    Image.fromarray(arr16, mode="I;16").save(p16)
    out16 = load_image(p16)
    assert out16.shape == (64, 64, 3) and out16.dtype == np.uint8
    assert out16.min() == 0 and out16.max() == 255

    # 4-band RGBA TIFF (8-bit)
    arr4 = np.zeros((64, 64, 4), dtype=np.uint8)
    arr4[..., 0] = 200
    arr4[..., 1] = 100
    arr4[..., 2] = 50
    arr4[..., 3] = 255
    p4 = tmp_path / "img4.tif"
    Image.fromarray(arr4, mode="RGBA").save(p4)
    out4 = load_image(p4)
    assert out4.shape == (64, 64, 3) and out4.dtype == np.uint8
    assert int(out4[0, 0, 0]) == 200 and int(out4[0, 0, 1]) == 100 and int(out4[0, 0, 2]) == 50

    # 4-frame multi-band 16-bit TIFF
    frames = [Image.fromarray((arr16 // (i + 1)).astype(np.uint16), mode="I;16") for i in range(4)]
    pmulti = tmp_path / "multi16.tif"
    frames[0].save(pmulti, save_all=True, append_images=frames[1:])
    out_m = load_image(pmulti)
    assert out_m.shape == (64, 64, 3) and out_m.dtype == np.uint8
    assert out_m.max() > 0


def _make_tilejson_fixture(root: Path) -> None:
    (root / "class_list.csv").write_text(
        "class_id,label\n"
        "0,a1b2c3d4\n"
        "1,e5f6a7b8\n"
        "2,c9d0e1f2\n"
        "3,11223344\n"
        "4,unknown\n",
        encoding="utf-8",
    )
    set_a = root / "set_a"
    (set_a / "images").mkdir(parents=True)
    (set_a / "labels").mkdir(parents=True)
    tiles_rows = ["tile_id,width,height"]
    rng = np.random.default_rng(7)
    codes = ["a1b2c3d4", "e5f6a7b8", "c9d0e1f2", "11223344"]
    for i in range(10):
        tid = f"tile_{i:04d}"
        img = rng.integers(30, 220, size=(128, 128, 3), dtype=np.uint8)
        Image.fromarray(img).save(set_a / "images" / f"{tid}.tif")
        tiles_rows.append(f"{tid},128,128")
        if i == 9:
            # Empty tile with no label JSON file
            continue
        anns = []
        if i == 0:
            # Test clipping, <1px drop, unknown label, and xyxy format
            anns.append({"label": codes[0], "class_id": 0, "obj_id": "o0", "bbox": [-5.0, 10.0, 40.0, 50.0]})
            anns.append({"label": codes[1], "class_id": 1, "obj_id": "o1", "bbox": [10.0, 10.0, 10.5, 40.0]})  # w=0.5 -> dropped
            anns.append({"label": "unknown", "class_id": 4, "obj_id": "o2", "bbox": [60.0, 60.0, 90.0, 90.0]})
            payload = {"tile_id": tid, "width": 128, "height": 128, "bbox_format": "xyxy", "annotations": anns}
        elif i == 1:
            # Test xywh format
            anns.append({"label": codes[2], "class_id": 2, "obj_id": "o3", "bbox": [20.0, 20.0, 30.0, 25.0]})
            payload = {"tile_id": tid, "width": 128, "height": 128, "bbox_format": "xywh", "annotations": anns}
        else:
            c_idx = i % len(codes)
            anns.append({"label": codes[c_idx], "class_id": c_idx, "obj_id": f"o{i}", "bbox": [15.0, 15.0, 55.0, 55.0]})
            anns.append({"label": codes[(c_idx + 1) % len(codes)], "class_id": (c_idx + 1) % len(codes), "obj_id": f"o{i}b", "bbox": [70.0, 70.0, 105.0, 105.0]})
            payload = {"tile_id": tid, "width": 128, "height": 128, "bbox_format": "xyxy", "annotations": anns}
        (set_a / "labels" / f"{tid}.json").write_text(json.dumps(payload), encoding="utf-8")
    (set_a / "tiles.csv").write_text("\n".join(tiles_rows) + "\n", encoding="utf-8")

    test_dir = root / "test"
    (test_dir / "images").mkdir(parents=True)
    test_rows = ["tile_id,width,height"]
    for i in range(2):
        tid = f"test_{i:04d}"
        Image.fromarray(rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)).save(test_dir / "images" / f"{tid}.tif")
        test_rows.append(f"{tid},128,128")
    (test_dir / "tiles.csv").write_text("\n".join(test_rows) + "\n", encoding="utf-8")


def test_tilejson_fixture_detect_load_split_smoke(tmp_path):
    _make_tilejson_fixture(tmp_path)
    set_a = tmp_path / "set_a"
    assert detect_format(set_a) == "tilejson"

    names = load_class_list(tmp_path / "class_list.csv")
    cid_map = load_class_id_map(tmp_path / "class_list.csv")
    recs = load_annotations(set_a, "auto", names, "stem", class_id_map=cid_map)
    assert len(recs) == 10
    assert recs[0].image_id == "tile_0000"
    # Tile 0: clipped [0,10,40,50] and unknown [60,60,90,90], tiny box dropped
    assert len(recs[0].boxes) == 2
    assert recs[0].boxes[0][0] == 0.0
    assert recs[0].labels == ["a1b2c3d4", "unknown"]
    # Tile 1: xywh -> xyxy [20, 20, 50, 45]
    np.testing.assert_allclose(recs[1].boxes[0], [20.0, 20.0, 50.0, 45.0])
    # Tile 9: no JSON -> 0 boxes
    assert len(recs[9].boxes) == 0

    # Cache file created and second load works
    assert (set_a / ".tilejson_cache.pkl").is_file()
    recs2 = load_annotations(set_a, "auto", names, "stem", class_id_map=cid_map)
    assert len(recs2) == 10

    # Test dir has 0 boxes
    test_recs = load_image_dir(tmp_path / "test", "stem")
    assert len(test_recs) == 2 and all(len(r.boxes) == 0 for r in test_recs)

    # Split
    sp = make_split(recs, n_holdout=1, val_frac=0.2, seed=42, class_list=names, min_val_images_per_heldout=1)
    sp_path = tmp_path / "split.json"
    save_split(sp, sp_path)

    # inspect_data CLI
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "inspect_data.py"), "--root", str(set_a), "--class_list", str(tmp_path / "class_list.csv"), "--configured_size", "128"],
        capture_output=True, text=True, check=True,
    )
    assert "format:               tilejson" in res.stdout
    assert "boxes per tile:" in res.stdout
    assert "(empty tiles: 1)" in res.stdout

    # One smoke training step
    out_dir = tmp_path / "smoke_out"
    cfg = load_config(
        [str(REPO_ROOT / "configs" / "stage1_setA.yaml"), str(REPO_ROOT / "configs" / "smoke.yaml")],
        [
            f"data.root={tmp_path}",
            "data.class_list=class_list.csv",
            f"data.split_file={sp_path}",
            f"out_dir={out_dir}",
            "train.max_iters=2",
            "train.max_hours=null",
            "train.log_every=1",
            "train.ckpt_every=0",
            "train.eval_at_end=false",
        ],
    )
    run_stage(cfg)
    assert (out_dir / "checkpoint.pt").is_file()
