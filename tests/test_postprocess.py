"""Phase 5: WBF, TTA geometry, unknown gate, conflict-aware finalisation, CSV writer/validator, determinism."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from anveshak.data.build import load_test_images
from anveshak.eval.conflict import apply_conflict_rule
from anveshak.postprocess.csv_check import CSVValidationError, check_csv
from anveshak.postprocess.factory import build_predictor, predict_test_set
from anveshak.postprocess.tta import VIEWS, apply_view, unmap_boxes
from anveshak.postprocess.unknown import Calibration, unknown_detections, unknown_scores
from anveshak.postprocess.wbf import weighted_boxes_fusion
from anveshak.postprocess.writer import validate_submission, write_predictions_csv
from anveshak.utils.config import load_config


def test_wbf_hand_computed():
    b0 = np.array([[0, 0, 10, 10.0]]); b1 = np.array([[2, 0, 12, 10.0]]); far = np.array([[50, 50, 60, 60.0]])
    boxes, scores, labels = weighted_boxes_fusion([b0, b1, far], [np.array([0.8]), np.array([0.4]), np.array([0.9])],
                                                  [np.array([1]), np.array([1]), np.array([1])], n_views=3, iou_thr=0.5)
    # cluster {b0,b1}: IoU = 80/120 = .667 > .5; x1 = (0*.8+2*.4)/1.2, x2 = (10*.8+12*.4)/1.2; score = mean(.8,.4)*2/3
    assert len(boxes) == 2
    i = int(np.argmin(np.abs(scores - 0.4)))
    assert np.allclose(boxes[i], [0.8 / 1.2, 0, 12.8 / 1.2, 10], atol=1e-4) and scores[i] == pytest.approx(0.6 * 2 / 3, abs=1e-5)
    j = 1 - i
    assert np.allclose(boxes[j], far[0]) and scores[j] == pytest.approx(0.9 / 3, abs=1e-5)  # seen by 1 of 3 views
    # different labels never merge; same view never merges twice
    b, s, l = weighted_boxes_fusion([np.vstack([b0, b0]), b0], [np.array([.9, .8]), np.array([.7])],
                                    [np.array([1, 2]), np.array([1])], 2, 0.5)
    assert sorted(l.tolist()) == [1, 2]
    b, s, l = weighted_boxes_fusion([np.vstack([b0, b0])], [np.array([.9, .8])], [np.array([1, 1])], 1, 0.5)
    assert len(b) == 2
    assert weighted_boxes_fusion([], [], [], 4)[0].shape == (0, 4)


def test_wbf_is_deterministic():
    rng = np.random.default_rng(0)
    bx = [rng.uniform(0, 50, (6, 2)) for _ in range(4)]
    bs = [np.hstack([c, c + rng.uniform(5, 20, (6, 2))]) for c in bx]
    sc = [rng.uniform(0.1, 1, 6) for _ in range(4)]
    lb = [rng.integers(0, 2, 6) for _ in range(4)]
    a = weighted_boxes_fusion(bs, sc, lb, 4, 0.4)
    b = weighted_boxes_fusion(bs, sc, lb, 4, 0.4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("view", VIEWS)
def test_tta_view_roundtrip_geometry(view):
    S = 32
    img = torch.zeros(3, S, S, dtype=torch.uint8)
    img[:, 4:12, 20:30] = 255  # object at x 20..30, y 4..12
    v = apply_view(img, view)
    ys, xs = torch.nonzero(v[0] > 0, as_tuple=True)
    box_in_view = np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], np.float32)
    assert np.allclose(unmap_boxes(box_in_view, view, S), [[20, 4, 30, 12]])


def test_unknown_gate_formula():
    c = Calibration(alpha=2.0, tau_unk=0.2, unk_cap=20, cos_ref=0.8)
    obj, cos = torch.tensor([0.5, 0.5, 0.9]), torch.tensor([0.0, 0.4, 0.8])
    # S_known = clamp(cos/0.8) = [0, .5, 1] -> score = obj*(1-S)*alpha = [1.0, .5, 0.0]
    assert torch.allclose(unknown_scores(obj, cos, c), torch.tensor([1.0, 0.5, 0.0]))
    raw = {"obj": obj, "cos_max": cos, "boxes": torch.tensor([[0, 0, 10, 10.], [20, 20, 30, 30.], [40, 40, 50, 50.]])}
    d = unknown_detections(raw, c)
    assert len(d["scores"]) == 2 and np.allclose(sorted(d["scores"]), [0.5, 1.0])  # third (score 0) is below tau


@pytest.fixture(scope="module")
def predictor(tiny_stage1, syn_root):
    cfg = load_config(overrides=[f"data.root={syn_root}", "calibration.tau_unk=0.05", "calibration.cos_ref=0.6"])
    pred, sm, meta = build_predictor(tiny_stage1 / "checkpoint.pt", cfg, None, "cpu")
    return cfg, pred, sm


def test_end_to_end_rows_obey_all_rules(predictor, syn_root, tmp_path):
    cfg, pred, sm = predictor
    recs = load_test_images(cfg)[:10]
    df = predict_test_set(pred, recs)
    assert set(df["class_name"]) <= set(sm.names) | {"unknown"}
    sizes = {r.image_id: (r.width, r.height) for r in recs}
    out = write_predictions_csv(df, tmp_path / "p.csv")
    info = validate_submission(out, sm.names, sizes, cfg.postprocess)
    assert info["rows"] == len(df)
    assert out.read_text().splitlines()[0] == "image_id,class_name,score,xmin,ymin,xmax,ymax"
    # conflict rule is already satisfied: re-applying it removes nothing (no known/unknown pair left at IoU >= .5)
    assert len(apply_conflict_rule(pd.read_csv(out))) == len(df)
    assert df.groupby("image_id").size().max() <= 1000
    assert (df[df.class_name == "unknown"].groupby("image_id").size().max() or 0) <= pred.calib.unk_cap


def test_unknown_cap_is_enforced(predictor):
    cfg, pred, sm = predictor
    recs = load_test_images(cfg)[:6]
    loose = Calibration(alpha=4.0, tau_unk=0.0, unk_cap=3, cos_ref=2.0)
    pred.calib = loose
    df = pred.predict_records(recs)
    n = df[df.class_name == "unknown"].groupby("image_id").size()
    assert len(n) and n.max() <= 3
    pred.calib = Calibration(1.0, 0.05, 20, 0.6)


def test_inference_is_deterministic(predictor, tmp_path):
    cfg, pred, sm = predictor
    recs = load_test_images(cfg)[:8]
    a = write_predictions_csv(pred.predict_records(recs), tmp_path / "a.csv")
    b = write_predictions_csv(pred.predict_records(recs), tmp_path / "b.csv")
    assert a.read_bytes() == b.read_bytes()
    assert torch.backends.cudnn.benchmark is False


def test_tta_switch_changes_view_count(predictor):
    cfg, pred, sm = predictor
    assert pred.views == VIEWS
    rec = load_test_images(cfg)[0]
    from anveshak.postprocess.engine import Predictor
    single = Predictor(pred.model, pred.slot_to_name, pred.size, cfg.infer, pred.calib, "cpu", tta=False)
    assert single.views == ("orig",) and isinstance(single.predict_record(rec), list)


def test_validator_integration_external_and_builtin(tmp_path, predictor):
    cfg, pred, sm = predictor
    good = tmp_path / "good.csv"
    good.write_text("image_id,class_name,score,xmin,ymin,xmax,ymax\nimg1,unknown,0.5,1.00,1.00,9.00,9.00\n")
    sizes = {"img1": (20, 20)}
    fake = tmp_path / "validate_predictions.py"
    fake.write_text("import sys\nprint('checked', sys.argv[1])\nsys.exit(0)\n")
    pc = load_config(overrides=[f"postprocess.validator_path={fake}"]).postprocess
    assert validate_submission(good, sm.names, sizes, pc)["external_ok"] is True
    fake.write_text("import sys\nprint('bad format')\nsys.exit(3)\n")
    with pytest.raises(CSVValidationError):
        validate_submission(good, sm.names, sizes, pc)
    # validator absent -> built-in checker runs and fails loudly on a violation
    bad = tmp_path / "bad.csv"
    bad.write_text("image_id,class_name,score,xmin,ymin,xmax,ymax\nimg1,UNKNOWN,1.5,9,1,1,9\n")
    with pytest.raises(CSVValidationError):
        validate_submission(bad, sm.names, sizes, load_config().postprocess)


def test_infer_script_end_to_end_and_scoring(tiny_stage1, syn_root, tmp_path):
    out = tmp_path / "sub" / "predictions.csv"
    r = subprocess.run([sys.executable, "scripts/infer_and_validate.py", "--checkpoint", str(tiny_stage1 / "checkpoint.pt"),
                        "--set", f"data.root={syn_root}", "calibration.tau_unk=0.05", "--out", str(out),
                        "--gt", str(syn_root / "test_gt_round1.csv")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "validation PASSED" in r.stdout and "OE" in r.stdout
    meta = json.loads((tmp_path / "sub" / "predictions.csv.meta.json").read_text())
    assert meta["n_images"] == 24 and meta["tta"] is True
