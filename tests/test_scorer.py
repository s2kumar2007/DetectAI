"""Hand-computed cases for the local scorer and the conflict rule."""
import numpy as np
import pandas as pd
import pytest

from anveshak.eval.conflict import COLS, apply_conflict_rule
from anveshak.eval.local_scorer import class_ap, oe_score, score_predictions, voc_ap

GT_COLS = ["image_id", "class_name", "xmin", "ymin", "xmax", "ymax"]


def P(rows):
    return pd.DataFrame(rows, columns=COLS)


def G(rows):
    return pd.DataFrame(rows, columns=GT_COLS)


def test_voc_ap_hand_computed():
    # tp,fp,tp with 2 GT: recall [.5,.5,1], precision [1,.5,2/3] -> AP = .5*1 + .5*(2/3)
    rec = np.array([0.5, 0.5, 1.0])
    prec = np.array([1.0, 0.5, 2 / 3])
    assert voc_ap(rec, prec) == pytest.approx(0.5 + 0.5 * 2 / 3)
    assert voc_ap(np.array([1.0]), np.array([1.0])) == pytest.approx(1.0)
    assert voc_ap(np.array([]), np.array([])) == 0.0


def test_oe_formula():
    assert oe_score(0.5, 0.5) == pytest.approx(50.0)
    assert oe_score(1.0, 0.0) == 0.0
    assert oe_score(0.0, 0.0) == 0.0
    assert oe_score(0.9, 0.3) == pytest.approx(100 * 2 * 0.9 * 0.3 / 1.2)


def test_full_example_by_hand():
    gts = G([
        ["i1", "A", 0, 0, 10, 10], ["i1", "A", 20, 20, 30, 30], ["i1", "unknown", 50, 50, 60, 60],
        ["i2", "B", 0, 0, 10, 10],
    ])
    preds = P([
        ["i1", "A", 0.9, 0, 0, 10, 10],        # TP
        ["i1", "A", 0.8, 40, 40, 48, 48],      # FP
        ["i1", "A", 0.7, 20, 20, 30, 30],      # TP
        ["i2", "B", 0.6, 0, 0, 10, 10],        # TP
        ["i1", "unknown", 0.5, 50, 50, 60, 60],  # matches unknown GT
        ["i1", "unknown", 0.4, 70, 70, 80, 80],  # FP
        ["i2", "unknown", 0.3, 30, 30, 40, 40],  # FP
    ])
    r = score_predictions(preds, gts, known_classes=["A", "B", "C"])
    ap_a = 0.5 * 1.0 + 0.5 * (2 / 3)
    assert r.ap_per_class["A"] == pytest.approx(ap_a)
    assert r.ap_per_class["B"] == pytest.approx(1.0)
    assert r.skipped_classes == ["C"]                         # known class without GT is skipped and listed
    assert r.map_k == pytest.approx((ap_a + 1.0) / 2)
    assert (r.u_rec, r.u_prec) == (pytest.approx(1.0), pytest.approx(1 / 3))
    assert r.u_f1 == pytest.approx(0.5)
    assert r.oe_score == pytest.approx(100 * 2 * r.map_k * 0.5 / (r.map_k + 0.5))
    assert r.oe_score == pytest.approx(64.70588, abs=1e-4)


def test_known_prediction_on_unknown_gt_is_false_positive():
    gts = G([["i1", "A", 0, 0, 10, 10], ["i1", "unknown", 50, 50, 60, 60]])
    preds = P([["i1", "A", 0.9, 50, 50, 60, 60]])  # class A box sitting on the unknown object
    r = score_predictions(preds, gts, known_classes=["A"])
    assert r.ap_per_class["A"] == 0.0 and r.map_k == 0.0
    assert r.u_rec == 0.0 and r.u_prec == 0.0 and r.oe_score == 0.0


def test_duplicate_prediction_is_fp_and_one_gt_matched_once():
    gts = G([["i1", "A", 0, 0, 10, 10]])
    preds = P([["i1", "A", 0.9, 0, 0, 10, 10], ["i1", "A", 0.8, 0, 0, 10, 10]])
    # rec [1,1] prec [1,.5] -> AP 1.0 ; second prediction is a FP (cannot match the same GT twice)
    assert class_ap(preds[["image_id", "class_name", "score", "xmin", "ymin", "xmax", "ymax"]], gts, 0.5) == pytest.approx(1.0)
    # unknown version: 2 preds on 1 unknown GT -> U-Prec 1/2
    gu = G([["i1", "unknown", 0, 0, 10, 10]])
    pu = P([["i1", "unknown", 0.9, 0, 0, 10, 10], ["i1", "unknown", 0.8, 0, 0, 10, 10]])
    r = score_predictions(pu, gu, known_classes=["A"])
    assert r.u_rec == 1.0 and r.u_prec == 0.5


def test_iou_threshold_boundary():
    gts = G([["i1", "A", 0, 0, 10, 10]])
    just_ok = P([["i1", "A", 0.9, 0, 0, 10, 5]])     # IoU exactly 0.5 -> match
    just_bad = P([["i1", "A", 0.9, 0, 0, 10, 4.9]])  # IoU 0.49 -> no match
    assert score_predictions(just_ok, gts, ["A"]).map_k == pytest.approx(1.0)
    assert score_predictions(just_bad, gts, ["A"]).map_k == 0.0


def test_score_ties_break_by_row_order():
    gts = G([["i1", "A", 0, 0, 10, 10]])
    # two equal-score preds; first row is the correct one -> TP first -> AP 1; reversed rows -> FP first
    good_first = P([["i1", "A", 0.5, 0, 0, 10, 10], ["i1", "A", 0.5, 50, 50, 60, 60]])
    bad_first = P([["i1", "A", 0.5, 50, 50, 60, 60], ["i1", "A", 0.5, 0, 0, 10, 10]])
    assert score_predictions(good_first, gts, ["A"]).map_k == pytest.approx(1.0)
    assert score_predictions(bad_first, gts, ["A"]).map_k == pytest.approx(0.5)  # rec[0,1] prec[0,.5]


def test_empty_cases_do_not_crash():
    gts = G([["i1", "A", 0, 0, 10, 10]])
    r = score_predictions(P([]), gts, ["A"])
    assert r.map_k == 0.0 and r.oe_score == 0.0 and r.n_unknown_gt == 0
    r2 = score_predictions(P([["i1", "A", 0.5, 0, 0, 10, 10]]), G([]), ["A"])
    assert r2.skipped_classes == ["A"] and r2.map_k == 0.0


# ----------------------------------------------------------------------------- conflict rule
def test_conflict_rule_cases():
    rows = [
        ["i1", "A", 0.6, 0, 0, 10, 10],            # 0 loses to row 1 (IoU exactly 0.5, higher score)
        ["i1", "unknown", 0.7, 0, 0, 10, 5],       # 1 survives
        ["i1", "B", 0.9, 100, 100, 110, 110],      # 2 survives (higher)
        ["i1", "unknown", 0.5, 100, 100, 110, 110],  # 3 removed
        ["i1", "C", 0.5, 200, 200, 210, 210],      # 4 tie -> earlier row wins (known)
        ["i1", "unknown", 0.5, 200, 200, 210, 210],  # 5 removed
        ["i1", "unknown", 0.5, 300, 300, 310, 310],  # 6 tie -> earlier row wins (unknown)
        ["i1", "A", 0.5, 300, 300, 310, 310],      # 7 removed
        ["i1", "A", 0.9, 400, 400, 410, 410],      # 8 known-known untouched
        ["i1", "B", 0.8, 400, 400, 410, 410],      # 9 untouched
        ["i1", "unknown", 0.9, 500, 500, 510, 510],  # 10 unknown-unknown untouched
        ["i1", "unknown", 0.8, 500, 500, 510, 510],  # 11 untouched
        ["i1", "A", 0.9, 600, 600, 610, 610],      # 12 survives
        ["i1", "unknown", 0.8, 600, 600, 610, 606],  # 13 IoU .6 -> removed
        ["i1", "A", 0.9, 700, 700, 710, 710],      # 14 IoU .4 < .5 -> both survive
        ["i1", "unknown", 0.95, 700, 700, 710, 704],  # 15
        ["i2", "A", 0.1, 0, 0, 10, 10],            # 16 other image: no conflict with row 1
    ]
    out = apply_conflict_rule(P(rows))
    kept = [tuple(r) for r in out[["image_id", "class_name", "score", "xmin"]].itertuples(index=False)]
    expected_idx = [1, 2, 4, 6, 8, 9, 10, 11, 12, 14, 15, 16]
    assert kept == [(rows[i][0], rows[i][1], rows[i][2], rows[i][3]) for i in expected_idx]


def test_conflict_rule_is_pairwise_on_original_set():
    # K1(.9) beats U(.8); U(.8) beats K2(.7) even though U is itself removed (documented interpretation A12)
    rows = [
        ["i", "A", 0.9, 0, 0, 10, 10],
        ["i", "unknown", 0.8, 0, 0, 10, 6],
        ["i", "B", 0.7, 0, 2, 10, 8],
    ]
    out = apply_conflict_rule(P(rows))
    assert list(out["class_name"]) == ["A"]


def test_scorer_applies_conflict_before_scoring():
    gts = G([["i1", "A", 0, 0, 10, 10], ["i1", "unknown", 0, 0, 10, 6]])
    preds = P([["i1", "A", 0.9, 0, 0, 10, 10], ["i1", "unknown", 0.5, 0, 0, 10, 6]])
    with_rule = score_predictions(preds, gts, ["A"], apply_conflict=True)
    without = score_predictions(preds, gts, ["A"], apply_conflict=False)
    assert with_rule.n_pred_after_conflict == 1 and with_rule.u_rec == 0.0   # unknown pred removed
    assert without.n_pred_after_conflict == 2 and without.u_rec == 1.0
