import json

import pandas as pd

from anveshak.data.classes import load_class_list
from anveshak.data.formats import load_annotations
from anveshak.data.split import load_split, make_split, save_split, train_view, val_view
from anveshak.eval.io import records_to_gt


def test_split_holds_out_classes_and_is_deterministic(syn_root, tmp_path):
    names = load_class_list(syn_root / "class_list.txt")
    recs = load_annotations(syn_root / "set_a", "auto", names)
    s1 = make_split(recs, n_holdout=2, val_frac=0.2, seed=3, class_list=names)
    s2 = make_split(recs, n_holdout=2, val_frac=0.2, seed=3, class_list=names)
    assert s1 == s2
    assert len(s1["heldout_classes"]) == 2
    assert set(s1["train_ids"]).isdisjoint(s1["val_ids"])
    assert set(s1["train_ids"]) | set(s1["val_ids"]) == {r.image_id for r in recs}
    held = set(s1["heldout_classes"])

    tr, va = train_view(recs, s1), val_view(recs, s1)
    assert {r.image_id for r in tr} == set(s1["train_ids"]) and {r.image_id for r in va} == set(s1["val_ids"])
    # labels removed from training, renamed 'unknown' in validation
    assert not (held & {l for r in tr for l in r.labels})
    val_labels = {l for r in va for l in r.labels}
    assert "unknown" in val_labels and not (held & val_labels)
    # the held-out objects still exist as boxes in the originals (they are only unlabelled in training)
    orig = {r.image_id: r for r in recs}
    n_orig = sum(sum(l in held for l in orig[r.image_id].labels) for r in tr)
    n_removed = sum(len(orig[r.image_id].labels) - len(r.labels) for r in tr)
    assert n_orig == n_removed and n_removed > 0
    # GT frame for the scorer
    gt = records_to_gt(va)
    assert (gt["class_name"] == "unknown").sum() > 0

    p = tmp_path / "split.json"
    save_split(s1, p)
    assert load_split(p) == json.loads(p.read_text())


def test_split_zero_holdout_gives_only_validation_subset(syn_root):
    names = load_class_list(syn_root / "class_list.txt")
    recs = load_annotations(syn_root / "set_a", "auto", names)
    s = make_split(recs, n_holdout=0, val_frac=0.1, seed=0, class_list=names)
    assert s["heldout_classes"] == [] and 0 < len(s["val_ids"]) < len(recs)
