#!/usr/bin/env python
"""Create the simulated-unknown validation split from a TRAINING set (never from test images).

Holds out N known classes as fake unknowns (labels removed from training, kept in validation as "unknown") plus a
validation image subset. Example:
    python scripts/make_val_split.py --root /path/to/anveshak_data/set_a --class_list /path/to/class_list.txt \
        --n_holdout 6 --val_frac 0.15 --seed 1337 --out splits/set_a_split.json
"""
import argparse

import _bootstrap  # noqa: F401
from anveshak.data.classes import load_class_list
from anveshak.data.formats import load_annotations
from anveshak.data.split import make_split, save_split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="training set directory (e.g. .../set_a)")
    ap.add_argument("--class_list", required=True)
    ap.add_argument("--format", default="auto")
    ap.add_argument("--image_id_mode", default="stem")
    ap.add_argument("--n_holdout", type=int, default=6)
    ap.add_argument("--holdout_classes", nargs="*", default=None, help="Explicit class IDs or names to hold out (e.g. 11 14 17 21 23 24)")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="splits/split.json")
    a = ap.parse_args()
    names = load_class_list(a.class_list)
    recs = load_annotations(a.root, a.format, names, a.image_id_mode)
    split = make_split(recs, a.n_holdout, a.val_frac, a.seed, names, holdout_classes=a.holdout_classes)
    save_split(split, a.out)
    print(f"images: {len(recs)}  train: {len(split['train_ids'])}  val: {len(split['val_ids'])}")
    print(f"held-out (fake unknown) classes: {split['heldout_classes']}")
    print(f"split written to {a.out}")


if __name__ == "__main__":
    main()
