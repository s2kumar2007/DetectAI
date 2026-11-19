#!/usr/bin/env python
"""Generate the SYNTHETIC remote-sensing-like dataset used for all tests and smoke runs.

Example:
    python scripts/make_synthetic_data.py --out data/synthetic --size 128 --n_a 96
"""
import argparse

import _bootstrap  # noqa: F401
from anveshak.data.synthetic import SyntheticSpec, generate_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--n_a", type=int, default=96)
    ap.add_argument("--n_b", type=int, default=12)
    ap.add_argument("--n_r2", type=int, default=16)
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fmt_a", default="coco", choices=["coco", "yolo", "voc", "csv"])
    ap.add_argument("--fmt_b", default="yolo", choices=["coco", "yolo", "voc", "csv"])
    ap.add_argument("--fmt_r2", default="voc", choices=["coco", "yolo", "voc", "csv"])
    a = ap.parse_args()
    spec = SyntheticSpec(size=a.size, n_a=a.n_a, n_b=a.n_b, n_r2=a.n_r2, n_test=a.n_test, seed=a.seed,
                         fmt_a=a.fmt_a, fmt_b=a.fmt_b, fmt_r2=a.fmt_r2)
    meta = generate_dataset(a.out, spec)
    print(f"synthetic dataset written to {a.out}")
    for k, v in meta.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
