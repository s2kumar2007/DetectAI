"""Shared CLI plumbing for the train_* scripts."""
import argparse
from typing import List, Optional

import _bootstrap  # noqa: F401
from anveshak.utils.config import load_config


def stage_args(description: str, default_config: str, stage: str, argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", default=[default_config], help="YAML file(s); merged after configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="dotted overrides, e.g. data.root=/data train.max_iters=100")
    ap.add_argument("--out", default=None, help="output directory (overrides out_dir)")
    ap.add_argument("--max_hours", type=float, default=None, help="wall-clock budget; training stops cleanly and the LR decays over it")
    ap.add_argument("--no_resume", action="store_true", help="ignore <out>/last.pt")
    a = ap.parse_args(argv)
    ov = list(a.set) + ([f"out_dir={a.out}"] if a.out else []) + [f"stage={stage}"]
    cfg = load_config(a.config, ov)
    return cfg, a
