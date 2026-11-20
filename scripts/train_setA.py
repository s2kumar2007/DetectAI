#!/usr/bin/env python
"""Stage 1 - train from RANDOM INIT on Set A (no pretrained weights; asserts and prints the initial-weights hash).

    python scripts/train_setA.py --config configs/stage1_setA.yaml --set data.root=/path/to/data --max_hours 8
"""
import json

from _common import stage_args

from anveshak.training.run import run_stage


def main() -> None:
    cfg, a = stage_args(__doc__, "configs/stage1_setA.yaml", "stage1")
    res = run_stage(cfg, max_hours=a.max_hours, resume=not a.no_resume)
    print(json.dumps(res["summary"], indent=2))


if __name__ == "__main__":
    main()
