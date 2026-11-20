#!/usr/bin/env python
"""Stage 2 - continue the SAME model on Set B (few-shot classes): imprinting, teacher pseudo-labels, Set A replay,
feature + logit distillation, frozen stem/early stages.

    python scripts/train_setB.py --config configs/stage2_setB.yaml --set data.root=/path/to/data \
        init.checkpoint=runs/stage1/checkpoint.pt --max_hours 2
"""
import json

from _common import stage_args

from anveshak.training.run import run_stage


def main() -> None:
    cfg, a = stage_args(__doc__, "configs/stage2_setB.yaml", "stage2")
    res = run_stage(cfg, max_hours=a.max_hours, resume=not a.no_resume)
    print(json.dumps(res["summary"], indent=2))
    print(json.dumps({k: v for k, v in res["report"].items() if not isinstance(v, dict)}, indent=2, default=str))


if __name__ == "__main__":
    main()
