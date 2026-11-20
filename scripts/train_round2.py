#!/usr/bin/env python
"""Stage 3 / Round 2 (on-site) - start ONLY from the verified Round 1 checkpoint, learn the 10 new few-shot classes
(slots 40..49) in a short, time-budgeted run: imprint first, backbone stages 1-2 frozen, replay from Sets A+B,
teacher = Round 1 checkpoint (pseudo-labels + distillation).

    python scripts/verify_checkpoint.py --checkpoint runs/stage2/checkpoint.pt --csv submission/predictions.csv --set data.root=...
    python scripts/train_round2.py --config configs/stage3_round2.yaml --set data.root=/path/to/data \
        init.checkpoint=runs/stage2/checkpoint.pt --max_hours 0.5
"""
import json

from _common import stage_args

from anveshak.training.run import run_stage


def main() -> None:
    cfg, a = stage_args(__doc__, "configs/stage3_round2.yaml", "stage3")
    res = run_stage(cfg, max_hours=a.max_hours, resume=not a.no_resume)
    print(json.dumps(res["summary"], indent=2))
    print(json.dumps({k: v for k, v in res["report"].items() if not isinstance(v, dict)}, indent=2, default=str))


if __name__ == "__main__":
    main()
