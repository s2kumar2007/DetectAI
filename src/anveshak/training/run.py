"""Entry points used by scripts/train_*.py."""
from __future__ import annotations

from typing import Any, Dict, Optional

from .engine import resolve_device, train
from .stages import prepare_stage1


def run_stage(cfg, max_hours: Optional[float] = None, resume: bool = True) -> Dict[str, Any]:
    device = resolve_device(str(cfg.train.device))
    if cfg.stage == "stage1":
        state = prepare_stage1(cfg)
    elif cfg.stage in ("stage2", "stage3"):
        from .continual import prepare_continual  # phase 6/7

        state = prepare_continual(cfg, device)
    else:
        raise ValueError(f"unknown stage {cfg.stage!r}")
    return train(state, device, max_hours=max_hours, resume=resume)
