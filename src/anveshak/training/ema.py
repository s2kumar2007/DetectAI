"""Exponential moving average of model weights."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9998, warmup: bool = True, updates: int = 0):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.warmup, self.updates = float(decay), bool(warmup), int(updates)

    def current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (1 + self.updates) / (10 + self.updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.current_decay()
        msd = model.state_dict()
        for k, e in self.module.state_dict().items():
            m = msd[k].detach()
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.to(e.dtype), alpha=1.0 - d)
            else:
                e.copy_(m)

    @torch.no_grad()
    def copy_from(self, model: nn.Module) -> None:
        self.module.load_state_dict(model.state_dict())
