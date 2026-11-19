"""Seeding and determinism helpers."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch. Optionally request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        set_determinism()


def set_determinism(warn_only: bool = True) -> None:
    """Deterministic algorithms where possible; cudnn.benchmark off."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except Exception:  # very old torch
        pass


def worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    base = info.seed if info is not None else worker_id
    random.seed(base % (2**32))
    np.random.seed(base % (2**32 - 1))
