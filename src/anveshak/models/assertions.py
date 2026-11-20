"""Start-up assertions enforcing hard rule 1 (no pretrained weights) and rule 3 (same lineage across stages)."""
from __future__ import annotations

import copy
from typing import Callable, Mapping, Sequence

import torch

from ..utils.hashing import state_dict_hash
from ..utils.log import get_logger
from ..utils.seed import seed_everything
from .detector import arch_signature

log = get_logger("anveshak.assert")
CREATED_BY = "anveshak-owod-rs"


class PretrainedWeightError(RuntimeError):
    """Raised when a pretrained / foreign checkpoint would be used."""


def init_weights_hash(cfg, build_fn: Callable, seed: int) -> str:
    """Hash of a freshly seeded random initialisation."""
    seed_everything(seed, deterministic=False)
    model = build_fn(cfg)
    return state_dict_hash(model.state_dict())


def assert_stage1_random_init(cfg, model, build_fn: Callable, seed: int) -> str:
    """Stage 1 must start from random init: no checkpoint configured AND the model's weights are bit-identical to a
    fresh, seeded random init (so nothing was loaded into it). Prints/returns the SHA-256 of the initial weights."""
    ck = cfg.init.get("checkpoint") if hasattr(cfg.init, "get") else None
    if ck:
        raise PretrainedWeightError(f"Stage 1 must not load a checkpoint (init.checkpoint={ck!r})")
    current = state_dict_hash(model.state_dict())
    reference = init_weights_hash(cfg, build_fn, seed)
    if current != reference:
        raise PretrainedWeightError(
            "Stage 1 model weights differ from a fresh seeded random init - something modified/loaded the weights."
        )
    msg = f"[no-pretrained check] Stage 1 initial weights SHA-256 = {current}  (random init, seed={seed})"
    print(msg)
    log.info(msg)
    return current


def assert_prev_checkpoint_ok(meta: Mapping, cfg, allowed_stages: Sequence[str], path: str) -> None:
    """Stage 2/3 may ONLY load our own previous-stage checkpoint with a matching architecture."""
    if "://" in str(path):
        raise PretrainedWeightError(f"refusing to load a remote checkpoint: {path}")
    if meta.get("created_by") != CREATED_BY:
        raise PretrainedWeightError(f"{path} was not produced by {CREATED_BY}; foreign/pretrained weights are forbidden")
    if meta.get("stage") not in allowed_stages:
        raise PretrainedWeightError(f"{path} is a {meta.get('stage')!r} checkpoint; expected one of {list(allowed_stages)}")
    want, got = arch_signature(cfg), meta.get("arch")
    if got != want:
        diff = {k: (got or {}).get(k) for k in want if (got or {}).get(k) != want[k]}
        raise PretrainedWeightError(
            f"architecture mismatch with {path}: checkpoint has {diff}, config wants "
            f"{ {k: want[k] for k in diff} } - the same architecture must be kept across stages (rule 3)"
        )
