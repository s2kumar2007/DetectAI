"""Stage-generic training engine: AdamW, warmup + (time-aware) cosine LR, EMA, BF16 autocast, channels_last,
gradient accumulation, periodic checkpoints with resume, and a wall-clock budget that ends training cleanly."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler

from ..data.dataset import collate
from ..data.records import ImageRecord
from ..data.slots import SlotMap
from ..models.assertions import CREATED_BY, init_weights_hash
from ..models.detector import OWODDetector, arch_signature, build_model
from ..utils.config import Config
from ..utils.hashing import git_commit_hash
from ..utils.log import get_logger
from ..utils.seed import seed_everything, worker_init_fn
from .checkpoint import atomic_save, load_payload, save_final
from .ema import ModelEMA
from .schedule import lr_factor

LossFn = Callable[[OWODDetector, Sequence[torch.Tensor], Sequence[dict]], Dict[str, torch.Tensor]]


@dataclass
class StageState:
    cfg: Config
    stage: str
    model: OWODDetector
    slotmap: SlotMap
    train_dataset: Dataset
    val_records: List[ImageRecord] = field(default_factory=list)
    split: Optional[dict] = None
    init_hash: str = ""
    sample_weights: Optional[Sequence[float]] = None
    old_slots: List[int] = field(default_factory=list)       # slots whose prototypes stay frozen (continual stages)
    compute_losses: Optional[LossFn] = None                  # defaults to model.forward_train
    setup_fn: Optional[Callable[[OWODDetector], None]] = None  # run once on a FRESH start (e.g. imprinting)
    report_fn: Optional[Callable[[OWODDetector, torch.device], Dict[str, Any]]] = None  # retention report
    extra_meta: Dict[str, Any] = field(default_factory=dict)


def _infinite_batches(ds: Dataset, batch_size: int, workers: int, seed: int, weights=None) -> Iterator:
    epoch = 0
    while True:
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)
        g = torch.Generator().manual_seed(seed * 1000 + epoch)
        if weights is not None:
            sampler = WeightedRandomSampler(list(weights), num_samples=len(ds), replacement=True, generator=g)
        else:
            sampler = RandomSampler(ds, generator=g)
        dl = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=workers, collate_fn=collate,
                        drop_last=len(ds) >= batch_size, worker_init_fn=worker_init_fn)
        yield from dl
        epoch += 1


def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    t = cfg.train
    groups = []
    for prefix in ("backbone", "fpn", "rpn", "roi_head"):
        mult = float(t.lr_mult.get(prefix, 1.0))
        dec, nodec = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad or not n.startswith(prefix + "."):
                continue
            (nodec if (p.ndim <= 1 or "prototypes" in n or "log_scale" in n) else dec).append(p)
        for params, wd in ((dec, float(t.weight_decay)), (nodec, 0.0)):
            if params:
                groups.append({"params": params, "weight_decay": wd, "lr": float(t.lr) * mult, "base_lr": float(t.lr) * mult})
    return torch.optim.AdamW(groups, lr=float(t.lr), betas=(0.9, 0.999))


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def resolve_device(name: str) -> torch.device:
    return torch.device("cuda" if (name == "auto" and torch.cuda.is_available()) else ("cpu" if name == "auto" else name))


def train(state: StageState, device: torch.device, max_hours: Optional[float] = None, resume: bool = True) -> Dict[str, Any]:
    cfg, t = state.cfg, state.cfg.train
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger("anveshak.train", out / "train.log")
    model = state.model.to(device)
    model.channels_last = bool(t.channels_last) and device.type == "cuda"
    if model.channels_last:
        model.to(memory_format=torch.channels_last)
    if int(cfg.freeze.backbone_stages) > 0:
        model.backbone.freeze_stages(int(cfg.freeze.backbone_stages))
    amp = (device.type == "cuda") if str(t.amp) == "auto" else bool(t.amp)
    max_iters = int(t.max_iters)
    budget_s = float(max_hours if max_hours is not None else (t.max_hours or 0)) * 3600.0

    ema = ModelEMA(model, float(t.ema_decay), bool(t.ema_warmup))
    start_it, elapsed0, resumed = 0, 0.0, False
    last_path = out / "last.pt"
    opt = None
    if resume and last_path.exists():
        pl = load_payload(last_path, map_location="cpu")
        if pl.get("created_by") != CREATED_BY or pl.get("arch") != arch_signature(cfg) or pl.get("stage") != state.stage:
            raise RuntimeError(f"{last_path} does not match this run (stage/architecture); remove it or use another out_dir")
        if state.stage == "stage1":
            fresh = init_weights_hash(cfg, build_model, int(cfg.seed))
            if pl.get("init_hash") != fresh:
                raise RuntimeError("resume checkpoint does not descend from this seed's random init (hard rule 1)")
            seed_everything(int(cfg.seed), deterministic=False)
        model.load_state_dict(pl["model"])
        ema.module.load_state_dict(pl["ema"])
        ema.updates = int(pl["ema_updates"])
        start_it, elapsed0, resumed = int(pl["iteration"]), float(pl["elapsed_s"]), True
        state.slotmap = SlotMap.from_dict(pl["slot_map"])
        log.info("RESUMED from %s at iteration %d (%.0fs already used)", last_path, start_it, elapsed0)
    elif state.setup_fn is not None:
        state.setup_fn(model)
        ema.copy_from(model)
    opt = build_optimizer(model, cfg)
    if resumed:
        opt.load_state_dict(pl["optimizer"])
    if bool(cfg.freeze.old_prototypes) and state.old_slots:
        mask = torch.ones(model.num_slots, 1, device=device)
        mask[list(state.old_slots)] = 0.0
        model.roi_head.cls.prototypes.register_hook(lambda g: g * mask)
        log.info("old prototypes frozen for slots %s", sorted(state.old_slots))

    loss_fn: LossFn = state.compute_losses or (lambda m, im, tg: m.forward_train(im, tg)[0])
    batches = _infinite_batches(state.train_dataset, int(t.batch_size), int(cfg.data.num_workers), int(cfg.seed) + start_it,
                                state.sample_weights)
    accum = max(1, int(t.accum_steps))
    log_every = max(1, int(t.log_every))
    metrics_path = out / "metrics.jsonl"
    t0 = time.time() - elapsed0
    it, bad, stopped_by = start_it, 0, "max_iters"
    window: Dict[str, float] = {}
    wcount = 0
    model.train()
    log.info("training %s: iters=%d budget=%s amp=%s device=%s train_images=%d active_slots=%d", state.stage, max_iters,
             f"{budget_s/3600:.2f}h" if budget_s else "none", amp, device, len(state.train_dataset), len(model.active_slots))
    while True:
        elapsed = time.time() - t0
        progress = max(it / max(1, max_iters), (elapsed / budget_s) if budget_s else 0.0)
        if it >= max_iters or progress >= 1.0:
            stopped_by = "max_iters" if it >= max_iters else "time_budget"
            break
        f = lr_factor(it, int(t.warmup_iters), progress, float(t.min_lr_frac))
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f
        model.mining_on = progress >= float(cfg.model.mining.start_frac)
        opt.zero_grad(set_to_none=True)
        step_losses: Dict[str, float] = {}
        ok = True
        for _ in range(accum):
            images, targets = next(batches)
            with _autocast(device, amp):
                losses = loss_fn(model, [im.to(device) for im in images], targets)
            total = sum(losses.values())
            if not torch.isfinite(total):
                ok = False
                break
            (total / accum).backward()
            for k, v in losses.items():
                step_losses[k] = step_losses.get(k, 0.0) + float(v) / accum
            step_losses["loss"] = step_losses.get("loss", 0.0) + float(total) / accum
        if not ok:
            bad += 1
            opt.zero_grad(set_to_none=True)
            log.warning("non-finite loss at iter %d (skipped, %d in a row)", it, bad)
            if bad >= 20:
                raise FloatingPointError("20 consecutive non-finite losses - aborting")
            continue
        bad = 0
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(t.grad_clip))
        opt.step()
        ema.update(model)
        it += 1
        for k, v in step_losses.items():
            window[k] = window.get(k, 0.0) + v
        wcount += 1
        if it % log_every == 0 or it == max_iters:
            rec = {k: round(v / wcount, 5) for k, v in window.items()}
            rec.update(iter=it, lr=opt.param_groups[0]["lr"], grad_norm=round(float(gn), 3), elapsed_s=round(time.time() - t0, 1),
                       progress=round(progress, 4), mining=bool(model.mining_on))
            with open(metrics_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            log.info("it %d/%d loss %.4f %s lr %.2e |g| %.2f", it, max_iters, rec["loss"],
                     " ".join(f"{k}={v:.3f}" for k, v in rec.items() if k.startswith(("rpn_", "roi_"))), rec["lr"], rec["grad_norm"])
            window, wcount = {}, 0
        if int(t.ckpt_every) and it % int(t.ckpt_every) == 0:
            _save_last(last_path, state, model, ema, opt, it, time.time() - t0, cfg)
    elapsed = time.time() - t0
    completed = it >= max_iters
    log.info("training stopped (%s) at iteration %d after %.1f min", stopped_by, it, elapsed / 60)

    report: Dict[str, Any] = {}
    if bool(t.eval_at_end):
        ema.module.to(device)
        report = state.report_fn(ema.module, device) if state.report_fn else {}
    payload = _payload(state, model, ema, it, elapsed, cfg)
    payload.update(completed=completed, stopped_by=stopped_by)
    digest = save_final(out / "checkpoint.pt", payload, state.slotmap)
    report.update(stage=state.stage, iterations=it, stopped_by=stopped_by, checkpoint_sha256=digest)
    (out / "retention_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    summary = {"stage": state.stage, "iterations": it, "completed": completed, "stopped_by": stopped_by, "elapsed_s": round(elapsed, 1),
               "init_hash": state.init_hash, "checkpoint_sha256": digest, "active_slots": model.active_slots}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("saved %s (sha256 %s)", out / "checkpoint.pt", digest[:16])
    return {"summary": summary, "report": report, "out_dir": str(out)}


def _payload(state: StageState, model, ema, it: int, elapsed: float, cfg) -> Dict[str, Any]:
    return {
        "created_by": CREATED_BY, "stage": state.stage, "arch": arch_signature(cfg),
        "model": model.state_dict(), "ema": ema.module.state_dict(), "config": cfg.to_dict(),
        "slot_map": state.slotmap.to_dict(), "seed": int(cfg.seed), "git_commit": git_commit_hash(),
        "init_hash": state.init_hash, "split": state.split, "iteration": int(it), "elapsed_s": float(elapsed),
        "ema_updates": int(ema.updates), **state.extra_meta,
    }


def _save_last(path: Path, state, model, ema, opt, it: int, elapsed: float, cfg) -> None:
    pl = _payload(state, model, ema, it, elapsed, cfg)
    pl["optimizer"] = opt.state_dict()
    atomic_save(pl, path)
