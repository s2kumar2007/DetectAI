"""Checkpoint IO. A checkpoint is ONLY ever produced/consumed by this project (``created_by`` tag)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from ..data.slots import SlotMap
from ..models.assertions import CREATED_BY
from ..models.detector import OWODDetector, arch_signature, build_model
from ..utils.config import Config
from ..utils.hashing import file_sha256


def atomic_save(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_final(path: str | Path, payload: Dict[str, Any], slotmap: SlotMap) -> str:
    """Write checkpoint.pt, class_slots.json (beside it) and checkpoint.pt.sha256. Returns the SHA-256."""
    path = Path(path)
    atomic_save(payload, path)
    slotmap.save(path.parent / "class_slots.json")
    digest = file_sha256(path)
    (path.parent / (path.name + ".sha256")).write_text(digest + "\n", encoding="utf-8")
    return digest


def load_payload(path: str | Path, map_location="cpu") -> Dict[str, Any]:
    return torch.load(str(path), map_location=map_location, weights_only=True)


def payload_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in payload.items() if k not in ("model", "ema", "optimizer")}


def model_from_checkpoint(path: str | Path, device="cpu", use: str = "ema") -> Tuple[OWODDetector, Dict[str, Any], SlotMap]:
    """Rebuild the detector from the config stored in the checkpoint and load EMA (default) or raw weights."""
    payload = load_payload(path, map_location="cpu")
    if payload.get("created_by") != CREATED_BY:
        raise RuntimeError(f"{path} was not produced by {CREATED_BY}")
    cfg = Config(payload["config"])
    model = build_model(cfg)
    model.load_state_dict(payload[use])
    model.to(device).eval()
    return model, payload_meta(payload), SlotMap.from_dict(payload["slot_map"])
