"""Hashes used for reproducibility records."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Mapping

import torch


def state_dict_hash(state: Mapping[str, torch.Tensor]) -> str:
    """SHA-256 over (name, dtype, shape, bytes) of every tensor, in sorted key order."""
    h = hashlib.sha256()
    for key in sorted(state.keys()):
        t = state[key]
        if not torch.is_tensor(t):
            continue
        t = t.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def git_commit_hash(repo: str | Path | None = None) -> str:
    """Current git commit hash, or 'nogit' if unavailable."""
    cwd = str(repo) if repo else str(Path(__file__).resolve().parents[3])
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "nogit"
