#!/usr/bin/env python
"""Preflight environment and data checks for GPU training on RTX A4000 (16 GB).

Checks:
1. CUDA available
2. Free VRAM (>= 12.0 GB recommended for 16 GB RTX A4000)
3. Data layout under data.root (set_a, set_b, test)
4. Class list parses successfully
5. Validator path (optional)
6. Output directory writable
7. Config sanity (no pretrained paths set)

Prints exactly one PASS/FAIL line per check.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch

import _bootstrap  # noqa: F401
from anveshak.data.classes import load_class_list
from anveshak.utils.config import REPO_ROOT, load_config


def run_preflight(cfg, allow_cpu: bool = False, min_vram_gb: float = 12.0) -> bool:
    all_passed = True

    def report(name: str, passed: bool, detail: str) -> None:
        nonlocal all_passed
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {name}: {detail}")

    # 1. CUDA available
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        device_name = torch.cuda.get_device_name(0)
        report("CUDA available", True, f"Found {device_name}")
    else:
        if allow_cpu:
            report("CUDA available", True, "CUDA not found (bypassed with --allow_cpu for CPU smoke test)")
        else:
            report("CUDA available", False, "torch.cuda.is_available() is False; GPU is required for real training")

    # 2. Free VRAM
    if cuda_ok:
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            free_gb = free_b / (1024**3)
            total_gb = total_b / (1024**3)
            vram_ok = free_gb >= min_vram_gb
            report("Free VRAM", vram_ok, f"{free_gb:.1f} GB free of {total_gb:.1f} GB total (>= {min_vram_gb:.1f} GB required)")
        except Exception as e:
            report("Free VRAM", False, f"Failed querying VRAM: {e}")
    else:
        if allow_cpu:
            report("Free VRAM", True, "N/A (CPU mode)")
        else:
            report("Free VRAM", False, "No CUDA device to query VRAM")

    # 3. Data layout under data.root
    root = Path(cfg.data.root) if cfg.data.root else None
    if not root or not root.exists():
        report("Data layout under data.root", False, f"data.root does not exist: {root}")
    else:
        dirs_to_check = [
            getattr(cfg.data, "set_a", "set_a"),
            getattr(cfg.data, "set_b", "set_b"),
            getattr(cfg.data, "test", "test"),
        ]
        if getattr(cfg, "stage", "") == "stage3":
            dirs_to_check.append(getattr(cfg.data, "round2", "round2"))

        missing = []
        for d in dirs_to_check:
            p = root / d if not Path(d).is_absolute() else Path(d)
            if not p.exists():
                missing.append(str(p.name))
        if missing:
            report("Data layout under data.root", False, f"Missing subdirectories under {root}: {missing}")
        else:
            report("Data layout under data.root", True, f"Verified {', '.join(dirs_to_check)} exist under {root}")

    # 4. Class list parses
    cl_path = Path(cfg.data.class_list)
    if not cl_path.is_absolute() and root:
        cl_path = root / cl_path
    if not cl_path.exists():
        report("Class list parses", False, f"Class list file not found at {cl_path}")
    else:
        try:
            classes = load_class_list(cl_path)
            if classes and len(classes) > 0:
                report("Class list parses", True, f"Successfully parsed {len(classes)} classes from {cl_path.name}")
            else:
                report("Class list parses", False, f"Class list at {cl_path} is empty")
        except Exception as e:
            report("Class list parses", False, f"Error parsing {cl_path}: {e}")

    # 5. Validator path (optional)
    val_path_str = getattr(cfg.postprocess, "validator_path", None)
    if val_path_str:
        val_path = Path(val_path_str)
        if not val_path.is_absolute() and root:
            val_path = root / val_path
        if val_path.exists():
            report("Validator path (optional)", True, f"Found validator script at {val_path}")
        else:
            report("Validator path (optional)", False, f"Configured validator not found at {val_path}")
    else:
        report("Validator path (optional)", True, "None configured (optional)")

    # 6. Output directory writable
    out_dir = Path(cfg.out_dir) if getattr(cfg, "out_dir", None) else Path("runs/preflight_test")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        test_file = out_dir / ".preflight_write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
        report("Output dir writable", True, f"Successfully verified write access to {out_dir}")
    except Exception as e:
        report("Output dir writable", False, f"Cannot write to output dir {out_dir}: {e}")

    # 7. Config sanity (no pretrained path set)
    sanity_issues = []
    backbone_weights = getattr(cfg.model.backbone, "weights", None)
    if backbone_weights is not None:
        sanity_issues.append(f"model.backbone.weights is set to {backbone_weights!r}")

    init_ckpt = getattr(cfg.init, "checkpoint", None)
    if cfg.stage == "stage1" and init_ckpt:
        sanity_issues.append(f"Stage 1 has init.checkpoint set to {init_ckpt!r} (must be random init)")

    if sanity_issues:
        report("Config sanity", False, "; ".join(sanity_issues))
    else:
        report("Config sanity", True, "No pretrained weights or invalid checkpoint paths configured")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[], help="Config YAML path(s)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Config overrides (key=val)")
    parser.add_argument("--allow_cpu", action="store_true", help="Allow CPU execution for smoke/testing")
    parser.add_argument("--min_vram_gb", type=float, default=12.0, help="Minimum free VRAM in GB (default: 12.0)")
    args = parser.parse_args()

    configs = args.config if args.config else [str(REPO_ROOT / "configs" / "base.yaml")]
    cfg = load_config(configs, args.overrides)

    ok = run_preflight(cfg, allow_cpu=args.allow_cpu, min_vram_gb=args.min_vram_gb)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
