"""Unit tests for Phase 11: GPU readiness, preflight checks, benchmark, and image size detection."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import _bootstrap  # noqa: F401
from anveshak.data.build import check_image_sizes, load_class_names, load_set
from anveshak.data.records import ImageRecord
from anveshak.data.synthetic import SyntheticSpec, generate_dataset
from anveshak.utils.config import REPO_ROOT, load_config
from scripts.preflight import run_preflight


def test_preflight_passes_on_valid_synthetic_data(tmp_path):
    """Preflight passes all checks when run with --allow_cpu and valid data layout."""
    syn_root = tmp_path / "syn_data"
    generate_dataset(syn_root, SyntheticSpec(size=128, n_a=12, n_b=4, n_r2=4, n_test=4, seed=42))

    cfg = load_config(
        [str(REPO_ROOT / "configs" / "stage1_setA.yaml"), str(REPO_ROOT / "configs" / "smoke.yaml")],
        [f"data.root={syn_root}", f"out_dir={tmp_path / 'out'}"],
    )

    passed = run_preflight(cfg, allow_cpu=True, min_vram_gb=1.0)
    assert passed is True, "Preflight should pass on valid synthetic dataset with allow_cpu=True"


def test_preflight_fails_on_missing_data(tmp_path):
    """Preflight fails when data root or required folders are missing."""
    empty_root = tmp_path / "non_existent_data"
    cfg = load_config(
        [str(REPO_ROOT / "configs" / "stage1_setA.yaml"), str(REPO_ROOT / "configs" / "smoke.yaml")],
        [f"data.root={empty_root}", f"out_dir={tmp_path / 'out'}"],
    )

    passed = run_preflight(cfg, allow_cpu=True)
    assert passed is False, "Preflight must fail when data root does not exist"


def test_benchmark_speed_runs_on_cpu_tiny():
    """benchmark_speed.py runs cleanly on CPU with --tiny flag."""
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "benchmark_speed.py"), "--cpu", "--tiny"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, f"benchmark_speed failed with output:\n{res.stderr}\n{res.stdout}"
    assert "OWOD Speed & VRAM Benchmark (CPU)" in res.stdout
    assert "RECOMMENDATION FOR RTX A4000 (16 GB):" in res.stdout


def test_image_size_auto_detection_warns_on_mismatch():
    """check_image_sizes logs a warning if tiles are much smaller or larger than input.size."""
    import io
    import logging
    from anveshak.data.build import log

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    log.addHandler(handler)
    try:
        # 1. Much smaller tiles (128px vs configured 800px)
        recs_small = [ImageRecord("img1", "dummy.png", 128, 128, [], [])]
        med_w, med_h = check_image_sizes(recs_small, configured_size=800)
        assert med_w == 128 and med_h == 128
        assert "MUCH SMALLER than configured input size" in stream.getvalue()

        stream.truncate(0)
        stream.seek(0)

        # 2. Much larger tiles (2048px vs configured 800px)
        recs_large = [ImageRecord("img2", "dummy2.png", 2048, 2048, [], [])]
        med_w, med_h = check_image_sizes(recs_large, configured_size=800)
        assert med_w == 2048 and med_h == 2048
        assert "MUCH LARGER than configured input size" in stream.getvalue()
    finally:
        log.removeHandler(handler)
