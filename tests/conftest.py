"""Shared fixtures. Every test uses SYNTHETIC data only (never real data)."""
import sys
from pathlib import Path

import torch

if torch.get_num_threads() > 2:
    torch.set_num_threads(2)

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from anveshak.data.synthetic import SyntheticSpec, generate_dataset  # noqa: E402


@pytest.fixture(scope="session")
def syn_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synthetic")
    generate_dataset(root, SyntheticSpec(size=128, n_a=96, n_b=12, n_r2=16, n_test=24, seed=0))
    return root


@pytest.fixture(scope="session")
def calib_split(syn_root, tmp_path_factory):
    """Simulated-unknown split of the synthetic Set A: 2 base classes held out as fake unknowns."""
    from anveshak.data.build import load_class_names, load_set
    from anveshak.data.split import make_split, save_split
    from anveshak.utils.config import load_config

    cfg = load_config(overrides=[f"data.root={syn_root}"])
    names = load_class_names(cfg)
    recs = load_set(cfg, "set_a", names)
    sp = make_split(recs, n_holdout=2, val_frac=0.2, seed=0, class_list=names)
    path = tmp_path_factory.mktemp("split") / "split.json"
    save_split(sp, path)
    return path


@pytest.fixture(scope="session")
def tiny_stage1(syn_root, calib_split, tmp_path_factory):
    """Tiny Stage-1 checkpoint (SYNTHETIC data, held-out split) shared by phase 5-7 tests."""
    from anveshak.training.run import run_stage
    from anveshak.utils.config import REPO_ROOT, load_config

    out = tmp_path_factory.mktemp("tiny_stage1")
    cfg = load_config([str(REPO_ROOT / "configs/stage1_setA.yaml"), str(REPO_ROOT / "configs/smoke.yaml")],
                      [f"data.root={syn_root}", f"out_dir={out}", "train.max_iters=250", "train.max_hours=null",
                       f"data.split_file={calib_split}", "train.log_every=25", "train.ckpt_every=0"])
    run_stage(cfg)
    return out


@pytest.fixture(scope="session")
def tiny_stage2(syn_root, tiny_stage1, tmp_path_factory):
    """Stage 2 on SYNTHETIC Set B from the tiny Stage-1 checkpoint (shared by phase 6-7 tests)."""
    from anveshak.training.run import run_stage
    from anveshak.utils.config import REPO_ROOT, load_config

    out = tmp_path_factory.mktemp("tiny_stage2")
    cfg = load_config([str(REPO_ROOT / "configs/stage2_setB.yaml"), str(REPO_ROOT / "configs/smoke.yaml")],
                      [f"data.root={syn_root}", f"out_dir={out}", f"init.checkpoint={tiny_stage1 / 'checkpoint.pt'}",
                       "train.max_iters=80", "train.max_hours=null", "train.log_every=10", "train.ckpt_every=0", "train.lr=0.0003"])
    return out, run_stage(cfg)
