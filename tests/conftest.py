"""Shared fixtures. Every test uses SYNTHETIC data only (never real data)."""
import sys
from pathlib import Path

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
