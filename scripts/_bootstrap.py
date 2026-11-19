"""Make ``src/`` importable when scripts run from a plain checkout (no ``pip install -e .`` needed)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
