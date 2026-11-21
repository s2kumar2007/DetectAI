"""Phase 8: PPT generator and README command coverage."""
import re
import subprocess
import sys
from pathlib import Path

from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]


def test_make_ppt_six_slides(tmp_path):
    out = tmp_path / "d.pptx"
    r = subprocess.run([sys.executable, "scripts/make_ppt.py", "--out", str(out), "--team", "T"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    prs = Presentation(str(out))
    assert len(prs.slides) == 6
    text = " ".join(sh.text_frame.text for s in prs.slides for sh in s.shapes if sh.has_text_frame)
    for key in ("Problem understanding", "Methodology", "First implementation", "Round 2 plan", "Team and roles", "SYNTHETIC"):
        assert key in text


def test_readme_mentions_every_required_command():
    readme = (ROOT / "README.md").read_text()
    for cmd in ("inspect_data.py", "make_val_split.py", "train_setA.py", "train_setB.py", "calibrate_unknown.py", "infer_and_validate.py",
                "verify_checkpoint.py", "train_round2.py", "build_submission_bundle.py", "--max_hours", "validate_predictions.py"):
        assert cmd in readme, cmd
    for script in re.findall(r"scripts/(\w+\.py)", readme):
        assert (ROOT / "scripts" / script).exists(), script
