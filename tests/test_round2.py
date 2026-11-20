"""Phase 7: verify_checkpoint, Round 2 (Stage 3) from a VERIFIED checkpoint, submission bundle - SYNTHETIC data."""
import json
import subprocess
import sys
import zipfile

import pandas as pd
import pytest

from anveshak.models.assertions import PretrainedWeightError
from anveshak.training.run import run_stage
from anveshak.utils.config import REPO_ROOT, load_config

S3 = [str(REPO_ROOT / "configs/stage3_round2.yaml"), str(REPO_ROOT / "configs/smoke.yaml")]


def _run(args):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True)


@pytest.fixture(scope="module")
def submission(tiny_stage2, syn_root, tmp_path_factory):
    out, _ = tiny_stage2
    d = tmp_path_factory.mktemp("sub")
    r = _run(["scripts/infer_and_validate.py", "--checkpoint", str(out / "checkpoint.pt"), "--set", f"data.root={syn_root}",
              "calibration.tau_unk=0.05", "--out", str(d / "predictions.csv")])
    assert r.returncode == 0, r.stdout + r.stderr
    return d / "predictions.csv"


def test_verify_checkpoint_pass_fail_and_gate(tiny_stage2, syn_root, submission, tmp_path):
    ck = tiny_stage2[0] / "checkpoint.pt"
    vf = tiny_stage2[0] / "checkpoint.pt.verified.json"
    assert not vf.exists()
    base = ["scripts/verify_checkpoint.py", "--checkpoint", str(ck), "--set", f"data.root={syn_root}"]
    # tampered CSV (a box shifted by 5 px) must FAIL and must not create the verified file
    df = pd.read_csv(submission)
    df.loc[0, ["xmin", "xmax"]] += 5
    bad = tmp_path / "bad.csv"
    df.to_csv(bad, index=False)
    (tmp_path / "bad.csv.meta.json").write_text((submission.parent / "predictions.csv.meta.json").read_text())
    r = _run(base + ["--csv", str(bad)])
    assert r.returncode == 1 and "FAIL" in r.stdout and not vf.exists()
    # dropped row -> count mismatch
    df.iloc[1:].to_csv(bad, index=False)
    assert "row count differs" in _run(base + ["--csv", str(bad)]).stdout
    # genuine CSV passes
    r = _run(base + ["--csv", str(submission)])
    assert r.returncode == 0 and "PASS" in r.stdout and vf.exists()
    assert json.loads(vf.read_text())["passed"] is True


def test_round2_requires_verified_checkpoint_then_runs(tiny_stage2, syn_root, tmp_path):
    ck = tiny_stage2[0] / "checkpoint.pt"
    common = [f"data.root={syn_root}", f"init.checkpoint={ck}", "train.max_iters=40", "train.max_hours=null", "train.log_every=10",
              "train.ckpt_every=0", "train.lr=0.0003", "replay.per_class=4"]
    vf = tiny_stage2[0] / "checkpoint.pt.verified.json"
    saved = vf.read_text() if vf.exists() else None
    if vf.exists():
        vf.unlink()
    with pytest.raises(PretrainedWeightError):  # require_verified is true in stage3_round2.yaml
        run_stage(load_config(S3, common + [f"out_dir={tmp_path / 'x'}"]))
    if saved:
        vf.write_text(saved)
    else:
        pytest.skip("verified file produced by the previous test is required")
    out = tmp_path / "r2"
    res = run_stage(load_config(S3, common + [f"out_dir={out}"]))
    slots = json.loads((out / "class_slots.json").read_text())["class_to_slot"]
    assert slots["SYN09"] == 40 and slots["SYN10"] == 41 and slots["SYN07"] == 4          # Round 2 slots 40..49
    rep = json.loads((out / "retention_report.json").read_text())
    print(f"[SMOKE, synthetic] Round 2 old-class mAP before={rep['old_map_before']:.3f} after={rep['old_map_after']:.3f}")
    assert rep["old_map_after"] >= 0.5 * rep["old_map_before"]
    assert set(rep["old_classes"]) >= {"SYN01", "SYN07", "SYN08"} and res["summary"]["completed"]
    # checkpoint edited after verification -> hash mismatch -> refused
    with open(ck, "ab") as fh:
        fh.write(b"\0")
    try:
        with pytest.raises(PretrainedWeightError):
            run_stage(load_config(S3, common + [f"out_dir={tmp_path / 'y'}"]))
    finally:
        data = ck.read_bytes()
        ck.write_bytes(data[:-1])


def test_bundle_builder(tiny_stage2, syn_root, submission, tmp_path):
    run_dir = tiny_stage2[0]
    assert (run_dir / "checkpoint.pt.verified.json").exists() or _run(
        ["scripts/verify_checkpoint.py", "--checkpoint", str(run_dir / "checkpoint.pt"), "--csv", str(submission), "--set", f"data.root={syn_root}"]).returncode == 0
    out = tmp_path / "b.zip"
    r = _run(["scripts/build_submission_bundle.py", "--run_dir", str(run_dir), "--csv", str(submission), "--set", f"data.root={syn_root}", "--out", str(out)])
    assert r.returncode == 0, r.stdout + r.stderr
    z = zipfile.ZipFile(out)
    names = set(z.namelist())
    assert {"predictions.csv", "class_slots.json", "MANIFEST.json", "REPRODUCE.md", "retention_report.json"} <= names
    assert "checkpoint.pt" not in names and not any(n.endswith((".png", ".jpg")) for n in names)
    man = json.loads(z.read("MANIFEST.json"))
    assert len(man["checkpoint_sha256"]) == 64 and man["n_test_images"] == 24
    # unverified checkpoints are refused
    (run_dir / "checkpoint.pt.verified.json").rename(run_dir / "v.bak")
    try:
        assert _run(["scripts/build_submission_bundle.py", "--run_dir", str(run_dir), "--csv", str(submission), "--set", f"data.root={syn_root}", "--out", str(tmp_path / "c.zip")]).returncode != 0
    finally:
        (run_dir / "v.bak").rename(run_dir / "checkpoint.pt.verified.json")
