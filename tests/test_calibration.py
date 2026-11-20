"""Phase 5: calibrate_unknown.py on SYNTHETIC data with a simulated-unknown split."""
import json
import subprocess
import sys

import pandas as pd


def test_calibration_script_writes_json_and_beats_defaults(tiny_stage1, syn_root, calib_split, tmp_path):
    out = tmp_path / "calibration.json"
    r = subprocess.run([sys.executable, "scripts/calibrate_unknown.py", "--checkpoint", str(tiny_stage1 / "checkpoint.pt"),
                        "--split", str(calib_split), "--set", f"data.root={syn_root}", "--out", str(out), "--no_tta",
                        "--alphas", "1,2", "--taus", "0.05,0.2", "--caps", "5,20", "--cos_ref_pcts", "25,50"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    j = json.loads(out.read_text())
    for k in ("alpha", "tau_unk", "unk_cap", "cos_ref", "val", "heldout_classes", "checkpoint_sha256", "note"):
        assert k in j
    grid = pd.read_csv(tmp_path / "calibration_grid.csv")
    assert j["val"]["oe"] == grid["oe"].max()
    if j["val_with_config_defaults"]:
        assert j["val"]["oe"] >= j["val_with_config_defaults"]["oe"]
    print(f"[SMOKE, synthetic] calibrated: {j['alpha'], j['tau_unk'], j['unk_cap'], round(j['cos_ref'], 3)} val OE={j['val']['oe']:.1f}")
    # and the file is consumed by inference
    from anveshak.postprocess.unknown import load_calibration
    from anveshak.utils.config import load_config
    c = load_calibration(load_config(), out)
    assert (c.alpha, c.tau_unk, c.unk_cap) == (j["alpha"], j["tau_unk"], j["unk_cap"])


def test_calibration_refuses_checkpoint_that_knows_heldout(tiny_stage1, syn_root, tmp_path):
    from anveshak.data.build import load_class_names, load_set
    from anveshak.data.split import make_split, save_split
    from anveshak.utils.config import load_config
    cfg = load_config(overrides=[f"data.root={syn_root}"])
    names = load_class_names(cfg)
    sp = make_split(load_set(cfg, "set_a", names), 0, 0.2, 0, names)
    sp["heldout_classes"] = [names[0]]  # a class the tiny checkpoint DOES know
    save_split(sp, tmp_path / "s.json")
    r = subprocess.run([sys.executable, "scripts/calibrate_unknown.py", "--checkpoint", str(tiny_stage1 / "checkpoint.pt"),
                        "--split", str(tmp_path / "s.json"), "--set", f"data.root={syn_root}", "--out", str(tmp_path / "c.json")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "held-out" in (r.stdout + r.stderr)
