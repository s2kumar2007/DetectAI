"""Phase 4: Stage 1 smoke training on SYNTHETIC data (loss must decrease), resume, time budget, no-pretrained check."""
import json
import shutil
from pathlib import Path

import pytest
import torch

from anveshak.models.assertions import PretrainedWeightError
from anveshak.training.checkpoint import load_payload, model_from_checkpoint
from anveshak.training.ema import ModelEMA
from anveshak.training.run import run_stage
from anveshak.training.schedule import lr_factor
from anveshak.utils.config import REPO_ROOT, load_config

SMOKE = [str(REPO_ROOT / "configs" / "stage1_setA.yaml"), str(REPO_ROOT / "configs" / "smoke.yaml")]


def _cfg(syn_root, out, iters, extra=()):
    return load_config(SMOKE, [f"data.root={syn_root}", f"out_dir={out}", f"train.max_iters={iters}", "stage=stage1",
                               "train.max_hours=null", "train.eval_at_end=false", *extra])


@pytest.fixture(scope="module")
def stage1_run(syn_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("stage1")
    cfg = _cfg(syn_root, out, 80, ["train.log_every=5", "train.ckpt_every=40", "train.eval_at_end=true"])
    res = run_stage(cfg)
    return out, res


def _losses(out):
    return [json.loads(l)["loss"] for l in (out / "metrics.jsonl").read_text().splitlines()]


def test_stage1_smoke_loss_decreases(stage1_run):
    out, res = stage1_run
    ls = _losses(out)
    first, last = sum(ls[:4]) / 4, sum(ls[-4:]) / 4
    print(f"[SMOKE, synthetic data] mean loss first 20 its = {first:.3f}, last 20 its = {last:.3f}")
    assert last < first
    assert res["summary"]["completed"] and res["summary"]["iterations"] == 80


def test_stage1_outputs_and_checkpoint_contents(stage1_run):
    out, _ = stage1_run
    for f in ("checkpoint.pt", "checkpoint.pt.sha256", "class_slots.json", "retention_report.json", "train_summary.json", "last.pt"):
        assert (out / f).exists(), f
    pl = load_payload(out / "checkpoint.pt")
    for k in ("model", "ema", "config", "slot_map", "seed", "git_commit", "init_hash", "arch", "stage", "created_by"):
        assert k in pl
    assert pl["stage"] == "stage1" and pl["seed"] == 1337 and len(pl["init_hash"]) == 64
    slots = json.loads((out / "class_slots.json").read_text())
    assert sorted(slots["class_to_slot"].values()) == list(range(6))
    rep = json.loads((out / "retention_report.json").read_text())
    assert rep["old_classes"] == [] and "val_map_known" in rep
    model, meta, sm = model_from_checkpoint(out / "checkpoint.pt")
    assert model.active_slots == sm.active_slots == list(range(6))


def test_resume_continues_and_checks_lineage(syn_root, tmp_path):
    cfg = _cfg(syn_root, tmp_path, 12, ["train.ckpt_every=6", "train.log_every=3"])
    run_stage(cfg)
    assert (tmp_path / "last.pt").exists()
    first = json.loads((tmp_path / "train_summary.json").read_text())["init_hash"]
    cfg2 = _cfg(syn_root, tmp_path, 18, ["train.ckpt_every=6", "train.log_every=3"])
    res = run_stage(cfg2)
    assert res["summary"]["iterations"] == 18 and res["summary"]["init_hash"] == first
    its = [json.loads(l)["iter"] for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert its == sorted(its) and its[-1] == 18 and its.count(12) == 1  # continued, did not restart at 0
    # a resume file from a different seed's init must be rejected (hard rule 1)
    cfg3 = _cfg(syn_root, tmp_path, 24, ["seed=5"])
    with pytest.raises(RuntimeError):
        run_stage(cfg3)


def test_time_budget_stops_cleanly(syn_root, tmp_path):
    cfg = _cfg(syn_root, tmp_path, 100000, ["train.warmup_iters=2"])
    res = run_stage(cfg, max_hours=0.0008)  # ~3 seconds
    s = res["summary"]
    assert s["stopped_by"] == "time_budget" and not s["completed"] and 0 < s["iterations"] < 100000
    assert (tmp_path / "checkpoint.pt").exists()


def test_stage1_refuses_checkpoint_in_config(syn_root, tmp_path):
    cfg = _cfg(syn_root, tmp_path, 2, ["init.checkpoint=foo.pt"])
    with pytest.raises(PretrainedWeightError):
        run_stage(cfg)


def test_stage1_prints_initial_weight_hash(syn_root, tmp_path, capsys):
    run_stage(_cfg(syn_root, tmp_path, 1))
    assert "[no-pretrained check] Stage 1 initial weights SHA-256" in capsys.readouterr().out


def test_bf16_autocast_path_on_cpu(syn_root, tmp_path):
    res = run_stage(_cfg(syn_root, tmp_path, 2, ["train.amp=true"]))
    assert res["summary"]["iterations"] == 2


def test_ema_and_schedule():
    m = torch.nn.Linear(2, 2)
    ema = ModelEMA(m, decay=0.5, warmup=False)
    with torch.no_grad():
        m.weight.fill_(1.0)
        ema.module.weight.fill_(0.0)
    ema.update(m)
    assert torch.allclose(ema.module.weight, torch.full((2, 2), 0.5))
    assert lr_factor(0, 10, 0.0, 0.01) == pytest.approx(0.1)
    assert lr_factor(100, 10, 0.0, 0.01) == pytest.approx(1.0)
    assert lr_factor(100, 10, 1.0, 0.01) == pytest.approx(0.01)
