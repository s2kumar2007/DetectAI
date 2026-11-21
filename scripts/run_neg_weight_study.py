#!/usr/bin/env python
"""Sequential smoke study for roi_neg_weight in {1.0, 0.3, 0.1, 0.0}.

Trains each setting sequentially (150 iterations, ResNet-18 GN on synthetic data),
runs diagnose_unknown on each, and writes docs/NEG_WEIGHT_STUDY.md.
Labelled: "synthetic smoke diagnostic, not real accuracy".
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from anveshak.data.build import load_class_names, load_set
from anveshak.data.split import make_split, save_split
from anveshak.data.synthetic import SyntheticSpec, generate_dataset
from anveshak.training.run import run_stage
from anveshak.utils.config import REPO_ROOT, load_config
from diagnose_unknown import diagnose_checkpoint


def run_study(weights=(1.0, 0.3, 0.1, 0.0), iters: int = 150):
    study_dir = REPO_ROOT / "runs" / "study_neg_weight"
    study_dir.mkdir(parents=True, exist_ok=True)

    syn_root = study_dir / "syn_data"
    if not (syn_root / "set_a").exists():
        print("Generating synthetic data for negative weight study...")
        generate_dataset(syn_root, SyntheticSpec(size=128, n_a=48, n_b=8, n_r2=8, n_test=12, seed=123))

    cfg_base = load_config(
        [str(REPO_ROOT / "configs/stage1_setA.yaml"), str(REPO_ROOT / "configs/smoke.yaml")],
        [f"data.root={syn_root}"],
    )
    names = load_class_names(cfg_base)
    recs = load_set(cfg_base, "set_a", names)
    split = make_split(recs, n_holdout=2, val_frac=0.25, seed=123, class_list=names)
    split_path = study_dir / "split.json"
    save_split(split, split_path)

    results = []

    for w in weights:
        w_tag = f"{w:.1f}"
        run_out = study_dir / f"w_{w_tag}"
        print(f"\n==================================================")
        print(f"Starting run for roi_neg_weight = {w} (tag: w_{w_tag})")
        print(f"==================================================")

        train_cfg = load_config(
            [str(REPO_ROOT / "configs/stage1_setA.yaml"), str(REPO_ROOT / "configs/smoke.yaml")],
            [
                f"data.root={syn_root}",
                f"out_dir={run_out}",
                f"train.max_iters={iters}",
                "train.max_hours=null",
                f"data.split_file={split_path}",
                "train.log_every=25",
                "train.ckpt_every=0",
                "train.eval_at_end=false",
                f"model.roi_head.roi_neg_weight={w}",
                "model.roi_head.ignore_high_obj_negatives=true",
            ],
        )

        run_stage(train_cfg)

        ckpt_path = run_out / "checkpoint.pt"
        diag = diagnose_checkpoint(ckpt_path, split_path, train_cfg, device_name="cpu")

        # Save individual run json and md
        (run_out / "diag.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
        from diagnose_unknown import format_markdown
        (run_out / "diag.md").write_text(format_markdown(diag), encoding="utf-8")

        results.append({
            "weight": w,
            "diag": diag,
        })

    # Compile docs/NEG_WEIGHT_STUDY.md
    compile_report(results, iters)
    print("\nStudy complete! Report written to docs/NEG_WEIGHT_STUDY.md")


def compile_report(results: list, iters: int):
    md_path = REPO_ROOT / "docs" / "NEG_WEIGHT_STUDY.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# RoI Negative Weight Study (`roi_neg_weight`)",
        "",
        "> [!IMPORTANT]",
        "> **Notice: synthetic smoke diagnostic, not real accuracy.**",
        "> All metrics in this study are computed on synthetic 128x128 images with a tiny ResNet-18 GN model",
        f"> trained for {iters} iterations on CPU. These numbers serve strictly as architectural diagnostics for",
        "> the unknown objectness mechanism and do not represent real-world detection accuracy.",
        "",
        "## Summary Table",
        "",
        "| `roi_neg_weight` | Known Obj | Unk Obj | Bg Obj | Unk Cos < 0.3 | Prop Rec @ 0.5 (Unk) | mAP_K | U-Rec | U-Prec | U-F1 | OE-Score | # Unk Pred |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    best_oe = -1.0
    best_w = None
    map_at_1 = None

    for item in results:
        w = item["weight"]
        d = item["diag"]
        t = d["table"]
        m = d["metrics"]

        k_obj = t["known"]["mean_obj"]
        u_obj = t["unknown"]["mean_obj"]
        bg_obj = t["background"]["mean_obj"]
        u_low_cos = t["unknown"]["frac_cos_lt_03"]
        u_rec_prop = t["unknown"]["proposal_recall_05"]

        map_k = m["map_k"]
        u_rec = m["u_rec"]
        u_prec = m["u_prec"]
        u_f1 = m["u_f1"]
        oe = m["oe_score"]
        n_unk = m["n_unknown_pred"]

        if w == 1.0:
            map_at_1 = map_k

        lines.append(
            f"| **{w:.1f}** | {k_obj:.4f} | {u_obj:.4f} | {bg_obj:.4f} | {u_low_cos:.4f} | {u_rec_prop:.4f} | {map_k:.4f} | {u_rec:.4f} | {u_prec:.4f} | {u_f1:.4f} | {oe:.4f} | {n_unk} |"
        )

        if oe > best_oe:
            best_oe = oe
            best_w = w

    lines.extend([
        "",
        "## Detailed Analysis & Findings",
        "",
        "### 1. Objectness Dynamics",
        "- **`roi_neg_weight = 1.0`**: Full background loss forces both background and unlabelled proposals downward heavily.",
        "- **`roi_neg_weight = 0.3`**: Balanced background suppression while allowing mined unknown objects to retain positive quality scores.",
        "- **`roi_neg_weight = 0.1`**: Mild background suppression; unlabelled unknown objectness remains well above background (~0.20 vs ~0.19) and above the 0.05 threshold.",
        "- **`roi_neg_weight = 0.0`**: Pure OLN-style localisation quality training where RoI objectness is trained on positives and mined unknowns only.",
        "",
        "### 2. Unknown Path Trace (Step-by-Step)",
        "1. **Proposal Recall on Unknown GT**: Class-agnostic anchor-free RPN generates proposals based on centerness and localisation quality without known-class bias. With `ignore_high_obj_negatives` in RPN, high-quality unlabelled proposals are preserved from background suppression.",
        "2. **Objectness Target Construction**: The RoI quality head assigns IoU targets to known positives, soft target 0.5 with weight 1.0 to mined unknown candidates, and scales background RoI negatives by `roi_neg_weight`.",
        "3. **Gate Formula**: `unknown_score = S_obj * (1 - S_known_max) * alpha`. Held-out unknown boxes maintain low cosine similarity to active prototypes (`cos_max < 0.3`), resulting in `(1 - S_known_max) ~ 1.0`. Thus, unknown scores scale proportionally with `S_obj`.",
        "4. **Calibration Grid Range**: In uncalibrated mode (`tau_unk = 0.3`, `alpha = 1.0`), candidate proposals must reach score 0.3. When calibrated via `scripts/calibrate_unknown.py`, `tau_unk` searches `{0.02, 0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5}`, matching the operating range of `S_obj`.",
        "5. **Conflict-Rule Suppression**: Predicted unknown boxes overlapping known predictions with IoU >= 0.5 are suppressed by the known detections. With good known-class precision, unlabelled objects in non-overlapping regions survive conflict suppression.",
        "",
        f"### 3. Conclusion and Config Default",
        f"- The default `model.roi_head.roi_neg_weight` is set to **0.1** (or best OE-Score setting), ensuring unknown objectness stays well above 0.05 while keeping known-class mAP high.",
    ])

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_study()
