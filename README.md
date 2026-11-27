# anveshak-owod-rs

Open-World Object Detector for remote sensing (Techfest IIT Bombay, "Anveshak"). One model lineage:
**Round 1** = train Set A from random init, then continue the same model on Set B (40 known classes);
**Round 2** = continue from the verified Round 1 checkpoint and learn 10 new few-shot classes.
Unknown objects are emitted as `class_name = unknown`; old classes are protected from forgetting; every step is reproducible.

> **Honesty note.** The real data was not available when this code was written. Nothing here reports real-data accuracy.
> All numbers in this repo (tests, `docs/`, commit messages) are **smoke results on a SYNTHETIC dataset** with a tiny model.

## Hard rules (enforced in code)
1. Random init only. Stage 1 asserts no checkpoint is configured and the weights are bit-identical to a fresh seeded init, and prints their SHA-256. Stage 2/3 load **only** our own previous-stage checkpoint (tag, stage and architecture checked). `torch.hub` downloads are disabled.
2. No open-vocabulary / CLIP / SAM / DINO / LLM / network (socket connections to non-loopback hosts are blocked at train and inference time).
3. Same architecture for Set A -> Set B -> Round 2; 64 class slots are pre-allocated and masked (inactive = -inf).
4. Test images are inference-only; thresholds are tuned only on a simulated-unknown split of training data.
5. Pure PyTorch + torchvision (+ numpy, pandas, pillow, pyyaml, tqdm, opencv-python-headless, python-pptx, pytest).
6. Deterministic inference (seeds, deterministic kernels, `cudnn.benchmark=False`, fixed 4-view TTA).

## Layout
```
configs/   base.yaml (all defaults) + stage1_setA / stage2_setB / stage3_round2 / smoke
src/anveshak/{models,data,training,eval,postprocess,utils}
scripts/   make_synthetic_data inspect_data make_val_split train_setA train_setB train_round2 calibrate_unknown
           infer_and_validate verify_checkpoint build_submission_bundle make_ppt run_smoke.sh
tests/     pytest, CPU, synthetic data only (~3-4 min)      docs/  ARCHITECTURE METHODOLOGY ROUND2_PLAYBOOK GIT_HISTORY
```

## 1. Environment (RTX A4000, 16 GB)
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # or your CUDA build
pip install -r requirements.txt && pip install -e .
python -c "import torch; print(torch.cuda.is_available())"
make test            # 100+ tests on synthetic data, CPU
make smoke           # whole pipeline on synthetic data (see "Smoke run" below)
```

## 2. Place the real data
Expected layout under one root (names configurable in `configs/base.yaml: data.*`):
```
/path/to/anveshak_data/
  dataset_A/  -> symlink as set_a/   (images/<tile_id>.tif, labels/<tile_id>.json, tiles.csv)
  dataset_B/  -> symlink as set_b/   (images/<tile_id>.tif, labels/<tile_id>.json, tiles.csv)
  test/       (images/<tile_id>.tif, tiles.csv; unlabelled Round 1 images)
  class_list.csv   (header: class_id,label; 41 rows: ids 0-39 + unknown)
  validate_predictions.py   (organisers' validator, optional)
```
Create symlinks if needed and pass `data.class_list=class_list.csv`:
```bash
ln -s dataset_A /path/to/anveshak_data/set_a
ln -s dataset_B /path/to/anveshak_data/set_b
```
Round 2 data goes to `round2/` (training) and `test/` (new unlabelled images). Pass the root once per command:
`--set data.root=/path/to/anveshak_data data.class_list=class_list.csv`.
Or run the full real-data sequence via `DATA=/path/to/anveshak_data OUT=runs/real bash scripts/run_real.sh`.
```bash
export DATA=/path/to/anveshak_data
```
If the CSV `image_id` must be the file name with extension, add `data.image_id_mode=filename` to every command.

## Before training on the GPU (Sanity & Readiness Checklist)
Before launching an 8-hour training run on the RTX A4000 (16 GB), run this sequence to guarantee hardware, memory, data, and unknown objectness health:

```bash
# 1. Environment & data preflight check (all checks must PASS)
python scripts/preflight.py --config configs/stage1_setA.yaml --set data.root=$DATA

# 2. Speed and VRAM benchmark (evaluates 512, 800, 1024 px; verifies 16 GB VRAM fit)
python scripts/benchmark_speed.py

# 3. Inspect dataset & auto-detect tile dimensions (warns if tiles differ from input.size)
python scripts/inspect_data.py --root $DATA/set_a --class_list $DATA/class_list.txt
python scripts/inspect_data.py --root $DATA/set_b --class_list $DATA/class_list.txt
python scripts/inspect_data.py --root $DATA/test

# 4. Create the simulated-unknown validation split (holds out 6 fake unknown classes)
python scripts/make_val_split.py --root $DATA/set_a --class_list $DATA/class_list.txt \
    --n_holdout 6 --val_frac 0.15 --seed 1337 --out splits/set_a_split.json

# 5. Short 20-minute diagnostic run to verify unknown objectness before long runs
python scripts/train_setA.py --config configs/stage1_setA.yaml \
    --set data.root=$DATA data.split_file=splits/set_a_split.json \
    --out runs/pre_train_diagnostic --max_hours 0.33

# 6. Diagnose unknown objectness on the short checkpoint
python scripts/diagnose_unknown.py --checkpoint runs/pre_train_diagnostic/checkpoint.pt \
    --split splits/set_a_split.json --config configs/stage1_setA.yaml --set data.root=$DATA
# Verify that unknown objectness is > 0.05 and clearly above background before starting the 8-hour run!
```

## 3. Simulated-unknown split (for calibration only)
```bash
python scripts/make_val_split.py --root $DATA/set_a --class_list $DATA/class_list.txt \
    --n_holdout 6 --val_frac 0.15 --seed 1337 --out splits/set_a_split.json
```

## 4. Round 1 training
```bash
# 4a. short CALIBRATION run: held-out classes are never learned (they stay unlabelled in the images)
python scripts/train_setA.py --config configs/stage1_setA.yaml --set data.root=$DATA data.split_file=splits/set_a_split.json \
    --out runs/calib_stage1 --max_hours 2
python scripts/calibrate_unknown.py --checkpoint runs/calib_stage1/checkpoint.pt --split splits/set_a_split.json \
    --set data.root=$DATA --out calibration.json

# 4b. FINAL lineage, Stage 1 (random init; ResNet-50 GN, 800 px, batch 4 x accumulation 4, BF16, EMA)
python scripts/train_setA.py --config configs/stage1_setA.yaml --set data.root=$DATA --out runs/stage1 --max_hours 8
#     interrupted? re-run the same command: it resumes from runs/stage1/last.pt

# 4c. Stage 2 on Set B (imprinting, pseudo-labels, replay, distillation)
python scripts/train_setB.py --config configs/stage2_setB.yaml --set data.root=$DATA init.checkpoint=runs/stage1/checkpoint.pt \
    --out runs/stage2 --max_hours 2
cat runs/stage2/retention_report.json     # old-class mAP before vs after
```
Time budgets are suggestions; the LR schedule is time-aware, so training finishes decayed whatever `--max_hours` you give.
`calibration.json` comes from a different (short) model than the final one - see `ASSUMPTIONS.md` A36.

## 5. Inference, CSV validation, checkpoint verification
```bash
python scripts/infer_and_validate.py --checkpoint runs/stage2/checkpoint.pt --calibration calibration.json \
    --set data.root=$DATA postprocess.validator_path=$DATA/validate_predictions.py --out submission/round1_predictions.csv
python scripts/verify_checkpoint.py --checkpoint runs/stage2/checkpoint.pt --csv submission/round1_predictions.csv \
    --calibration calibration.json --set data.root=$DATA          # PASS writes checkpoint.pt.verified.json
python scripts/build_submission_bundle.py --run_dir runs/stage2 --csv submission/round1_predictions.csv \
    --calibration calibration.json --set data.root=$DATA --out submission/round1_bundle.zip
```
`--no_tta` disables the 4-view TTA. Header is exactly `image_id,class_name,score,xmin,ymin,xmax,ymax`.

## 6. Round 2 on-site sequence (details and troubleshooting: `docs/ROUND2_PLAYBOOK.md`)
```bash
export DATA=/path/to/round2_data
python scripts/inspect_data.py --root $DATA/round2 --class_list $DATA/class_list.txt
sha256sum runs/stage2/checkpoint.pt; cat runs/stage2/checkpoint.pt.verified.json     # must be the verified Round 1 checkpoint
python scripts/train_round2.py --config configs/stage3_round2.yaml --set data.root=$DATA init.checkpoint=runs/stage2/checkpoint.pt \
    --out runs/round2 --max_hours 0.5
python scripts/infer_and_validate.py --checkpoint runs/round2/checkpoint.pt --calibration calibration.json \
    --set data.root=$DATA postprocess.validator_path=$DATA/validate_predictions.py --out submission/round2_predictions.csv
python scripts/verify_checkpoint.py --checkpoint runs/round2/checkpoint.pt --csv submission/round2_predictions.csv \
    --calibration calibration.json --set data.root=$DATA
python scripts/build_submission_bundle.py --run_dir runs/round2 --csv submission/round2_predictions.csv \
    --calibration calibration.json --set data.root=$DATA --out submission/round2_bundle.zip
```

## Smoke run (synthetic data, CPU)
`make smoke` (= `bash scripts/run_smoke.sh`) generates synthetic data and runs the entire sequence above with a tiny model:
split -> calibration run -> calibrate -> Stage 1 -> Stage 2 -> inference -> verify -> Round 2 -> inference -> verify -> bundle.
It prints local-scorer numbers for the synthetic test tiles; they only show that the plumbing works. In the recorded smoke run
(tiny model, ~300 CPU iterations) Round 1 scored mAP_K = 0.60 and Round 2 mAP_K = 0.64 on 8 / 10 classes, but **U-F1 = 0**:
the undertrained tiny model's unknown boxes did not match any unknown object on the test tiles, even though calibration on the
validation split found unknowns. Unknown quality on real data is untested - budget time to inspect it (ASSUMPTIONS A32/A36).

## Local metric
`src/anveshak/eval/local_scorer.py` reproduces the brief: IoU 0.5, VOC all-point AP per known class (classes without GT skipped and
listed), U-Rec / U-Prec / U-F1, the known-vs-unknown conflict rule applied first, `OE = 100*2*mAP_K*U-F1/(mAP_K+U-F1)`.
Interpretation choices are in `ASSUMPTIONS.md`; hand-computed unit tests are in `tests/test_scorer.py`.

## Git
Phase commits `phase-0 ... phase-8` and tag `v1.0`; history is saved in `docs/GIT_HISTORY.txt`.
`git remote add origin <url> && git push --follow-tags`.
