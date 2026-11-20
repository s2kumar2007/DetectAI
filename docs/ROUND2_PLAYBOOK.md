# Round 2 on-site playbook (A4000, ~60 min door to door)

Goal: continue the SAME lineage from the verified Round 1 checkpoint, learn the 10 new few-shot classes (slots 40..49),
keep Round 1 classes, flag unknowns, and submit a validated CSV. Everything runs offline, from random-init-derived weights only.

Placeholders: `DATA=/path/to/round2_data`, `R1=runs/stage2` (Round 1 final run dir), `CAL=calibration.json` (from Round 1).

## 0. Before leaving home (checklist)
- [ ] `runs/stage2/checkpoint.pt`, its `.sha256`, `class_slots.json` and `calibration.json` are on the laptop AND a USB stick.
- [ ] `python scripts/verify_checkpoint.py ...` printed **PASS** for the submitted Round 1 CSV (this writes `checkpoint.pt.verified.json`, which Round 2 requires).
- [ ] `git log -1` hash noted; `pip install -r requirements.txt` works offline (wheel cache) on the on-site machine.
- [ ] `make test` passes (about 3-4 min on CPU).

## 1. Place data and look at it (5 min)
```bash
export DATA=/path/to/round2_data
python scripts/inspect_data.py --root $DATA/round2 --class_list $DATA/class_list.txt
```
Check: image sizes (set `input.size` if tiles are not ~800 px), annotation format detected, the 10 NEW class codes, and that
the class list is the extended one. The Round 2 training images contain unlabelled base, few-shot and unknown objects - this is expected.

## 2. Confirm the Round 1 checkpoint is the verified one (2 min)
```bash
sha256sum runs/stage2/checkpoint.pt            # must equal the hash recorded at submission time
cat runs/stage2/checkpoint.pt.verified.json     # "passed": true and the same sha256
```
Round 2 refuses to start otherwise (`init.require_verified: true`).

## 3. Train Round 2 (20-40 min, hard budget)
```bash
python scripts/train_round2.py --config configs/stage3_round2.yaml \
    --set data.root=$DATA init.checkpoint=runs/stage2/checkpoint.pt data.class_list=class_list.txt \
    --out runs/round2 --max_hours 0.5
```
What happens: previous-checkpoint assertion (our own file, same architecture) -> new classes get slots 40..49 -> prototypes imprinted from
the exemplars -> short schedule with stem+layer1+layer2 frozen, old prototypes frozen, replay from Set A + Set B, teacher = Round 1
(pseudo-labels for old classes + feature/logit distillation). Time-aware cosine LR: it finishes decayed even if the budget ends first.
Resume after an interruption: re-run the same command (it picks up `runs/round2/last.pt`).
Read `runs/round2/retention_report.json`: `old_map_after / old_map_before` should stay close to 1.0. If it drops a lot
(e.g. < 0.9): lower `train.lr` (8e-5 -> 4e-5), raise `replay.ratio` (0.5 -> 0.6) and re-run with `--no_resume`.

## 4. Inference + validation (5 min)
```bash
python scripts/infer_and_validate.py --checkpoint runs/round2/checkpoint.pt --calibration calibration.json \
    --set data.root=$DATA postprocess.validator_path=$DATA/validate_predictions.py --out submission/round2_predictions.csv
```
The organisers' validator (if the path exists) AND our checker both run; any violation aborts. Fix, never ignore.

## 5. Verify reproducibility, bundle, submit (5 min)
```bash
python scripts/verify_checkpoint.py --checkpoint runs/round2/checkpoint.pt --csv submission/round2_predictions.csv \
    --calibration calibration.json --set data.root=$DATA
python scripts/build_submission_bundle.py --run_dir runs/round2 --csv submission/round2_predictions.csv \
    --calibration calibration.json --set data.root=$DATA --out submission/round2_bundle.zip
```
Submit `submission/round2_predictions.csv` (and the bundle if requested). Keep `runs/round2/checkpoint.pt`.

## Rules to remember
- Never train/tune/pseudo-label on the test images. Calibration is NOT re-run on site on test data; reuse `calibration.json`.
- No downloads, no pretrained weights. If something tries to reach the network, stop and report it.
- Do not change `model.*` settings (architecture must match the checkpoint; the assertion will stop you).

## Troubleshooting
| Symptom | Fix |
|---|---|
| `PretrainedWeightError ... verified.json` | run step 2 / `verify_checkpoint.py` for the Round 1 checkpoint |
| `architecture mismatch` | pass the same config overlays used for Round 1 (e.g. `configs/smoke.yaml` only for smoke runs) |
| CUDA OOM | `train.batch_size=2 train.accum_steps=2` (same effective batch) or `input.size=640` |
| `no NEW classes found` | wrong `data.train_dir` / class list: new classes must be labelled in `$DATA/round2` |
| non-finite loss warnings repeating | lower `train.lr` by 2x, re-run with `--no_resume` |
| validator complains about `image_id` | switch `data.image_id_mode` (`stem` or `filename`) and re-run step 4 |
