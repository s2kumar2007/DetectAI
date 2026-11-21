#!/usr/bin/env bash
# End-to-end pipeline on SYNTHETIC data only (tiny ResNet-18 GN, CPU, ~5-8 minutes). Numbers printed here are smoke results,
# not real-data accuracy. Usage: bash scripts/run_smoke.sh [workdir]
set -euo pipefail
PY=${PY:-python}
W=${1:-runs/smoke}
D=$W/data
SM=configs/smoke.yaml
mkdir -p "$W"
$PY scripts/make_synthetic_data.py --out "$D" --size 128 --n_a 120 --n_b 12 --n_r2 16 --n_test 40
$PY scripts/inspect_data.py --root "$D/set_a" --class_list "$D/class_list.txt"

echo "== simulated-unknown split + calibration run (held-out classes are never learned)"
$PY scripts/make_val_split.py --root "$D/set_a" --class_list "$D/class_list.txt" --n_holdout 2 --val_frac 0.2 --out "$W/split.json"
$PY scripts/train_setA.py --config configs/stage1_setA.yaml $SM --set data.root="$D" train.max_iters=250 train.max_hours=null \
    data.split_file="$W/split.json" --out "$W/calib_stage1"
$PY scripts/calibrate_unknown.py --checkpoint "$W/calib_stage1/checkpoint.pt" --split "$W/split.json" --set data.root="$D" \
    --out "$W/calibration.json"

echo "== Round 1: Stage 1 (Set A) -> Stage 2 (Set B)"
$PY scripts/train_setA.py --config configs/stage1_setA.yaml $SM --set data.root="$D" train.max_iters=300 train.max_hours=null --out "$W/stage1"
$PY scripts/train_setB.py --config configs/stage2_setB.yaml $SM --set data.root="$D" init.checkpoint="$W/stage1/checkpoint.pt" \
    train.max_iters=80 train.max_hours=null train.lr=0.0003 --out "$W/stage2"
$PY scripts/infer_and_validate.py --checkpoint "$W/stage2/checkpoint.pt" --calibration "$W/calibration.json" --set data.root="$D" \
    --out "$W/submission/round1_predictions.csv" --gt "$D/test_gt_round1.csv"
$PY scripts/verify_checkpoint.py --checkpoint "$W/stage2/checkpoint.pt" --csv "$W/submission/round1_predictions.csv" \
    --calibration "$W/calibration.json" --set data.root="$D"

echo "== Round 2 (on-site sequence)"
$PY scripts/train_round2.py --config configs/stage3_round2.yaml $SM --set data.root="$D" init.checkpoint="$W/stage2/checkpoint.pt" \
    train.max_iters=60 train.max_hours=null train.lr=0.0003 replay.per_class=4 --out "$W/round2"
$PY scripts/infer_and_validate.py --checkpoint "$W/round2/checkpoint.pt" --calibration "$W/calibration.json" --set data.root="$D" \
    --out "$W/submission/round2_predictions.csv" --gt "$D/test_gt_round2.csv"
$PY scripts/verify_checkpoint.py --checkpoint "$W/round2/checkpoint.pt" --csv "$W/submission/round2_predictions.csv" \
    --calibration "$W/calibration.json" --set data.root="$D"
$PY scripts/build_submission_bundle.py --run_dir "$W/round2" --csv "$W/submission/round2_predictions.csv" \
    --calibration "$W/calibration.json" --set data.root="$D" --out "$W/submission/round2_bundle.zip"
echo "SMOKE PIPELINE OK (synthetic data only)"
