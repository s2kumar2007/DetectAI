#!/usr/bin/env bash
# Real-data Round 1 training & submission pipeline on RTX A4000 (16 GB).
# Usage:
#   DATA=/path/to/anveshak_data OUT=runs/real MAX_HOURS_S1=10 MAX_HOURS_S2=2 bash scripts/run_real.sh
set -euo pipefail

DATA=${DATA:?Set DATA=/path/to/anveshak_data}
OUT=${OUT:-runs/real}
MAX_HOURS_S1=${MAX_HOURS_S1:-10}
MAX_HOURS_S2=${MAX_HOURS_S2:-2}
PY=${PY:-python}

mkdir -p "$OUT" logs splits "$OUT/submission"

echo "=== 1. Symlink check & inspect_data ==="
if [ ! -e "$DATA/set_a" ] && [ -d "$DATA/dataset_A" ]; then
    ln -s dataset_A "$DATA/set_a"
fi
if [ ! -e "$DATA/set_b" ] && [ -d "$DATA/dataset_B" ]; then
    ln -s dataset_B "$DATA/set_b"
fi

$PY scripts/preflight.py --config configs/stage1_setA.yaml --set data.root="$DATA" data.class_list=class_list.csv out_dir="$OUT/stage1"
$PY scripts/inspect_data.py --root "$DATA/set_a" --class_list "$DATA/class_list.csv" --configured_size 1024
$PY scripts/inspect_data.py --root "$DATA/set_b" --class_list "$DATA/class_list.csv" --configured_size 1024

echo "=== 2. Simulated-unknown validation split ==="
$PY scripts/make_val_split.py --root "$DATA/set_a" --class_list "$DATA/class_list.csv" \
    --holdout_classes 11 14 17 21 23 24 --val_frac 0.15 --seed 1337 --out splits/set_a_split.json

echo "=== 3. 20-minute diagnostic run (--max_hours 0.33) ==="
nohup $PY scripts/train_setA.py --config configs/stage1_setA.yaml \
    --set data.root="$DATA" data.class_list=class_list.csv data.split_file=splits/set_a_split.json \
    --out "$OUT/diag_stage1" --max_hours 0.33 > logs/diag_stage1.log 2>&1 &
PID=$!
echo "diag_stage1 PID=$PID"
sleep 3; ps -p $PID || true; tail -n 5 logs/diag_stage1.log || true
wait $PID

$PY scripts/diagnose_unknown.py --checkpoint "$OUT/diag_stage1/checkpoint.pt" \
    --split splits/set_a_split.json --set data.root="$DATA" data.class_list=class_list.csv

echo "=== 4. Stage 1 full training (Set A, max_hours=$MAX_HOURS_S1) ==="
nohup $PY scripts/train_setA.py --config configs/stage1_setA.yaml \
    --set data.root="$DATA" data.class_list=class_list.csv data.split_file=splits/set_a_split.json \
    --out "$OUT/stage1" --max_hours "$MAX_HOURS_S1" > logs/stage1.log 2>&1 &
PID=$!
echo "stage1 PID=$PID"
sleep 3; ps -p $PID || true; tail -n 5 logs/stage1.log || true
wait $PID

echo "=== 5. Calibrate unknown gate after Stage 1 ==="
$PY scripts/calibrate_unknown.py --checkpoint "$OUT/stage1/checkpoint.pt" \
    --split splits/set_a_split.json --set data.root="$DATA" data.class_list=class_list.csv \
    --out "$OUT/calibration_stage1.json"

echo "=== 6. Stage 2 continual learning (Set B, max_hours=$MAX_HOURS_S2) ==="
nohup $PY scripts/train_setB.py --config configs/stage2_setB.yaml \
    --set data.root="$DATA" data.class_list=class_list.csv init.checkpoint="$OUT/stage1/checkpoint.pt" \
    --out "$OUT/stage2" --max_hours "$MAX_HOURS_S2" > logs/stage2.log 2>&1 &
PID=$!
echo "stage2 PID=$PID"
sleep 3; ps -p $PID || true; tail -n 5 logs/stage2.log || true
wait $PID

echo "=== 7. Re-calibrate unknown gate after Stage 2 ==="
$PY scripts/calibrate_unknown.py --checkpoint "$OUT/stage2/checkpoint.pt" \
    --split splits/set_a_split.json --set data.root="$DATA" data.class_list=class_list.csv \
    --out "$OUT/calibration.json"

echo "=== 8. Inference, validation, verification & bundle ==="
$PY scripts/infer_and_validate.py --checkpoint "$OUT/stage2/checkpoint.pt" \
    --calibration "$OUT/calibration.json" \
    --set data.root="$DATA" data.class_list=class_list.csv postprocess.validator_path="$DATA/validate_predictions.py" \
    --out "$OUT/submission/round1_predictions.csv"

$PY scripts/verify_checkpoint.py --checkpoint "$OUT/stage2/checkpoint.pt" \
    --csv "$OUT/submission/round1_predictions.csv" \
    --calibration "$OUT/calibration.json" --set data.root="$DATA" data.class_list=class_list.csv

$PY scripts/build_submission_bundle.py --run_dir "$OUT/stage2" \
    --csv "$OUT/submission/round1_predictions.csv" \
    --calibration "$OUT/calibration.json" --set data.root="$DATA" data.class_list=class_list.csv \
    --out "$OUT/submission/round1_bundle.zip"

echo "REAL PIPELINE COMPLETE: $OUT/submission/round1_bundle.zip"
