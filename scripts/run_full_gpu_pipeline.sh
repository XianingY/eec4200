#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/src}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs_final}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DRY_RUN="${DRY_RUN:-0}"

HMDB_EPOCHS="${HMDB_EPOCHS:-30}"
HMDB_BATCH_SIZE="${HMDB_BATCH_SIZE:-8}"
HMDB_LR="${HMDB_LR:-1e-3}"
HMDB_WEIGHT_DECAY="${HMDB_WEIGHT_DECAY:-1e-4}"
HMDB_PATIENCE="${HMDB_PATIENCE:-5}"

ARID_EPOCHS="${ARID_EPOCHS:-20}"
ARID_BATCH_SIZE="${ARID_BATCH_SIZE:-8}"
ARID_LR="${ARID_LR:-3e-4}"
ARID_WEIGHT_DECAY="${ARID_WEIGHT_DECAY:-1e-4}"
ARID_PATIENCE="${ARID_PATIENCE:-5}"

mkdir -p "$OUTPUT_DIR/logs"

run_step() {
  local step_name="$1"
  shift
  local log_path="$OUTPUT_DIR/logs/${step_name}.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running ${step_name}"
  echo "Command: $PYTHON_BIN -m eec4200 $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    {
      printf 'DRY RUN: %q -m eec4200' "$PYTHON_BIN"
      for token in "$@"; do
        printf ' %q' "$token"
      done
      printf '\n'
    } | tee "$log_path"
    return
  fi
  "$PYTHON_BIN" -m eec4200 "$@" 2>&1 | tee "$log_path"
}

run_step summarize-data \
  summarize-data \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR"

run_step train-hmdb \
  train-hmdb \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --epochs "$HMDB_EPOCHS" \
  --batch-size "$HMDB_BATCH_SIZE" \
  --lr "$HMDB_LR" \
  --weight-decay "$HMDB_WEIGHT_DECAY" \
  --patience "$HMDB_PATIENCE" \
  --num-workers "$NUM_WORKERS" \
  --seed "$SEED"

run_step eval-cross-dataset \
  eval-cross-dataset \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint "$OUTPUT_DIR/checkpoints/hmdb_best.pt" \
  --device "$DEVICE" \
  --clip-length 16 \
  --image-size 112 \
  --num-test-clips 3 \
  --seed "$SEED"

run_step train-arid \
  train-arid \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint "$OUTPUT_DIR/checkpoints/hmdb_best.pt" \
  --device "$DEVICE" \
  --epochs "$ARID_EPOCHS" \
  --batch-size "$ARID_BATCH_SIZE" \
  --lr "$ARID_LR" \
  --weight-decay "$ARID_WEIGHT_DECAY" \
  --patience "$ARID_PATIENCE" \
  --num-workers "$NUM_WORKERS" \
  --seed "$SEED"

run_step build-report \
  build-report \
  --output-dir "$OUTPUT_DIR"

echo
echo "Final report: $OUTPUT_DIR/report/EEC4200_report.pdf"
echo "Logs: $OUTPUT_DIR/logs/"
