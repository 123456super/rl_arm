#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/c211/jxf/rl
PY=/home/c211/anaconda3/envs/rl/bin/python
CONFIG="$ROOT/configs/experiments/paper_full.yaml"
OUT="$ROOT/outputs/retrain_20261011/full"
mkdir -p "$OUT/logs"
seed="$1"
cpus="$2"
log="$OUT/logs/full_seed${seed}.log"
run_name="full_seed${seed}"
exec setsid env CUDA_VISIBLE_DEVICES=0 taskset -c "$cpus" "$PY" -u "$ROOT/scripts/core/train_thesis_homotopy.py" \
  --config "$CONFIG" \
  --seed "$seed" \
  --run-name "$run_name" \
  --num-envs 8 \
  >"$log" 2>&1
