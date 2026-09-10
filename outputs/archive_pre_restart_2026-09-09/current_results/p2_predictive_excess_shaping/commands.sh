#!/usr/bin/env bash
set -euo pipefail

run_a=outputs/current_results/p2_predictive_excess_shaping/train/predictive_h0p5_excess0p25_seed101_steps30000/actor_step_30000.pt
if [[ -s "$run_a" ]]; then
  echo "skip: $run_a"
else
  conda run --no-capture-output -n rl python scripts/train.py \
    --config configs/experiments/random_crossing_predictive_link_30k_excess0p25.yaml
fi

run_b=outputs/current_results/p2_predictive_excess_shaping/train/predictive_h0p5_excess0p5_seed101_steps30000/actor_step_30000.pt
if [[ -s "$run_b" ]]; then
  echo "skip: $run_b"
else
  conda run --no-capture-output -n rl python scripts/train.py \
    --config configs/experiments/random_crossing_predictive_link_30k_excess0p5.yaml
fi
