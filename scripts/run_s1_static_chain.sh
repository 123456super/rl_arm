#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
read -r -a PYTHON_CMD <<< "${PYTHON:-python}"

if [[ -z "${STAGE}" ]]; then
  echo "usage: bash scripts/run_s1_static_chain.sh {eval-frozen|finetune|select-finetune|eval-finetune|summarize-frozen|summarize-finetune}" >&2
  exit 2
fi

FROZEN_ROOT="outputs/reaching_incremental/s1_static_obstacle"
FINETUNE_ROOT="outputs/reaching_incremental/s1_static_obstacle_finetune"

S0_ACTOR_4301="outputs/reaching_recovery_v2/train/seed_4301/link_fixed_no_obstacle_speed100_seed4301_steps300000/actor_step_240000.pt"
S0_ACTOR_4302="outputs/reaching_recovery_v2/train/seed_4302/link_fixed_no_obstacle_speed100_seed4302_steps300000/actor_step_280000.pt"
S0_ACTOR_4303="outputs/reaching_recovery_v2/train/seed_4303/link_fixed_no_obstacle_speed100_seed4303_steps300000/actor_step_220000.pt"
S0_STATE_4301="outputs/reaching_recovery_v2/train/seed_4301/link_fixed_no_obstacle_speed100_seed4301_steps300000/agent_state_step_240000.pt"
S0_STATE_4302="outputs/reaching_recovery_v2/train/seed_4302/link_fixed_no_obstacle_speed100_seed4302_steps300000/agent_state_step_280000.pt"
S0_STATE_4303="outputs/reaching_recovery_v2/train/seed_4303/link_fixed_no_obstacle_speed100_seed4303_steps300000/agent_state_step_220000.pt"

TRAIN_RUN_4301="${FINETUNE_ROOT}/train/seed_4301/link_fixed_static_obstacle_finetune_seed4301_steps440000"
TRAIN_RUN_4302="${FINETUNE_ROOT}/train/seed_4302/link_fixed_static_obstacle_finetune_seed4302_steps480000"
TRAIN_RUN_4303="${FINETUNE_ROOT}/train/seed_4303/link_fixed_static_obstacle_finetune_seed4303_steps420000"

summarize_frozen() {
  "${PYTHON_CMD[@]}" scripts/summarize_s1_static_chain.py \
    --protocol s1_static_obstacle_frozen_actor_final \
    --eval "4301=${FROZEN_ROOT}/eval/seed_4301_final.csv" \
    --eval "4302=${FROZEN_ROOT}/eval/seed_4302_final.csv" \
    --eval "4303=${FROZEN_ROOT}/eval/seed_4303_final.csv" \
    --output-json "${FROZEN_ROOT}/summary.json" \
    --output-csv "${FROZEN_ROOT}/episodes_joined.csv"
}

summarize_finetune() {
  "${PYTHON_CMD[@]}" scripts/summarize_s1_static_chain.py \
    --protocol s1_static_obstacle_finetune_final \
    --eval "4301=${FINETUNE_ROOT}/eval/seed_4301_final.csv" \
    --eval "4302=${FINETUNE_ROOT}/eval/seed_4302_final.csv" \
    --eval "4303=${FINETUNE_ROOT}/eval/seed_4303_final.csv" \
    --output-json "${FINETUNE_ROOT}/summary.json" \
    --output-csv "${FINETUNE_ROOT}/episodes_joined.csv"
}

case "${STAGE}" in
  eval-frozen)
    mkdir -p "${FROZEN_ROOT}/eval"
    "${PYTHON_CMD[@]}" scripts/evaluate.py --config configs/experiments/reaching_incremental/s1_seed4301.yaml --checkpoint "${S0_ACTOR_4301}" --output "${FROZEN_ROOT}/eval/seed_4301_final.csv"
    "${PYTHON_CMD[@]}" scripts/evaluate.py --config configs/experiments/reaching_incremental/s1_seed4302.yaml --checkpoint "${S0_ACTOR_4302}" --output "${FROZEN_ROOT}/eval/seed_4302_final.csv"
    "${PYTHON_CMD[@]}" scripts/evaluate.py --config configs/experiments/reaching_incremental/s1_seed4303.yaml --checkpoint "${S0_ACTOR_4303}" --output "${FROZEN_ROOT}/eval/seed_4303_final.csv"
    summarize_frozen
    ;;
  finetune)
    "${PYTHON_CMD[@]}" scripts/train.py --config configs/experiments/reaching_incremental/s1_static_finetune_seed4301.yaml --resume-actor "${S0_ACTOR_4301}" --resume-state "${S0_STATE_4301}" --start-step 240000
    "${PYTHON_CMD[@]}" scripts/train.py --config configs/experiments/reaching_incremental/s1_static_finetune_seed4302.yaml --resume-actor "${S0_ACTOR_4302}" --resume-state "${S0_STATE_4302}" --start-step 280000
    "${PYTHON_CMD[@]}" scripts/train.py --config configs/experiments/reaching_incremental/s1_static_finetune_seed4303.yaml --resume-actor "${S0_ACTOR_4303}" --resume-state "${S0_STATE_4303}" --start-step 220000
    ;;
  select-finetune)
    "${PYTHON_CMD[@]}" scripts/select_checkpoint.py --config configs/experiments/reaching_incremental/s1_static_finetune_seed4301.yaml --run-dir "${TRAIN_RUN_4301}"
    "${PYTHON_CMD[@]}" scripts/select_checkpoint.py --config configs/experiments/reaching_incremental/s1_static_finetune_seed4302.yaml --run-dir "${TRAIN_RUN_4302}"
    "${PYTHON_CMD[@]}" scripts/select_checkpoint.py --config configs/experiments/reaching_incremental/s1_static_finetune_seed4303.yaml --run-dir "${TRAIN_RUN_4303}"
    ;;
  eval-finetune)
    mkdir -p "${FINETUNE_ROOT}/eval"
    CHECKPOINT_4301="$("${PYTHON_CMD[@]}" -c 'import csv; print(next(csv.DictReader(open("outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4301/link_fixed_static_obstacle_finetune_seed4301_steps440000/checkpoint_selection/selected_checkpoint.csv")))["checkpoint"])')"
    CHECKPOINT_4302="$("${PYTHON_CMD[@]}" -c 'import csv; print(next(csv.DictReader(open("outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4302/link_fixed_static_obstacle_finetune_seed4302_steps480000/checkpoint_selection/selected_checkpoint.csv")))["checkpoint"])')"
    CHECKPOINT_4303="$("${PYTHON_CMD[@]}" -c 'import csv; print(next(csv.DictReader(open("outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4303/link_fixed_static_obstacle_finetune_seed4303_steps420000/checkpoint_selection/selected_checkpoint.csv")))["checkpoint"])')"
    "${PYTHON_CMD[@]}" scripts/evaluate.py --config configs/experiments/reaching_incremental/s1_static_finetune_seed4301.yaml --checkpoint "${CHECKPOINT_4301}" --output "${FINETUNE_ROOT}/eval/seed_4301_final.csv"
    "${PYTHON_CMD[@]}" scripts/evaluate.py --config configs/experiments/reaching_incremental/s1_static_finetune_seed4302.yaml --checkpoint "${CHECKPOINT_4302}" --output "${FINETUNE_ROOT}/eval/seed_4302_final.csv"
    "${PYTHON_CMD[@]}" scripts/evaluate.py --config configs/experiments/reaching_incremental/s1_static_finetune_seed4303.yaml --checkpoint "${CHECKPOINT_4303}" --output "${FINETUNE_ROOT}/eval/seed_4303_final.csv"
    summarize_finetune
    ;;
  summarize-frozen)
    summarize_frozen
    ;;
  summarize-finetune)
    summarize_finetune
    ;;
  *)
    echo "unknown stage: ${STAGE}" >&2
    exit 2
    ;;
esac
