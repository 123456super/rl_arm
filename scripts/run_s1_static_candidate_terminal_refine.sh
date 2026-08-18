#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
read -r -a PYTHON_CMD <<< "${PYTHON:-python}"

ROOT="outputs/reaching_incremental/s1_static_candidate_terminal_refine"

CONFIG_4301="configs/experiments/reaching_incremental/s1_static_candidate_terminal_refine_seed4301.yaml"
CONFIG_4302="configs/experiments/reaching_incremental/s1_static_candidate_terminal_refine_seed4302.yaml"
CONFIG_4303="configs/experiments/reaching_incremental/s1_static_candidate_terminal_refine_seed4303.yaml"

S1_ACTOR_4301="outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4301/link_fixed_static_obstacle_finetune_seed4301_steps440000/actor_step_440000.pt"
S1_ACTOR_4302="outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4302/link_fixed_static_obstacle_finetune_seed4302_steps480000/actor_step_480000.pt"
S1_ACTOR_4303="outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4303/link_fixed_static_obstacle_finetune_seed4303_steps420000/actor_step_420000.pt"

S1_STATE_4301="outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4301/link_fixed_static_obstacle_finetune_seed4301_steps440000/agent_state_step_440000.pt"
S1_STATE_4302="outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4302/link_fixed_static_obstacle_finetune_seed4302_steps480000/agent_state_step_480000.pt"
S1_STATE_4303="outputs/reaching_incremental/s1_static_obstacle_finetune/train/seed_4303/link_fixed_static_obstacle_finetune_seed4303_steps420000/agent_state_step_420000.pt"

RUN_4301="${ROOT}/train/seed_4301/link_fixed_static_candidate_terminal_refine_seed4301_steps560000"
RUN_4302="${ROOT}/train/seed_4302/link_fixed_static_candidate_terminal_refine_seed4302_steps600000"
RUN_4303="${ROOT}/train/seed_4303/link_fixed_static_candidate_terminal_refine_seed4303_steps540000"

config_for_seed() {
  local seed="$1"
  case "${seed}" in
    4301) echo "${CONFIG_4301}" ;;
    4302) echo "${CONFIG_4302}" ;;
    4303) echo "${CONFIG_4303}" ;;
    *) echo "unknown seed: ${seed}" >&2; exit 2 ;;
  esac
}

run_dir_for_seed() {
  local seed="$1"
  case "${seed}" in
    4301) echo "${RUN_4301}" ;;
    4302) echo "${RUN_4302}" ;;
    4303) echo "${RUN_4303}" ;;
    *) echo "unknown seed: ${seed}" >&2; exit 2 ;;
  esac
}

selected_checkpoint_for_seed() {
  local seed="$1"
  local run_dir
  run_dir="$(run_dir_for_seed "${seed}")"
  "${PYTHON_CMD[@]}" -c '
import csv
import pathlib
import sys
path = pathlib.Path(sys.argv[1]) / "checkpoint_selection" / "selected_checkpoint.csv"
with path.open(newline="", encoding="utf-8") as file:
    print(next(csv.DictReader(file))["checkpoint"])
' "${run_dir}"
}

train_seed() {
  local seed="$1"
  case "${seed}" in
    4301)
      "${PYTHON_CMD[@]}" scripts/train.py --config "${CONFIG_4301}" --resume-actor "${S1_ACTOR_4301}" --resume-state "${S1_STATE_4301}" --start-step 440000 --no-restore-optimizers
      ;;
    4302)
      "${PYTHON_CMD[@]}" scripts/train.py --config "${CONFIG_4302}" --resume-actor "${S1_ACTOR_4302}" --resume-state "${S1_STATE_4302}" --start-step 480000 --no-restore-optimizers
      ;;
    4303)
      "${PYTHON_CMD[@]}" scripts/train.py --config "${CONFIG_4303}" --resume-actor "${S1_ACTOR_4303}" --resume-state "${S1_STATE_4303}" --start-step 420000 --no-restore-optimizers
      ;;
    *) echo "unknown seed: ${seed}" >&2; exit 2 ;;
  esac
}

select_seed() {
  local seed="$1"
  "${PYTHON_CMD[@]}" scripts/select_checkpoint.py \
    --config "$(config_for_seed "${seed}")" \
    --run-dir "$(run_dir_for_seed "${seed}")"
}

eval_seed() {
  local seed="$1"
  mkdir -p "${ROOT}/eval"
  "${PYTHON_CMD[@]}" scripts/evaluate.py \
    --config "$(config_for_seed "${seed}")" \
    --checkpoint "$(selected_checkpoint_for_seed "${seed}")" \
    --output "${ROOT}/eval/seed_${seed}_final.csv"
}

summarize() {
  "${PYTHON_CMD[@]}" scripts/summarize_s1_static_chain.py \
    --protocol s1_static_candidate_terminal_refine_final \
    --eval "4301=${ROOT}/eval/seed_4301_final.csv" \
    --eval "4302=${ROOT}/eval/seed_4302_final.csv" \
    --eval "4303=${ROOT}/eval/seed_4303_final.csv" \
    --output-json "${ROOT}/summary.json" \
    --output-csv "${ROOT}/episodes_joined.csv"
}

case "${STAGE}" in
  train-4301) train_seed 4301 ;;
  train-4302) train_seed 4302 ;;
  train-4303) train_seed 4303 ;;
  select-4301) select_seed 4301 ;;
  select-4302) select_seed 4302 ;;
  select-4303) select_seed 4303 ;;
  eval-4301) eval_seed 4301 ;;
  eval-4302) eval_seed 4302 ;;
  eval-4303) eval_seed 4303 ;;
  summarize) summarize ;;
  *)
    echo "usage: bash scripts/run_s1_static_candidate_terminal_refine.sh {train-4301|train-4302|train-4303|select-4301|select-4302|select-4303|eval-4301|eval-4302|eval-4303|summarize}" >&2
    exit 2
    ;;
esac
