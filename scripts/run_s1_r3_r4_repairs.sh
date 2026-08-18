#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-}"
BUCKET="${2:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
read -r -a PYTHON_CMD <<< "${PYTHON:-python}"

blind_manifest="configs/experiments/reaching_incremental/manifests/s1_blind_final_v1.json"

config_for() {
  local bucket="$1" seed="$2"
  echo "configs/experiments/reaching_incremental/s1_${bucket}_repair_seed${seed}.yaml"
}

run_dir_for() {
  local bucket="$1" seed="$2"
  local steps
  if [[ "$bucket" == r3_candidate ]]; then
    case "$seed" in
      4301) steps=680000 ;;
      4302) steps=650000 ;;
      4303) steps=700000 ;;
      *) echo "unknown seed: $seed" >&2; exit 2 ;;
    esac
  elif [[ "$bucket" == r4_hard_case ]]; then
    case "$seed" in
      4301) steps=840000 ;;
      4302) steps=800000 ;;
      4303) steps=860000 ;;
      *) echo "unknown seed: $seed" >&2; exit 2 ;;
    esac
  else
    echo "unknown bucket: $bucket" >&2; exit 2
  fi
  echo "outputs/reaching_incremental/s1_${bucket}_repair/train/seed_${seed}/link_fixed_s1_${bucket}_repair_seed${seed}_steps${steps}"
}

selected_for() {
  "${PYTHON_CMD[@]}" -c '
import csv, pathlib, sys
path = pathlib.Path(sys.argv[1]) / "checkpoint_selection" / "selected_checkpoint.csv"
with path.open(newline="", encoding="utf-8") as f:
    print(next(csv.DictReader(f))["checkpoint"])
' "$(run_dir_for "$1" "$2")"
}

latest_actor_for_run() {
  "${PYTHON_CMD[@]}" -c '
import re, sys
from pathlib import Path

run_dir = Path(sys.argv[1])
pattern = re.compile(r"actor_step_(\d+)\.pt$")
candidates: list[tuple[int, Path]] = []
for path in run_dir.glob("actor_step_*.pt"):
    match = pattern.fullmatch(path.name)
    if match is not None:
        candidates.append((int(match.group(1)), path))
if not candidates:
    raise SystemExit(1)
print(max(candidates)[1])
' "$1"
}

require_r3_selected() {
  local seed="$1"
  local run_dir
  run_dir="$(run_dir_for r3_candidate "$seed")"
  local selected_csv="${run_dir}/checkpoint_selection/selected_checkpoint.csv"
  if [[ ! -f "$selected_csv" ]]; then
    echo "missing R3 checkpoint selection for seed ${seed}: ${selected_csv}" >&2
    echo "next: bash scripts/run_s1_r3_r4_repairs.sh select r3_candidate ${seed}" >&2
    exit 2
  fi
}

require_run_dir() {
  local bucket="$1" seed="$2"
  local run_dir
  run_dir="$(run_dir_for "$bucket" "$seed")"
  if [[ ! -d "$run_dir" ]]; then
    echo "missing training run directory: ${run_dir}" >&2
    echo "next: bash scripts/run_s1_r3_r4_repairs.sh train ${bucket} ${seed}" >&2
    exit 2
  fi
}

train_one() {
  local bucket="$1" seed="$2" config actor state start
  config="$(config_for "$bucket" "$seed")"
  if [[ "$bucket" == r3_candidate ]]; then
    case "$seed" in
      4301) actor="outputs/reaching_incremental/s1_static_candidate_terminal_refine/train/seed_4301/link_fixed_static_candidate_terminal_refine_seed4301_steps560000/actor_step_520000.pt"; state="outputs/reaching_incremental/s1_static_candidate_terminal_refine/train/seed_4301/link_fixed_static_candidate_terminal_refine_seed4301_steps560000/agent_state_step_520000.pt"; start=520000 ;;
      4302) actor="outputs/reaching_incremental/s1_static_candidate_terminal_refine/train/seed_4302/link_fixed_static_candidate_terminal_refine_seed4302_steps600000/actor_step_490000.pt"; state="outputs/reaching_incremental/s1_static_candidate_terminal_refine/train/seed_4302/link_fixed_static_candidate_terminal_refine_seed4302_steps600000/agent_state_step_490000.pt"; start=490000 ;;
      4303) actor="outputs/reaching_incremental/s1_static_candidate_terminal_refine/train/seed_4303/link_fixed_static_candidate_terminal_refine_seed4303_steps540000/actor_step_540000.pt"; state="outputs/reaching_incremental/s1_static_candidate_terminal_refine/train/seed_4303/link_fixed_static_candidate_terminal_refine_seed4303_steps540000/agent_state_step_540000.pt"; start=540000 ;;
      *) echo "unknown seed: $seed" >&2; exit 2 ;;
    esac
  elif [[ "$bucket" == r4_hard_case ]]; then
    case "$seed" in
      4301|4302|4303)
        local run_dir latest_actor
        run_dir="$(run_dir_for "$bucket" "$seed")"
        if latest_actor="$(latest_actor_for_run "$run_dir" 2>/dev/null)"; then
          actor="$latest_actor"
        else
          require_r3_selected "$seed"
          actor="$(selected_for r3_candidate "$seed")"
        fi
        state="${actor/actor_step_/agent_state_step_}"
        start="$(basename "$actor" | sed -E 's/actor_step_([0-9]+)\.pt/\1/')"
        ;;
      *) echo "unknown seed: $seed" >&2; exit 2 ;;
    esac
  else
    echo "unknown bucket: $bucket" >&2; exit 2
  fi
  "${PYTHON_CMD[@]}" scripts/train.py --config "$config" --resume-actor "$actor" --resume-state "$state" --start-step "$start" --no-restore-optimizers
}

select_one() {
  local bucket="$1" seed="$2"
  if [[ "$bucket" == r4_hard_case ]]; then
    require_r3_selected "$seed"
  fi
  require_run_dir "$bucket" "$seed"
  "${PYTHON_CMD[@]}" scripts/select_checkpoint.py --config "$(config_for "$bucket" "$seed")" --run-dir "$(run_dir_for "$bucket" "$seed")"
}

blind_eval_one() {
  local bucket="$1"
  local seed="$2"
  local out="outputs/reaching_incremental/s1_${bucket}_repair/eval_blind/seed_${seed}_blind_final.csv"
  require_run_dir "$bucket" "$seed"
  mkdir -p "$(dirname "$out")"
  "${PYTHON_CMD[@]}" scripts/evaluate.py --config "$(config_for "$bucket" "$seed")" --checkpoint "$(selected_for "$bucket" "$seed")" --seed-manifest "$blind_manifest" --episodes 200 --output "$out"
}

summarize_blind() {
  local bucket="$1"
  local root="outputs/reaching_incremental/s1_${bucket}_repair/eval_blind"
  "${PYTHON_CMD[@]}" scripts/summarize_s1_static_chain.py \
    --protocol "s1_${bucket}_repair_blind_final" \
    --eval "4301=${root}/seed_4301_blind_final.csv" \
    --eval "4302=${root}/seed_4302_blind_final.csv" \
    --eval "4303=${root}/seed_4303_blind_final.csv" \
    --output-json "${root}/summary.json" \
    --output-csv "${root}/episodes_joined.csv"
}

case "$STAGE" in
  train) train_one "$BUCKET" "${3:?seed required}" ;;
  select) select_one "$BUCKET" "${3:?seed required}" ;;
  blind-eval) blind_eval_one "$BUCKET" "${3:?seed required}" ;;
  summarize-blind) summarize_blind "$BUCKET" ;;
  *) echo "usage: $0 {train|select|blind-eval|summarize-blind} {r3_candidate|r4_hard_case} [4301|4302|4303]" >&2; exit 2 ;;
esac
