#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_risk_sac.algorithms.homotopy_curriculum import HomotopyCurriculum
from rl_risk_sac.algorithms.homotopy_replay import HomotopyReplayBuffer
from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.utils.config import load_config


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(path: Path, agent, replay, curriculum, counters, config, env, active_episode) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "protocol": config["thesis"]["protocol"], "agent": agent.state_dict(),
        "replay": replay.state_dict(), "curriculum": curriculum.state_dict(),
        "counters": counters, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "environment_rng": env.rng.bit_generator.state,
        "active_episode": active_episode,
        "config": config,
    }, path)


def restore_torch_rng(state: dict) -> None:
    torch.set_rng_state(state["torch_rng"].cpu())
    if torch.cuda.is_available() and state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_rng"]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Chapters 1--3 task-first safety homotopy")
    parser.add_argument("--config", default="configs/experiments/thesis_homotopy.yaml")
    parser.add_argument("--stage", choices=("s0", "s1", "s2"))
    parser.add_argument("--resume", help="complete checkpoint from the preceding block or stage")
    parser.add_argument("--steps", type=int, help="override this block's environment steps")
    parser.add_argument("--run-name")
    parser.add_argument("--seed", type=int, help="override training seed")
    parser.add_argument("--validation-mode", action="store_true", help="small replay/batch for implementation checks only")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    thesis = config["thesis"]
    stage = args.stage or str(thesis["stage"])
    seed = int(args.seed if args.seed is not None else config["seed"]); config["seed"] = seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{stamp}_{stage}_seed{seed}"
    output = ROOT / str(config["train"]["output_dir"]) / run_name
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = output / "checkpoints"; checkpoints.mkdir()
    env = ThesisHomotopyEnv(config)
    agent = ThesisSACAgent(55, 6, config)
    capacities = ({"none": 512, "static": 512, "dynamic": 512} if args.validation_mode else None)
    replay = HomotopyReplayBuffer(
        55, 6, str(agent.device), capacities=capacities, seed=seed,
        reward_gamma=float(config["sac"]["gamma"]), reward_horizon=int(thesis["horizon"]),
    )
    curriculum = HomotopyCurriculum(stage, seed=seed, ramp_steps=int(thesis["ramp_steps"]))
    counters = {"global_step": 0, "stage_step": 0, "updates": 0, "episodes": 0, "block": 1,
                "checkpoint_index": 0, "random_steps_used": 0, "replay_redistributions": 0}
    resume_mode = None
    active_episode = None
    if args.resume:
        state = torch.load(args.resume, map_location=agent.device, weights_only=False)
        if state.get("protocol") != thesis["protocol"]:
            raise ValueError(
                f"checkpoint protocol {state.get('protocol')!r} does not match "
                f"configured protocol {thesis['protocol']!r}"
            )
        previous_stage = state["curriculum"]["stage"]
        if {("s0", "s1"), ("s1", "s2")}.intersection({(previous_stage, stage)}):
            agent.load_state_dict(state["agent"]); replay.load_state_dict(state["replay"])
            inherited = state["counters"]; counters["global_step"] = inherited["global_step"]
            counters["updates"] = inherited["updates"]; counters["episodes"] = inherited["episodes"]
            counters["random_steps_used"] = inherited["random_steps_used"]
            random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
            restore_torch_rng(state)
            curriculum.rng.bit_generator.state = state["curriculum"]["rng_state"]
            resume_mode = f"stage_transition_{previous_stage}_to_{stage}"
        elif previous_stage == stage:
            agent.load_state_dict(state["agent"]); replay.load_state_dict(state["replay"])
            curriculum.load_state_dict(state["curriculum"]); counters.update(state["counters"])
            random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
            restore_torch_rng(state)
            counters["block"] += 1; counters["stage_step"] = 0
            active_episode = state.get("active_episode")
            resume_mode = "same_stage_continuation"
        else:
            raise ValueError(f"invalid stage transition {previous_stage}->{stage}")
        env.rng.bit_generator.state = state["environment_rng"]
    elif stage != "s0":
        raise ValueError("S1/S2 must use --resume with the preceding stage's complete checkpoint")

    manifest = {
        "protocol": config["thesis"]["protocol"], "stage": stage, "seed": seed,
        "resume": args.resume, "resume_mode": resume_mode, "config": str(args.config),
        "urdf_sha256": file_sha256(ROOT / config["robot"]["urdf"]),
        "joint_names": config["robot"]["joint_names"], "tool_link_name": config["robot"]["tool_link_name"],
        "observation_dim": 55, "action_dim": 6, "external_collision_bodies": [],
        "validation_mode": args.validation_mode,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path = output / "episodes.csv"
    fields = ["episode", "global_step", "stage_step", "scene", "length", "return", "task_reached",
              "collision_assisted_reach", "safe_success", "obstacle_collision", "self_collision",
              "environment_collision", "joint_limit", "timeout", "xi", "strict", "rolling_task_reach_rate",
              "eligible_steps", "strict_steps"]
    transition_fields = ["global_step", "stage_step", "episode", "episode_step", "scene", "xi", "strict",
                         "reward", "r_goal", "c_proximity", "hard_penalty", "safety_penalty",
                         "terminal_guard_penalty", "risk_max", "d_min",
                         "task_reached", "obstacle_collision", "self_collision", "environment_collision", "joint_limit",
                         "sample_none", "sample_static", "sample_dynamic"]
    total_steps = int(args.steps or config["train"]["total_steps"])
    save_interval = int(config["train"]["save_interval"]); last_saved_step = -1
    batch_size = 8 if args.validation_mode else int(config["sac"]["batch_size"])
    update_after = 8 if args.validation_mode else int(config["sac"]["update_after"])
    with log_path.open("w", newline="", encoding="utf-8") as handle, \
            (output / "transitions.csv").open("w", newline="", encoding="utf-8") as transition_handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        transition_writer = csv.DictWriter(transition_handle, fieldnames=transition_fields); transition_writer.writeheader()
        while counters["stage_step"] < total_steps:
            if active_episode is None:
                scene = curriculum.choose_scene(); xi, strict = curriculum.contract(scene)
                env.configure_episode(scene, xi=xi, strict=strict)
                obs, _ = env.reset(); episode_return = 0.0; episode_length = 0
                final = {}; terminated = truncated = False
            else:
                scene = active_episode["scene"]; xi, strict = curriculum.contract(scene)
                obs = env.restore_episode_state(active_episode["environment"])
                episode_return = active_episode["return"]; episode_length = active_episode["length"]
                final = active_episode["final"]; terminated = truncated = False; active_episode = None
            while not (terminated or truncated) and counters["stage_step"] < total_steps:
                if stage == "s0" and counters["random_steps_used"] < int(config["sac"]["warmup_steps"]):
                    action = env.action_space.sample(); counters["random_steps_used"] += 1
                else: action = agent.select_action(obs)
                next_obs, reward, _, terminated, truncated, info = env.step(action)
                replay.add(scene, obs, action, next_obs, terminated, info, counters["episodes"], episode_length)
                obs = next_obs; final = info; episode_return += reward; episode_length += 1
                counters["global_step"] += 1; counters["stage_step"] += 1
                if len(replay) >= max(update_after, batch_size):
                    batch = replay.sample(stage, curriculum.xi_map(), batch_size)
                    agent.update(batch); counters["updates"] += 1
                transition_writer.writerow({
                    "global_step": counters["global_step"], "stage_step": counters["stage_step"],
                    "episode": counters["episodes"], "episode_step": episode_length, "scene": scene,
                    "xi": xi, "strict": int(strict), "reward": reward, "r_goal": info["r_goal"],
                    "c_proximity": info["c_proximity"], "hard_penalty": info["hard_penalty"],
                    "safety_penalty": info["safety_penalty"],
                    "terminal_guard_penalty": info["terminal_guard_penalty"],
                    "risk_max": info["control_max_risk"],
                    "d_min": info["control_min_distance"], "task_reached": int(info["task_reached"]),
                    "obstacle_collision": int(info["obstacle_collision"]), "self_collision": int(info["self_collision"]),
                    "environment_collision": int(info["environment_collision"]), "joint_limit": int(info["joint_limit"]),
                    "sample_none": replay.last_sample_counts["none"], "sample_static": replay.last_sample_counts["static"],
                    "sample_dynamic": replay.last_sample_counts["dynamic"],
                }); transition_handle.flush()
                if counters["stage_step"] % save_interval == 0 and not (terminated or truncated):
                    active_episode = {"scene": scene, "return": episode_return, "length": episode_length,
                                      "final": final, "environment": env.episode_state_dict()}
                    counters["checkpoint_index"] += 1; counters["replay_redistributions"] = replay.redistribution_count
                    checkpoint = checkpoints / f"step_{counters['stage_step']:07d}.pt"
                    save_checkpoint(checkpoint, agent, replay, curriculum, counters, config, env, active_episode)
                    agent.save_actor(checkpoints / f"actor_step_{counters['stage_step']:07d}.pt")
                    print(f"saved {checkpoint} scene={scene} xi={xi:.4f} replay={len(replay)}", flush=True)
                    last_saved_step = counters["stage_step"]
                    active_episode = None
            if not (terminated or truncated):
                active_episode = {"scene": scene, "return": episode_return, "length": episode_length,
                                  "final": final, "environment": env.episode_state_dict()}
                break
            update = curriculum.finish_episode(scene, bool(final.get("task_reached", False)), episode_length)
            if update["became_strict"]:
                report = replay.strictify(scene)
                curriculum.states[scene].replay_strictified = True
                manifest.setdefault("strictification", []).append({"scene": scene, "global_step": counters["global_step"], **report})
                (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            state = curriculum.states.get(scene)
            writer.writerow({
                "episode": counters["episodes"], "global_step": counters["global_step"], "stage_step": counters["stage_step"],
                "scene": scene, "length": episode_length, "return": episode_return,
                "task_reached": int(final.get("task_reached", False)),
                "collision_assisted_reach": int(final.get("collision_assisted_reach", False)),
                "safe_success": int(final.get("safe_success", False)),
                "obstacle_collision": int(final.get("obstacle_collision", False)), "self_collision": int(final.get("self_collision", False)),
                "environment_collision": int(final.get("environment_collision", False)), "joint_limit": int(final.get("joint_limit", False)),
                "timeout": int(truncated), "xi": 1.0 if state is None else state.xi,
                "strict": 1 if state is None else int(state.strict),
                "rolling_task_reach_rate": "" if state is None else update["rolling_task_reach_rate"],
                "eligible_steps": 0 if state is None else state.eligible_steps, "strict_steps": 0 if state is None else state.strict_steps,
            }); handle.flush(); counters["episodes"] += 1
            if (counters["stage_step"] % save_interval == 0 or counters["stage_step"] >= total_steps) and last_saved_step != counters["stage_step"]:
                counters["checkpoint_index"] += 1
                checkpoint = checkpoints / f"step_{counters['stage_step']:07d}.pt"
                counters["replay_redistributions"] = replay.redistribution_count
                save_checkpoint(checkpoint, agent, replay, curriculum, counters, config, env, None)
                agent.save_actor(checkpoints / f"actor_step_{counters['stage_step']:07d}.pt")
                print(f"saved {checkpoint} scene={scene} xi={xi:.4f} replay={len(replay)}", flush=True)
                last_saved_step = counters["stage_step"]
        if active_episode is not None and last_saved_step != counters["stage_step"]:
            counters["checkpoint_index"] += 1; counters["replay_redistributions"] = replay.redistribution_count
            checkpoint = checkpoints / f"step_{counters['stage_step']:07d}.pt"
            save_checkpoint(checkpoint, agent, replay, curriculum, counters, config, env, active_episode)
            agent.save_actor(checkpoints / f"actor_step_{counters['stage_step']:07d}.pt")
            print(f"saved {checkpoint} active_episode=1 replay={len(replay)}", flush=True)
    env.close()
    summary = {"stage": stage, **counters, "replay_size": len(replay), "xi": curriculum.xi_map(),
               "strict_steps": {k: v.strict_steps for k, v in curriculum.states.items()}}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
