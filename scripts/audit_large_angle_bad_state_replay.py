#!/usr/bin/env python3
"""Read-only audit of local O7--O9 evaluation states against training replay."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from rl_risk_sac.algorithms.homotopy_replay import HomotopyReplayBuffer, _vectorized_replay_rewards
from rl_risk_sac.algorithms.thesis_sac import ThesisSACAgent
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool
from rl_risk_sac.utils.config import load_config
from scripts.analyze_large_angle_alignment_onset import contiguous_collapse
from scripts.analyze_orientation_state_provenance import STEP_INFO_KEYS, evaluate
from scripts.evaluate_thesis_homotopy import physical_cpu_ids

RUN = ROOT / "outputs/serial_hybrid_keypoint_jacobian_auto_chain/s0_seed11001_hybrid_keypoint_jacobian_auto_chain_4000k"
FEATURES = {"q": slice(0, 6), "qdot": slice(6, 12), "j_kp": slice(21, 75)}
THRESHOLDS = {"rho_r": .10, "rho_p": .12, "steps": 20,
              "q": .25, "qdot": .35, "j_kp": .18}


def describe(values):
    values = np.asarray(list(values) if not isinstance(values, np.ndarray) else values,
                        dtype=np.float64)
    return {"count": int(len(values)), "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None}


def select_centers(episodes, rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["episode"]].append(row)
    bad, good = defaultdict(list), []
    for episode in episodes:
        timeline = sorted(grouped[episode["episode"]], key=lambda row: row["step"])
        if episode["success"]:
            crossing = episode["crossing_0p6_step"]
            candidate = next((r for r in timeline if .6 <= r["rho_orientation"] <= .8
                              and (crossing is None or r["step"] < crossing)
                              and r["orientation_progress_rad_s"] > 0), None)
            if candidate is not None:
                good.append(candidate)
            continue
        event = contiguous_collapse(timeline)
        if event is not None:
            bad["low_alignment"].append(next(r for r in timeline if r["step"] == event["onset_step"]))
        # First five actual steps in the band with <= .05 rad/s mean progress,
        # allowing small positive/negative oscillations to count as stagnation.
        for index in range(4, len(timeline)):
            window = timeline[index - 4:index + 1]
            if (all(.6 <= r["rho_orientation"] <= 1. for r in window)
                and all(window[j]["step"] == window[j - 1]["step"] + 1 for j in range(1, 5))
                and np.mean([r["orientation_progress_rad_s"] for r in window]) <= .05):
                bad["stagnation"].append(window[0])
                break
        crossing = episode["crossing_0p6_step"]
        if crossing is not None:
            rebound = next((r for r in timeline if r["step"] >= crossing
                            and .6 <= r["rho_orientation"] <= 1.), None)
            if rebound is not None:
                bad["post_crossing_rebound"].append(rebound)
    return bad, good


def neighborhood(part, centers):
    """Return pose/time candidates and full-state neighbors for a replay partition."""
    n = part.size
    pose = np.zeros(n, bool)
    local = np.zeros(n, bool)
    per_center = []
    if not n or not centers:
        return pose, local, [0] * len(centers)
    raw, obs = part.raw[:n], part.obs[:n]
    step = part.step[:n]
    for center in centers:
        candidates = (np.abs(raw[:, 2] - center["rho_orientation"]) <= THRESHOLDS["rho_r"])
        candidates &= np.abs(raw[:, 0] - center["rho_position"]) <= THRESHOLDS["rho_p"]
        candidates &= np.abs(step - center["step"]) <= THRESHOLDS["steps"]
        pose |= candidates
        indices = np.flatnonzero(candidates)
        if not len(indices):
            per_center.append(0)
            continue
        query = center["observation"]
        keep = np.ones(len(indices), bool)
        for feature in ("q", "qdot", "j_kp"):
            sl = FEATURES[feature]
            distance = np.sqrt(np.mean(np.square(obs[indices, sl] - query[sl]), axis=1))
            keep &= distance <= THRESHOLDS[feature]
        local[indices[keep]] = True
        per_center.append(int(keep.sum()))
    return pose, local, per_center


def episode_labels(path, maximum_step):
    result = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["global_step"]) > maximum_step:
                break
            result[int(row["episode"])] = (
                "success" if int(row["task_reached"]) else
                "timeout" if int(row["timeout"]) else "failure"
            )
    return result


def partition_metrics(part, mask, labels, reward_config, gamma, horizon, lambda_self):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return {"count": 0}
    raw = part.raw[indices].astype(np.float64)
    rewards, _ = _vectorized_replay_rewards(
        part.raw[indices], part.done[indices], np.ones(len(indices)),
        lambda_self, gamma, reward_config,
    )
    outcomes = [labels.get(int(e), "unfinished_or_evicted") for e in part.episode[indices]]
    error = part.obs[indices, 78:81].astype(np.float64)
    velocity = part.next_obs[indices, 86:89].astype(np.float64)
    norm = np.linalg.norm(error, axis=1) * np.linalg.norm(velocity, axis=1)
    valid = (norm > 1e-10) & np.all(np.abs(velocity) < 1., axis=1)
    alignment = np.sum(error[valid] * velocity[valid], axis=1) / norm[valid]
    dt = .05
    return {
        "count": len(indices),
        "source_counts": {name: outcomes.count(name) for name in
                          ("success", "timeout", "failure", "unfinished_or_evicted")},
        "orientation_progress_rad_s": describe((raw[:, 2] - raw[:, 3]) / dt),
        "position_progress_m_s": describe((raw[:, 0] - raw[:, 1]) / dt),
        "orientation_rebound_fraction": float(np.mean(raw[:, 3] > raw[:, 2])),
        "reward_relabelled": describe(rewards),
        "remaining_time_fraction": describe(1 - part.step[indices] / horizon),
        "mean_q": part.obs[indices, FEATURES["q"]].mean(axis=0).tolist(),
        "mean_qdot": part.obs[indices, FEATURES["qdot"]].mean(axis=0).tolist(),
        "mean_j_kp_rms": float(np.sqrt(np.mean(part.obs[indices, FEATURES["j_kp"]] ** 2))),
        "alignment": None,  # Pre-action command alignment needs J_omega.
        "measured_endpoint_angular_alignment": describe(alignment),
        "measured_endpoint_negative_alignment_fraction": float(np.mean(alignment < 0)) if len(alignment) else None,
        "measured_endpoint_alignment_excluded": int((~valid).sum()),
    }


def pool_list(replay_state):
    pools = {
        "recent": replay_state["s0_current"],
        "success": replay_state["s0_current_success"],
        "anchor": replay_state["s0_anchor"],
    }
    pools.update({f"history:{k}": v for k, v in replay_state["s0_history"].items()})
    semantic = replay_state["semantic_long_term"]
    for bin_id, parts in semantic["slots"].items():
        for index, part in enumerate(parts):
            pools[f"semantic:{bin_id}:{index}"] = part
    return pools


def critic_metrics(agent, obs, actions, next_obs, raw, done, reward_config, gamma,
                   lambda_self,
                   *, seed=51001):
    if not len(obs):
        return {"count": 0}, None
    rewards, _ = _vectorized_replay_rewards(raw, done, np.ones(len(obs)), lambda_self, gamma,
                                             reward_config)
    result = defaultdict(list)
    fixed_values = []
    alpha = agent.alpha.detach()
    with torch.no_grad():
        for start in range(0, len(obs), 512):
            end = start + 512
            o = torch.as_tensor(obs[start:end], dtype=torch.float32, device=agent.device)
            a = torch.as_tensor(actions[start:end], dtype=torch.float32, device=agent.device)
            no = torch.as_tensor(next_obs[start:end], dtype=torch.float32, device=agent.device)
            r = torch.as_tensor(rewards[start:end, None], dtype=torch.float32, device=agent.device)
            d = torch.as_tensor(done[start:end], dtype=torch.float32, device=agent.device)
            q1, q2 = agent.q1(o, a), agent.q2(o, a)
            # Fixed seed gives reproducible per-state target estimates across groups.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + start)
                targets = []
                for _ in range(8):
                    next_action, log_prob = agent.actor.sample(no)
                    next_q = torch.minimum(agent.target_q1(no, next_action),
                                           agent.target_q2(no, next_action))
                    targets.append(r + gamma * (1 - d) * (next_q - alpha * log_prob))
            ts = torch.stack(targets, dim=0)
            result["q_disagreement"].extend(torch.abs(q1 - q2).flatten().cpu().tolist())
            result["td_abs_mean_target"].extend(
                (.5 * (torch.abs(q1 - ts.mean(0)) + torch.abs(q2 - ts.mean(0))))
                .flatten().cpu().tolist())
            result["target_q_variance"].extend(ts.var(dim=0, unbiased=False).flatten().cpu().tolist())
            fixed_values.extend(torch.minimum(q1, q2).flatten().cpu().tolist())
    return {"count": len(obs), **{name: describe(values) for name, values in result.items()}}, np.asarray(fixed_values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--reuse-centers", type=Path,
                        help="Reuse centers from a completed audit; skip trajectory replay")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    config = load_config(ROOT / "configs/experiments/thesis_serial_hybrid_keypoint_jacobian_auto_chain.yaml")
    config["device"] = "cpu"
    checkpoint = args.run / "checkpoints/step_1000000.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    agent = ThesisSACAgent(166, 6, config)
    agent.load_state_dict(state["agent"])
    agent.actor.eval()
    if args.reuse_centers:
        cached = json.loads(args.reuse_centers.read_text())
        if cached["protocol"]["checkpoint"] != str(checkpoint):
            raise ValueError("cached centers use a different checkpoint")
        centers = {}
        for name, group in cached["centers"].items():
            centers[name] = []
            for item in group["individual_states"]:
                observation = np.zeros(166, dtype=np.float32)
                for feature, key in (("q", "q_normalized"), ("qdot", "qdot_normalized"),
                                     ("j_kp", "J_KP_normalized_clipped")):
                    observation[FEATURES[feature]] = np.asarray(item[key]).reshape(-1)
                centers[name].append({**item, "observation": observation,
                                      "rho_orientation": item["rho_R"], "rho_position": item["rho_p"],
                                      "actor_command_alignment": item["alignment"]})
        episode_counts = cached["episode_counts"]
    else:
        ids = [i for i in range(1000) if i % 10 in (7, 8, 9)]
        with ParallelThesisEnvPool(config, [51001 + 1_000_000 + i for i in range(args.num_envs)],
                                   cpu_ids=physical_cpu_ids(args.num_envs), step_info_keys=STEP_INFO_KEYS,
                                   start_method="fork" if "fork" in mp.get_all_start_methods() else "spawn") as pool:
            episodes, rows, _ = evaluate(
                agent, pool, config, episodes=len(ids), seed=51001, level_index=0,
                band_low=.6, band_high=1., damping=.05, counterfactuals=False,
                episode_ids=ids, capture_all_large_states=True, capture_observations=True,
            )
        bad, good = select_centers(episodes, rows)
        centers = {**bad, "good_descent": good}
        episode_counts = {"success": sum(e["success"] for e in episodes),
                          "failure": sum(not e["success"] for e in episodes)}
    labels = episode_labels(args.run / "episodes.csv", 1000000)
    pools = pool_list(state["replay"])
    reward_config = state["replay"]["reward_parameters"]
    gamma = float(state["replay"]["reward_gamma"])
    horizon = int(state["replay"]["reward_horizon"])
    with (args.run / "progress.csv").open(newline="") as handle:
        progress_rows = [row for row in csv.DictReader(handle)
                         if int(row["global_step"]) <= 1000000]
    progress = progress_rows[-1]
    lambda_self = float(progress["lambda_self"])
    masks = {name: {} for name in centers}
    coverage = {name: {} for name in centers}
    for group, queries in centers.items():
        for pool_name, part in pools.items():
            pose, local, per_center = neighborhood(part, queries)
            masks[group][pool_name] = (pose, local)
            coverage[group][pool_name] = {
                "total": part.size, "pose_time": partition_metrics(part, pose, labels, reward_config, gamma, horizon, lambda_self),
                "full_local": partition_metrics(part, local, labels, reward_config, gamma, horizon, lambda_self),
                "neighbors_per_center": per_center,
                "centers_with_zero_neighbors": per_center.count(0),
            }
    # No local state identifiers are recorded per historical update; this is
    # current-checkpoint sampling, with a separate unmodified RNG copy.
    replay = HomotopyReplayBuffer(166, 6, "cpu", reward_parameters=reward_config)
    replay.load_state_dict(state["replay"])
    sem = state["replay"]["semantic_long_term"]
    replay.configure_semantic_long_term(
        enabled=sem["enabled"], fraction=sem["fraction"],
        position_bins=sem["position_bins"], orientation_bins=sem["orientation_bins"],
        episodes_per_bin=sem["episodes_per_bin"], transitions_per_episode=sem["transitions_per_episode"],
        distance_min=sem["bounds"][0], distance_max=sem["bounds"][1],
        orientation_min=sem["bounds"][2], orientation_max=sem["bounds"][3], seed=51001,
    )
    part_names = {id(part): name for name, part in pools.items()}
    # The loaded replay owns the same partition objects from the checkpoint.
    sample_counts = {name: defaultdict(int) for name in centers}
    sample_by_source = {name: defaultdict(int) for name in centers}
    sample_sources = defaultdict(int)
    # The progress log records the effective level fractions; use the latest
    # record at or before checkpoint rather than guessing curriculum internals.
    prev_frac = float(progress["replay_previous_level_fraction"])
    recent_frac = float(progress["replay_current_recent_fraction"]) + float(progress["replay_semantic_long_term_fraction"])
    batches = 20
    for _ in range(batches):
        selected = replay._sample_joint_pose_rows(1024, replay.s0_current_scale,
                                                   prev_frac, recent_frac)
        for part, index, source in selected:
            sample_sources[source] += 1
            key = part_names[id(part)]
            for group in centers:
                pose, local = masks[group][key]
                sample_counts[group]["pose_time"] += int(pose[index])
                sample_counts[group]["full_local"] += int(local[index])
                sample_by_source[group][source] += int(local[index])
    del replay

    # Use unique training transitions in the recent pool for Critic comparisons.
    recent = pools["recent"]
    critic = {}
    fixed_q = {}
    selected_indices = {}
    for group in centers:
        indices = np.flatnonzero(masks[group]["recent"][1])
        selected_indices[group] = indices[:2048]
    for checkpoint_step in (800000, 900000, 1000000):
        checkpoint_progress = next(row for row in reversed(progress_rows)
                                   if int(row["global_step"]) <= checkpoint_step)
        if checkpoint_step != 1000000:
            other = torch.load(args.run / f"checkpoints/step_{checkpoint_step:07d}.pt",
                               map_location="cpu", weights_only=False)
            agent.load_state_dict(other["agent"])
            del other
        else:
            agent.load_state_dict(state["agent"])
        agent.actor.eval()
        critic[str(checkpoint_step)] = {}
        fixed_q[str(checkpoint_step)] = {}
        for group, indices in selected_indices.items():
            result, values = critic_metrics(
                agent, recent.obs[indices], recent.action[indices], recent.next_obs[indices],
                recent.raw[indices], recent.done[indices], reward_config, gamma,
                float(checkpoint_progress["lambda_self"]),
            )
            critic[str(checkpoint_step)][group] = result
            fixed_q[str(checkpoint_step)][group] = values
    for group in centers:
        for earlier, later in (("800000", "900000"), ("900000", "1000000")):
            x, y = fixed_q[earlier][group], fixed_q[later][group]
            critic[later][group][f"q_churn_from_{earlier}"] = (
                describe(np.abs(y - x)) if x is not None else None)
    output = {
        "protocol": {"checkpoint": str(checkpoint), "checkpoint_step": 1000000,
                     "lambda_self_at_checkpoint": lambda_self,
                     "reuse_centers": str(args.reuse_centers) if args.reuse_centers else None,
                     "bad_state_rules": {"low_alignment": "first five-step contiguous low-alignment window in [0.6,1.0], first negative action",
                                         "stagnation": "first five successive band steps mean orientation progress <=0.05 rad/s, window start",
                                         "post_crossing_rebound": "first pre-action rho_R >=0.6 following first rho_R<0.6 crossing"},
                     "good_state_rule": "first successful O7--O9 descending pre-crossing pre-action state in [0.6,0.8]",
                     "neighborhood": THRESHOLDS, "feature_slices": {k: [v.start, v.stop] for k, v in FEATURES.items()},
                    "matching": "union over centers; rho_R/p/step calipers then RMS normalized q/qdot/clipped J_KP calipers",
                     "counting": "partition counts are not deduplicated across recent, success, and semantic copies; the same transition may appear in multiple groups",
                     "sampling": "20 current-replay batches of 1024, read-only reconstruction; NOT historic logged per-transition sampling",
                     "limitations": "training CSV lacks q, qdot, J_KP, actor alignment, and per-draw IDs; historic local-state entry/sampling rates and exact EE angular alignment are unidentifiable"},
        "episode_counts": episode_counts,
        "centers": {name: {"states": len(items), "episodes": len({r["episode"] for r in items}),
                           "individual_states": [
                               {"episode": int(r["episode"]), "step": int(r["step"]),
                                "rho_R": float(r["rho_orientation"]),
                                "rho_p": float(r["rho_position"]),
                                "remaining_time_fraction": float(r["remaining_time_fraction"]),
                                "q_normalized": r["observation"][FEATURES["q"]].tolist(),
                                "qdot_normalized": r["observation"][FEATURES["qdot"]].tolist(),
                                "J_KP_normalized_clipped": r["observation"][FEATURES["j_kp"]].reshape(9, 6).tolist(),
                                "alignment": float(r["actor_command_alignment"]),
                                "orientation_progress_rad_s": float(r["orientation_progress_rad_s"]),
                                "position_progress_m_s": float(r["position_progress_m_s"]),
                                "reward": float(r["reward"])} for r in items
                           ],
                           "rho_R": describe(r["rho_orientation"] for r in items),
                           "rho_p": describe(r["rho_position"] for r in items),
                           "step": describe(r["step"] for r in items),
                           "remaining_time": describe(r["remaining_time_fraction"] for r in items),
                           "mean_q": np.mean([r["observation"][FEATURES["q"]] for r in items], axis=0).tolist() if items else None,
                           "mean_qdot": np.mean([r["observation"][FEATURES["qdot"]] for r in items], axis=0).tolist() if items else None,
                           "mean_j_kp": np.mean([r["observation"][FEATURES["j_kp"]] for r in items], axis=0).tolist() if items else None}
                    for name, items in centers.items()},
        "coverage": coverage,
        "sampling": {"batches": batches, "source_counts": dict(sample_sources),
                     "full_local_by_source": {k: dict(v) for k, v in sample_by_source.items()},
                     "local_counts": {k: dict(v) for k, v in sample_counts.items()},
                     "batch_fractions": {k: {kind: v.get(kind, 0) / (1024 * batches)
                                             for kind in ("pose_time", "full_local")}
                                         for k, v in sample_counts.items()},
                     "mix": {"previous": prev_frac, "recent_including_semantic": recent_frac}},
        "critic_recent_full_local": critic,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"centers": {k: len(v) for k, v in centers.items()},
                      "local_recent": {k: coverage[k]["recent"]["full_local"]["count"] for k in centers},
                      "batch_fractions": output["sampling"]["batch_fractions"]}, indent=2))


if __name__ == "__main__":
    main()
