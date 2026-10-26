"""Contracts and admission checks for immutable, training-only S1 experts."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expert_contract(config):
    """Conservatively bind data to physics, observation, reward and levels.

    Exclude sampling/optimizer/logging settings: experts intentionally have a
    different proposal distribution, but must represent the same transitions.
    """
    thesis = config["thesis"]
    if (thesis.get("stage") != "s1" or not thesis["obstacles"]["enabled"]
            or thesis["obstacles"]["dynamic"]):
        raise ValueError("expert replay supports single-static-obstacle S1 only")
    keys = (
        "protocol", "physics_dt", "control_dt", "horizon", "action_scale",
        "qdot_observation_scale", "precision_control", "success_hold_steps",
        "ee_linear_velocity_scale", "ee_angular_velocity_scale", "joint_motor_force",
        "d_safe", "self_collision", "self_safety_projection", "observation", "reward",
    )
    urdf = Path(config["robot"]["urdf"])
    assets = {str(urdf): file_sha256(urdf)}
    for node in ET.parse(urdf).findall(".//collision/geometry/mesh"):
        mesh = urdf.parent / node.attrib["filename"]
        assets[str(mesh)] = file_sha256(mesh)
    predictor = thesis["obstacles"].get("predictor_checkpoint")
    if predictor:
        assets[str(predictor)] = file_sha256(predictor)
    return copy.deepcopy(dict(
        format_version=1, stage="s1", robot=config["robot"], environment=config["env"],
        thesis={key: thesis.get(key) for key in keys},
        levels=thesis["joint_pose_curriculum"]["levels"],
        orientation_scales=thesis["orientation_curriculum"]["levels"],
        external_risk=thesis["obstacles"]["risk"],
        lambda_ext=thesis["obstacles"]["lambda_ext"],
        reward_gamma=config["sac"]["gamma"], assets=assets,
    ))


def contract_sha256(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def candidate_rejection(candidate, baselines, min_advantage):
    """Admission at both safety-weight endpoints implies it at every mixture."""
    if not np.isfinite(min_advantage) or min_advantage < 0 or not baselines:
        raise ValueError("admission requires baselines and a finite nonnegative advantage")
    if not all(np.isfinite(row[key]) for row in [candidate, *baselines]
               for key in ("discounted_return", "discounted_self_penalty")):
        return "nonfinite_return"
    if candidate["cause"] != "success":
        return candidate["cause"]
    if not candidate["physical_substep_margin_passed"]:
        return "safety_margin"
    for weight in (.2, 1.):
        def value(row):
            return row["discounted_return"] + (1 - weight) * row["discounted_self_penalty"]
        if value(candidate) < max(value(row) for row in baselines) + min_advantage:
            return "return_advantage"
    return None


def validate_expert_bank(bank, expected_contract=None):
    if bank.get("format_version") != 1 or bank.get("stage") != "s1":
        raise ValueError("unsupported S1 expert bank format/stage")
    if not bank.get("complete"):
        raise ValueError("S1 expert generation did not meet its requested episode quota")
    contract = bank["contract"]
    if bank.get("contract_sha256") != contract_sha256(contract):
        raise ValueError("S1 expert contract checksum mismatch")
    if expected_contract is not None and contract != expected_contract:
        raise ValueError("S1 expert bank physics/reward/observation contract mismatch; regenerate the bank")
    if not bank.get("partitions") or not bank.get("episodes"):
        raise ValueError("empty S1 expert bank")
    episodes = {int(row["episode_id"]): row for row in bank["episodes"]}
    if len(episodes) != len(bank["episodes"]):
        raise ValueError("duplicate expert episode ids")
    seen = set()
    for level, part in bank["partitions"].items():
        if not isinstance(level, int) or not 0 <= level < len(contract["levels"]):
            raise ValueError("invalid expert precision level")
        if part.size != part.capacity or part.size < 1:
            raise ValueError("expert partitions must be compact, immutable complete datasets")
        if part.obs.shape[1] != 241 or part.action.shape[1] != 6 or part.raw.shape[1] != 29:
            raise ValueError("expert observation/action/raw dimensions mismatch")
        for field in ("obs", "next_obs", "action", "raw", "done"):
            if not np.isfinite(getattr(part, field)[:part.size]).all():
                raise ValueError(f"nonfinite expert {field}")
        if np.any(np.abs(part.action[:part.size]) > 1):
            raise ValueError("expert actions must be normalized policy actions, not joint velocities")
        for episode, indices in part.episode_rows().items():
            if episode not in episodes or episode in seen:
                raise ValueError("expert episode metadata/level mismatch")
            seen.add(episode)
            meta = episodes[episode]
            if meta["level"] != level or not meta["physical_substep_margin_passed"]:
                raise ValueError("expert precision or safety validation failed")
            if (not meta.get("fresh_training_scene") or meta["cause"] != "success"
                    or meta["min_advantage"] < 0):
                raise ValueError("expert admission metadata failed")
            safe = contract["thesis"]["d_safe"]
            self_safe = contract["thesis"]["self_collision"]["safe_distance_m"]
            substeps = round(contract["thesis"]["control_dt"] / contract["thesis"]["physics_dt"])
            if (not np.isfinite(meta["min_external_mesh_distance_m"])
                    or not np.isfinite(meta["min_self_mesh_distance_m"])
                    or meta["min_external_mesh_distance_m"] < safe
                    or meta["min_self_mesh_distance_m"] < self_safe
                    or meta["physics_substep_checks"] != meta["steps"] * substeps + 1):
                raise ValueError("expert lacks complete physical-substep safety checks")
            if candidate_rejection(meta, [meta["reference"], meta["waiting"]], meta["min_advantage"]) is not None:
                raise ValueError("expert return admission failed")
            ordered = sorted(indices, key=lambda i: int(part.step[i]))
            raw = part.raw[ordered]
            tolerance = contract["levels"][level]
            if (not np.all(raw[:, 23] == level)
                    or not np.allclose(raw[:, 16], contract["orientation_scales"][level], rtol=0, atol=1e-7)
                    or not np.allclose(raw[:, 24], tolerance["position_tolerance_m"], rtol=0, atol=1e-7)
                    or not np.allclose(raw[:, 25], tolerance["orientation_tolerance_rad"], rtol=0, atol=1e-7)):
                raise ValueError("expert transition precision contract mismatch")
            if (len(ordered) != meta["steps"] or len(ordered) > contract["thesis"]["horizon"]
                    or not np.array_equal(part.step[ordered], np.arange(1, len(ordered) + 1))):
                raise ValueError("incomplete expert episode sequence")
            if (np.any(part.done[ordered[:-1]]) or part.done[ordered[-1], 0] != 1
                    or np.any(raw[:-1, 5]) or raw[-1, 5] != 1
                    or np.any(raw[:, 6:11]) or np.any(raw[:, 13:15]) or np.any(raw[:, 22])):
                raise ValueError("expert episode is not an intact collision-free terminal success")
            hold = contract["thesis"]["success_hold_steps"]
            if (len(ordered) < hold or np.any(raw[-hold:, 1] > tolerance["position_tolerance_m"])
                    or np.any(raw[-hold:, 3] > tolerance["orientation_tolerance_rad"])):
                raise ValueError("expert terminal success lacks the required strict pose hold")
            if any(len(part.obstacle_sets[i]) != 1 or len(part.next_obstacle_sets[i]) != 1
                   or not np.isfinite(part.obstacle_sets[i]).all()
                   or not np.isfinite(part.next_obstacle_sets[i]).all() for i in ordered):
                raise ValueError("S1 experts require one preserved finite raw obstacle set per state")
            if any(not np.array_equal(part.next_obs[a], part.obs[b])
                   or not np.array_equal(part.next_obstacle_sets[a], part.obstacle_sets[b])
                   for a, b in zip(ordered, ordered[1:])):
                raise ValueError("expert transition chain is discontinuous")
            from rl_risk_sac.algorithms.homotopy_replay import _vectorized_replay_rewards
            rewards, _ = _vectorized_replay_rewards(
                raw, part.done[ordered], np.ones(len(ordered)), 1., contract["reward_gamma"],
                bank["reward_parameters"],
                obstacle_sets=[part.obstacle_sets[i] for i in ordered],
                next_obstacle_sets=[part.next_obstacle_sets[i] for i in ordered],
            )
            value = float(np.dot(np.power(contract["reward_gamma"], np.arange(len(ordered))), rewards))
            if abs(value - meta["discounted_return"]) > 5e-4:
                raise ValueError("expert stored return disagrees with recomputed replay rewards")
            rewards_without_self, _ = _vectorized_replay_rewards(
                raw, part.done[ordered], np.ones(len(ordered)), 0., contract["reward_gamma"],
                bank["reward_parameters"],
                obstacle_sets=[part.obstacle_sets[i] for i in ordered],
                next_obstacle_sets=[part.next_obstacle_sets[i] for i in ordered],
            )
            self_cost = float(np.dot(np.power(contract["reward_gamma"], np.arange(len(ordered))),
                                     rewards_without_self - rewards))
            if abs(self_cost - meta["discounted_self_penalty"]) > 5e-4:
                raise ValueError("expert stored self penalty disagrees with replay rewards")
    if seen != set(episodes):
        raise ValueError("orphan expert episode metadata")


def relabel_coupling_bank(bank, config, reward_parameters):
    """Copy a validated bank; only an exact, terminal-zero PBRS change is allowed.

    Physics, raw transitions and physical admission are unchanged. Every complete
    same-start baseline has the same return shift. No safety check is bypassed.
    """
    from rl_risk_sac.tasks.goal_obstacle_coupling import (
        COUPLING_KEYS, coupling_potential, obstacle_distances,
    )
    from rl_risk_sac.algorithms.homotopy_replay import _vectorized_replay_rewards
    validate_expert_bank(bank)
    target = expert_contract(config)
    old_contract, new_contract = copy.deepcopy(bank["contract"]), copy.deepcopy(target)
    for contract in (old_contract, new_contract):
        contract["thesis"]["reward"].pop("goal_obstacle_coupling", None)
    if old_contract != new_contract:
        raise ValueError("bank relabel permits only goal-obstacle coupling; physics/base reward differs")
    old_parameters = {k: v for k, v in bank["reward_parameters"].items() if k not in COUPLING_KEYS}
    new_parameters = {k: v for k, v in reward_parameters.items() if k not in COUPLING_KEYS}
    if old_parameters != new_parameters:
        raise ValueError("bank relabel cannot change base reward parameters")
    result = copy.deepcopy(bank)
    result.update(contract=target, contract_sha256=contract_sha256(target),
                  reward_parameters=dict(reward_parameters))
    shifts, max_residual = {}, 0.
    by_id = {row["episode_id"]: row for row in result["episodes"]}
    for part in result["partitions"].values():
        for episode, indices in part.episode_rows().items():
            ordered = sorted(indices, key=lambda i: int(part.step[i]))
            raw = part.raw[ordered]
            sets = [part.obstacle_sets[i] for i in ordered]
            next_sets = [part.next_obstacle_sets[i] for i in ordered]
            distance, present = obstacle_distances(sets[:1])
            def initial(parameters):
                if not parameters.get(COUPLING_KEYS[0], 0.):
                    return 0.
                return float(coupling_potential(raw[0, 0], raw[0, 2], distance[0], present[0], parameters))
            shift = initial(reward_parameters) - initial(bank["reward_parameters"])
            old, _ = _vectorized_replay_rewards(
                raw, part.done[ordered], np.ones(len(raw)), 1., target["reward_gamma"],
                bank["reward_parameters"], obstacle_sets=sets, next_obstacle_sets=next_sets)
            new, _ = _vectorized_replay_rewards(
                raw, part.done[ordered], np.ones(len(raw)), 1., target["reward_gamma"],
                reward_parameters, obstacle_sets=sets, next_obstacle_sets=next_sets)
            discount = target["reward_gamma"] ** np.arange(len(raw))
            residual = float(np.dot(discount, new - old) + shift)
            max_residual = max(max_residual, abs(residual))
            if abs(residual) > 2e-6:
                raise ValueError("expert chain does not telescope under coupling relabel")
            meta = by_id[episode]
            if "candidates" in meta:
                # Rejected candidates can be partial. Do not falsely apply an
                # absorbing-terminal shift to their historical diagnostics.
                meta["source_reward_candidate_diagnostics"] = meta.pop("candidates")
            for row in (meta, meta["reference"], meta["waiting"]):
                row["discounted_return"] -= shift
            shifts[str(episode)] = shift
    result["coupling_relabel"] = dict(
        source_contract_sha256=bank["contract_sha256"], initial_potential_delta=shifts,
        max_telescoping_residual=max_residual,
        safety_checks_reused="validated immutable physical trajectories; no new safety claim",
        candidate_diagnostic_reward="source_reward_candidate_diagnostics retain source rewards",
    )
    validate_expert_bank(result, target)
    return result
