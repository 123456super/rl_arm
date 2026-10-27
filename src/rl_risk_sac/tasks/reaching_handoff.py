"""S0 recovery from real handoffs, with obstacles removed only for recovery training."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import torch

from rl_risk_sac.algorithms.replay_buffer import Batch
from rl_risk_sac.tasks.local_avoidance import actor_digest, MeshClearance

BANK_PROTOCOL = "reaching_handoff_starts_v1"
TRAIN_PROTOCOL = "reaching_handoff_finetune_v1"


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_digest(state, scene_id, handoff_index):
    return hashlib.sha256(json.dumps(dict(scene_id=scene_id, handoff_index=handoff_index,
        q=np.asarray(state["q"]).tolist(), qdot=np.asarray(state["qdot"]).tolist(),
        goal=np.asarray(state["goal_position"]).tolist(), physics_steps=state["physics_step_count"]),
        sort_keys=True).encode()).hexdigest()


def recovery_state(snapshot):
    """Preserve q/qdot/goal and physical clock; remove obstacles and reset only episode accounting."""
    state = copy.deepcopy(snapshot)
    state["obstacles"], state["predictor_history"] = [], {}
    state["step_count"] = state["success_hold_count"] = 0
    state["obstacle_contact_seen"] = False
    state["contract"].scene = "none"
    state["contract"].terminate_on_success = True
    state["contract"].strict_obstacle_collision = True
    return state


def recovery_config(source, *, actor_lr=1e-5, critic_lr=3e-5, alpha_lr=1e-5):
    config = copy.deepcopy(source)
    config["local_avoidance"]["enabled"] = False
    config["dual_sac"]["enabled"] = False
    config["thesis"]["stage"] = "s0"
    config["thesis"]["obstacles"].update(enabled=False, dynamic=False,
                                           count_probabilities=[1., 0., 0., 0.])
    config["thesis"]["obstacles"].setdefault("path_sampling", {})["enabled"] = False
    config["sac"].update(actor_lr=actor_lr, critic_lr=critic_lr, alpha_lr=alpha_lr)
    return config


def validate_handoff_bank(bank, *, split=None, source_sha=None):
    if bank.get("protocol") != BANK_PROTOCOL or not bank.get("samples"):
        raise ValueError("requires a nonempty handoff-state bank")
    if split is not None and bank.get("split") not in split:
        raise ValueError("handoff split cannot be used for this training/evaluation")
    if source_sha is not None and bank["source_checkpoint_sha256"] != source_sha:
        raise ValueError("handoff bank came from a different joint checkpoint")
    ids = set()
    excluded = set(bank["excluded_trace_seeds"])
    for sample in bank["samples"]:
        state = sample["state"]
        if (sample["source_seed"] in excluded or sample["sample_id"] in ids
                or sample["sample_id"] != sample_digest(state, sample["source_scene_id"], sample["handoff_index"])
                or state["obstacle_contact_seen"] or sample["clearance_m"] < .06
                or sample["self_clearance_m"] < .005):
            raise ValueError("invalid, overlapping or unsafe handoff sample")
        for key in ("q", "qdot", "goal_position", "goal_quaternion"):
            if not np.isfinite(state[key]).all():
                raise ValueError("nonfinite handoff state")
        ids.add(sample["sample_id"])
    return ids


def capture_handoff(env, controller, source_scene):
    state = copy.deepcopy(env.episode_state_dict())
    margins = MeshClearance(env)
    sample = dict(state=state, source_seed=int(source_scene["provenance"]["seed"]),
                  source_scene_id=source_scene["sample_id"], handoff_index=controller.passages,
                  clearance_m=margins.external, self_clearance_m=margins.self_distance)
    sample["sample_id"] = sample_digest(state, sample["source_scene_id"], sample["handoff_index"])
    return sample


def bank_fingerprint(bank):
    return hashlib.sha256(json.dumps(sorted(s["sample_id"] for s in bank["samples"])).encode()).hexdigest()


class RecoveryReplay:
    """Fresh S0 rewards, no obstacle sets and no local-passage transitions."""
    def __init__(self, capacity, device, seed):
        self.capacity, self.device = capacity, device
        self.ptr = self.size = 0
        self.rng = np.random.default_rng(seed)
        self.observations = np.zeros((capacity, 241), np.float32)
        self.next_observations = np.zeros_like(self.observations)
        self.actions = np.zeros((capacity, 6), np.float32)
        self.rewards = np.zeros((capacity, 1), np.float32)
        self.dones = np.zeros_like(self.rewards)

    def add(self, observation, action, reward, following, terminated):
        if np.shape(observation) != (241,) or np.shape(following) != (241,):
            raise ValueError("recovery replay requires the unchanged 241D reaching schema")
        for value in (observation, action, reward, following):
            if not np.isfinite(value).all():
                raise ValueError("nonfinite recovery transition")
        i = self.ptr
        self.observations[i], self.next_observations[i] = observation, following
        self.actions[i], self.rewards[i], self.dones[i] = action, reward, terminated
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, count):
        indices = self.rng.integers(self.size, size=count)
        tensors = [torch.as_tensor(getattr(self, key)[indices], device=self.device)
                   for key in ("observations", "actions", "rewards", "next_observations", "dones")]
        observation, action, reward, following, done = tensors
        return Batch(observation, action, reward, torch.zeros_like(reward), following, done)

    def state_dict(self):
        return dict(protocol=TRAIN_PROTOCOL, capacity=self.capacity, ptr=self.ptr, size=self.size,
                    rng=self.rng.bit_generator.state,
                    **{key:getattr(self,key)[:self.size].copy() for key in (
                        "observations", "actions", "rewards", "next_observations", "dones")})

    def load_state_dict(self, state):
        if state.get("protocol") != TRAIN_PROTOCOL or state["capacity"] != self.capacity:
            raise ValueError("recovery replay protocol/capacity mismatch")
        self.ptr, self.size = state["ptr"], state["size"]
        for key in ("observations", "actions", "rewards", "next_observations", "dones"):
            getattr(self,key)[:self.size] = state[key]
        self.rng.bit_generator.state = state["rng"]


def load_reaching_override(agent, state, source_sha, evaluation_seeds):
    """Only swap S0 after validating the original scene-generating policy."""
    if state.get("protocol") != TRAIN_PROTOCOL or state["source_checkpoint_sha256"] != source_sha:
        raise ValueError("reaching override must be fine-tuned from this joint checkpoint")
    if set(evaluation_seeds).intersection(state["training_trace_seeds"]):
        raise ValueError("reaching fine-tuning traces overlap with evaluation scenes")
    if actor_digest(agent.avoidance.actor) != state["frozen_avoidance_actor_sha256"]:
        raise ValueError("avoidance actor differs from the frozen fine-tuning source")
    agent.reaching.load_state_dict(state["agent"], actor_only=True)
    agent.reaching.actor.eval().requires_grad_(False)
