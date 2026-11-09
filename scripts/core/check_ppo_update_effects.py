#!/usr/bin/env python3
"""Paired numerical ablation on frozen, real S0 rollouts; never edits checkpoints.

This diagnoses update dynamics, NOT long-horizon learning or evaluation success.
All variants retain numerical guards, the same 10 epochs and the same KL gate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
import torch
from torch.distributions import Normal, kl_divergence

from rl_risk_sac.algorithms.thesis_ppo import PPONumericalError, ThesisPPOAgent, _cpu_copy
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool
from rl_risk_sac.tasks.reference_reaching import REFERENCE_PPO_OBSERVATION_DIM
from rl_risk_sac.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("ppo_effects_trainer", ROOT / "scripts/core/train_thesis_ppo.py")
TRAINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINER)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def flat_parameters(parameters):
    return torch.cat([p.detach().reshape(-1) for p in parameters])


def cosine(left, right):
    denominator = left.norm() * right.norm()
    return float(torch.dot(left, right) / denominator) if float(denominator) > 0 else None


def collect(config, agent, count, env_count, seed, episode_fraction):
    """Mirror trainer GAE, terminal masks and conditional reference sampling."""
    if count % env_count:
        raise ValueError("transitions must be divisible by environment count")
    keys = ("obs", "actions", "std_scale", "logp", "values", "rewards", "dones", "terminated", "next_obs")
    rollout = {key: [] for key in keys}
    episodes, successes = 0, 0
    with ParallelThesisEnvPool(config, [seed + 9973 * i for i in range(env_count)]) as pool:
        reset = pool.reset_many({i: TRAINER.make_contract("s0", "none") for i in range(env_count)},
                               seeds={i: seed + 9973 * i for i in range(env_count)})
        obs = np.stack([reset[i][0] for i in range(env_count)])
        for t in range(count // env_count):
            actions, logp, values, policy = agent.act(obs, episode_fraction=episode_fraction)
            results = pool.step_many({i: actions[i] for i in range(env_count)})
            raw_next = np.stack([results[i][0] for i in range(env_count)])
            next_obs = raw_next.copy()
            rewards = np.array([results[i][1] for i in range(env_count)], dtype=np.float32)
            terminated = np.array([results[i][3] for i in range(env_count)], dtype=np.float32)
            dones = np.array([results[i][3] or results[i][4] for i in range(env_count)], dtype=np.float32)
            for i in range(env_count):
                if dones[i]:
                    episodes += 1
                    successes += int(results[i][5].get("success", False))
                    next_obs[i] = pool.reset_many({i: TRAINER.make_contract("s0", "none")})[i][0]
            data = (obs.copy(), policy["raw_actions"], policy["std_scale"], logp, values,
                    rewards, dones, terminated, raw_next)
            for key, value in zip(keys, data):
                rollout[key].append(value)
            obs = next_obs
            if (t + 1) * env_count % 1024 == 0:
                print(f"  collected {(t+1)*env_count}/{count}", flush=True)
    arrays = {key: np.asarray(value, dtype=np.float32) for key, value in rollout.items()}
    next_values = agent.predict_values(arrays["next_obs"].reshape(-1, agent.obs_dim)).reshape(-1, env_count)
    mean_logp = arrays["logp"] / agent.action_dim
    gammas = np.empty_like(mean_logp)
    positive = mean_logp > 0
    gammas[positive] = np.tanh(.1 * mean_logp[positive])
    gammas[~positive] = np.exp(mean_logp[~positive])
    gammas = np.clip(gammas, .6, .99)
    adv, targets = TRAINER.compute_gae(arrays["rewards"], arrays["values"], next_values,
                                       arrays["terminated"], arrays["dones"], gammas, agent.gae_lambda)
    batch = dict(obs=arrays["obs"].reshape(-1, agent.obs_dim),
                 actions=arrays["actions"].reshape(-1, agent.action_dim),
                 std_scale=arrays["std_scale"].reshape(-1, 1), logp=arrays["logp"].reshape(-1),
                 advantages=adv.reshape(-1), returns=targets.reshape(-1))
    return batch, dict(transitions=count, num_envs=env_count, episodes=episodes, successes=successes,
                       gamma_mean=float(gammas.mean()), mean_reward=float(arrays["rewards"].mean()),
                       episode_fraction=episode_fraction)


def initial_gradient_probe(agent, batch, shuffle_seed):
    """Compare directions on exactly the first minibatch of the paired update."""
    seed_all(shuffle_seed)
    idx = torch.randperm(len(batch["obs"]), device=agent.device)[:agent.minibatch_size]
    obs = torch.as_tensor(batch["obs"], device=agent.device)[idx]
    actions = torch.as_tensor(batch["actions"], device=agent.device)[idx]
    scale = torch.as_tensor(batch["std_scale"], device=agent.device)[idx]
    old = torch.as_tensor(batch["logp"], device=agent.device)[idx]
    advantages = torch.as_tensor(batch["advantages"], device=agent.device)
    normalized = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    parameters = agent._actor_parameters()

    def gradient(loss):
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        return torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).reshape(-1)
                          for g, p in zip(gradients, parameters)])

    gradients = []
    for a in (advantages[idx], normalized[idx]):
        ratio = (agent._log_prob(obs, actions, scale) - old).exp()
        surrogate = torch.minimum(ratio*a, ratio.clamp(1-agent.clip_ratio, 1+agent.clip_ratio)*a)
        dist = agent._dist(obs)
        entropy = Normal(dist.mean, dist.stddev * scale).entropy().sum(-1).mean()
        gradients.append(gradient(-surrogate.mean() - agent.entropy_coef * entropy))
    dist = agent._dist(obs)
    entropy_gradient = gradient(-agent.entropy_coef * Normal(dist.mean, dist.stddev*scale).entropy().sum(-1).mean())
    return dict(advantage_mean=float(advantages.mean()), advantage_std=float(advantages.std(unbiased=False)),
                normalized_mean=float(normalized.mean()), normalized_std=float(normalized.std(unbiased=False)),
                advantage_sign_flip_fraction=float(((advantages > 0) != (normalized > 0)).float().mean()),
                actor_gradient_norm_raw=float(gradients[0].norm()),
                actor_gradient_norm_normalized=float(gradients[1].norm()),
                actor_gradient_cosine=cosine(*gradients),
                entropy_gradient_norm=float(entropy_gradient.norm()))


@torch.no_grad()
def distribution_and_value(agent, batch):
    obs = torch.as_tensor(batch["obs"], device=agent.device)
    scale = torch.as_tensor(batch["std_scale"], device=agent.device)
    dist = agent._dist(obs)
    return (
        dist.mean.clone(),
        (dist.stddev * scale).clone(),
        agent._value_output(obs).squeeze(-1).clone(),
    )


def paired_update(config, state, batch, normalize, clip, shuffle_seed, directory):
    agent = ThesisPPOAgent(state["obs_dim"], 6, config)
    # Deepcopy prevents Adam.load_state_dict() from aliasing the shared CPU baseline.
    agent.load_state_dict(deepcopy(state))
    agent.normalize_advantage = normalize
    # Diagnostic object only: +inf is exact no-clipping, retaining nonfinite checks.
    # No training implementation or YAML is altered by this ablation.
    agent.max_grad_norm = clip if clip is not None else float("inf")
    seed_all(shuffle_seed)
    before_mean, before_std, before_values = distribution_and_value(agent, batch)
    actors = agent._actor_parameters()
    critics = list(agent.value.parameters()) + list(
        agent.value_obstacle_encoder.parameters()
    )
    before_actor = flat_parameters(actors).clone()
    before_critic = flat_parameters(critics).clone()
    gradient_records = {"actor": [], "critic": []}
    delta_records = {"actor": [], "critic": []}
    original_clip = torch.nn.utils.clip_grad_norm_
    actor_ids = {id(p) for p in actors}

    def observed_clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        norm = original_clip(parameters, *args, **kwargs)
        post = torch.stack([p.grad.detach().square().sum() for p in parameters if p.grad is not None]).sum().sqrt()
        which = "actor" if id(parameters[0]) in actor_ids else "critic"
        gradient_records[which].append(dict(raw_norm=float(norm), post_norm=float(post),
                                            retained_fraction=float(post/(norm+1e-30))))
        return norm

    def observe_step(optimizer, parameters, which):
        original = optimizer.step

        def step(*args, **kwargs):
            before = flat_parameters(parameters).clone()
            result = original(*args, **kwargs)
            delta_records[which].append(float((flat_parameters(parameters)-before).norm()))
            return result
        return step

    error = None
    with patch("torch.nn.utils.clip_grad_norm_", side_effect=observed_clip), \
            patch.object(agent.actor_optimizer, "step", side_effect=observe_step(agent.actor_optimizer, actors, "actor")), \
            patch.object(agent.value_optimizer, "step", side_effect=observe_step(agent.value_optimizer, critics, "critic")):
        try:
            # progress=None fixes lr across all variants (no artificial final-step lr=0).
            metrics = agent.update(batch, diagnostic_dir=directory,
                                   diagnostic_context=dict(normalize=normalize, clip=clip, shuffle_seed=shuffle_seed))
        except PPONumericalError as exc:
            metrics = {}
            error = str(exc)
    after_mean, after_std, after_values = distribution_and_value(agent, batch)
    targets = torch.as_tensor(batch["returns"], device=agent.device)
    result = dict(normalize=normalize, clip=clip, shuffle_seed=shuffle_seed, error=error, metrics=metrics,
                  exact_old_new_kl=float(kl_divergence(Normal(before_mean, before_std), Normal(after_mean, after_std)).sum(-1).mean()),
                  policy_mean_change_rms_rad=float((after_mean-before_mean).square().mean().sqrt()),
                  actor_parameter_delta_l2=float((flat_parameters(actors)-before_actor).norm()),
                  critic_parameter_delta_l2=float((flat_parameters(critics)-before_critic).norm()),
                  critic_mse_before=float((before_values-targets).square().mean()),
                  critic_mse_after=float((after_values-targets).square().mean()),
                  gradients=gradient_records, optimizer_step_delta_l2=delta_records)
    agent.validate_training_state()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/paper_full.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reuse-batches-from", help="Reuse compatible frozen batches from a previous diagnostic run")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--transitions", type=int, default=4096)
    parser.add_argument("--shuffle-seeds", type=int, nargs="+", default=[21001, 21002, 21003])
    parser.add_argument("--cases", nargs="+", default=["reference_initial", "reference_narrow_std_stress"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    base_config = load_config(ROOT / args.config)
    results = {"method": "frozen real rollouts; same parameters/Adam state/data/shuffle within each paired comparison",
               "limits": "local numerical update effects only, not convergence or held-out task success",
               "device": args.device, "epochs": 10, "minibatch_size": 1024, "target_kl": .02, "cases": {}}
    variants = [(False, None), (True, None), (False, .5), (True, .5), (True, 5.), (True, 50.)]
    for case in args.cases:
        if case not in {"reference_initial", "reference_narrow_std_stress"}:
            raise ValueError(f"unknown case: {case}")
        seed_all(11001)
        config = deepcopy(base_config)
        config["device"] = args.device
        config["thesis"].update(simple_reaching_mode=True, blocking_obstacle_mode=False, stage="s0")
        config["thesis"]["reward"]["reward_mode"] = "reference_ppo"
        config["ppo"].update(epochs=10, minibatch_size=1024, target_kl=.02)
        agent = ThesisPPOAgent(REFERENCE_PPO_OBSERVATION_DIM, 6, config)
        source = "fresh initialization"
        if case == "reference_narrow_std_stress":
            source = "fresh mean/value weights with std forced to .035 rad; stress test, NOT a trained reference checkpoint"
            with torch.no_grad():
                agent.actor_log_std.fill_(np.log(.035))
        # This is an update-sensitivity comparison at the same fixed nonzero lr.
        for optimizer, lr in ((agent.actor_optimizer, agent.actor_lr), (agent.value_optimizer, agent.critic_lr)):
            for group in optimizer.param_groups:
                group["lr"] = lr
        state = _cpu_copy(agent.state_dict())
        reused = Path(args.reuse_batches_from) / case / "frozen_batch.pt" if args.reuse_batches_from else None
        if reused is not None and reused.is_file():
            saved_batch = torch.load(reused, map_location="cpu", weights_only=False)
            if saved_batch["collection"]["transitions"] != args.transitions:
                raise ValueError("reused batch transition count mismatch")
            config = saved_batch["config"]
            config["device"] = args.device
            state, batch, collection, source = (saved_batch[k] for k in ("state", "batch", "collection", "source"))
            agent = ThesisPPOAgent(state["obs_dim"], 6, config)
            agent.load_state_dict(deepcopy(state))
            print(f"CASE {case}: reusing {reused}", flush=True)
        else:
            print(f"CASE {case}: collecting frozen real rollout", flush=True)
            batch, collection = collect(config, agent, args.transitions, 1, 31001,
                                        .5 if case == "reference_narrow_std_stress" else 0.)
        case_dir = output / case
        case_dir.mkdir()
        torch.save(dict(config=config, state=state, batch=batch, collection=collection, source=source), case_dir / "frozen_batch.pt")
        case_result = dict(source=source, collection=collection,
                           action_std=state["actor_log_std"].clamp(-5, 1).exp().tolist(),
                           initial_gradient_probe=initial_gradient_probe(agent, batch, args.shuffle_seeds[0]), updates=[])
        for shuffle_seed in args.shuffle_seeds:
            for normalize, clip in variants:
                result = paired_update(config, state, batch, normalize, clip, shuffle_seed, case_dir / "diagnostics")
                case_result["updates"].append(result)
                print(f"  seed={shuffle_seed} norm={normalize} clip={clip} "
                      f"steps={result['metrics'].get('optimizer_steps')} KL={result['exact_old_new_kl']:.5g} "
                      f"critic_mse={result['critic_mse_after']:.5g} error={result['error'] is not None}", flush=True)
        results["cases"][case] = case_result
        (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Results: {output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
