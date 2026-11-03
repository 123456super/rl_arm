from __future__ import annotations

from pathlib import Path
import random
import tempfile
import traceback
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from rl_risk_sac.algorithms.networks import mlp
from rl_risk_sac.utils.device import resolve_device


def _cpu_copy(value: Any) -> Any:
    """Detached snapshot: state_dict() alone aliases live model tensors."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_copy(item) for item in value)
    return value


def _device_copy(value: Any) -> Any:
    """Detached snapshot that stays on-device for inexpensive step rollback."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _device_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_device_copy(item) for item in value)
    return value


def _require_finite(value: Any, name: str) -> None:
    """Check nested model/Adam state, with one GPU synchronization per device."""
    checks: dict[torch.device, list[tuple[str, torch.Tensor]]] = {}

    def visit(item, path):
        if isinstance(item, torch.Tensor):
            checks.setdefault(item.device, []).append((path, torch.isfinite(item).all()))
        elif isinstance(item, np.ndarray):
            if not np.isfinite(item).all():
                raise FloatingPointError(f"non-finite {path}")
        elif isinstance(item, (float, np.floating)) and not np.isfinite(item):
            raise FloatingPointError(f"non-finite {path}: {item}")
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{path}.{key}")
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, name)
    for group in checks.values():
        valid = torch.stack([check for _, check in group]).cpu().tolist()
        invalid = [path for (path, _), ok in zip(group, valid) if not ok]
        if invalid:
            raise FloatingPointError("non-finite " + ", ".join(invalid[:8]))


def _gradient_norm(parameters) -> torch.Tensor:
    """Measure the global L2 gradient norm without modifying gradients."""
    gradients = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return torch.zeros((), dtype=torch.float32)
    per_parameter = torch.stack([
        torch.linalg.vector_norm(gradient, ord=2) for gradient in gradients
    ])
    return torch.linalg.vector_norm(per_parameter, ord=2)


class PPONumericalError(RuntimeError):
    def __init__(self, message: str, diagnostic_path: Path, rolled_back: bool):
        self.diagnostic_path = diagnostic_path
        self.rolled_back = rolled_back
        super().__init__(f"{message}; rolled_back={rolled_back}; diagnostic={diagnostic_path}")


class ThesisPPOAgent:
    """On-policy PPO actor-critic used by the thesis reaching trainer.

    Actions are absolute joint-angle targets in radians. Keep latent Gaussian
    samples/log probabilities separate from the actuator-clipped commands.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: dict[str, Any]) -> None:
        reward_mode = str(
            config.get("thesis", {}).get("reward", {}).get(
                "reward_mode", "reference_ppo"
            )
        )
        if reward_mode != "reference_ppo":
            raise ValueError("ThesisPPOAgent supports only reference_ppo")
        self.reference_ppo = True
        self.reference_training_contract = True
        ppo = dict(config.get("ppo", {}))
        self.policy_contract = "ppo_absolute_raw_gaussian_v2"
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(resolve_device(config))
        hidden = [int(v) for v in ppo.get("hidden_dims", [256, 256])]
        self.gamma = float(ppo.get("gamma", 0.99))
        self.gae_lambda = float(ppo.get("gae_lambda", 0.95))
        self.clip_ratio = float(ppo.get("clip_ratio", 0.2))
        self.epochs = int(ppo.get("epochs", 10))
        self.minibatch_size = int(ppo.get("minibatch_size", 1024))
        self.actor_lr = float(ppo.get("actor_lr", 3e-4))
        self.critic_lr = float(ppo.get("critic_lr", 3e-4))
        self.actor_lr_final = float(ppo.get("actor_lr_final", 0.0))
        self.critic_lr_final = float(ppo.get("critic_lr_final", 0.0))
        self.entropy_coef = float(ppo.get("entropy_coef", 0.01))
        self.value_coef = float(ppo.get("value_coef", 1.0))
        self.normalize_advantage = bool(ppo.get("normalize_advantage", True))
        configured_max_grad_norm = ppo.get("max_grad_norm", 0.5)
        self.max_grad_norm = (
            None if configured_max_grad_norm is None
            else float(configured_max_grad_norm)
        )
        configured_target_kl = ppo.get("target_kl", 0.05)
        self.target_kl = (
            None if configured_target_kl is None else float(configured_target_kl)
        )
        self.max_abs_log_ratio = float(ppo.get("max_abs_log_ratio", 20.0))
        self.adaptive_kl_lr = bool(ppo.get("adaptive_kl_lr", False))
        self.kl_lr_decrease_factor = float(ppo.get("kl_lr_decrease_factor", 0.5))
        self.kl_lr_increase_factor = float(ppo.get("kl_lr_increase_factor", 1.05))
        self.kl_lr_low_threshold = float(ppo.get("kl_lr_low_threshold", 0.01))
        self.kl_lr_low_patience = int(ppo.get("kl_lr_low_patience", 10))
        self.actor_lr_scale = 1.0
        self.low_kl_streak = 0
        if self.target_kl is not None and (
            not np.isfinite(self.target_kl) or self.target_kl <= 0.0
        ):
            raise ValueError("target_kl must be null or finite and positive")
        if not np.isfinite(self.max_abs_log_ratio) or not 0.0 < self.max_abs_log_ratio <= 40.0:
            raise ValueError("max_abs_log_ratio must be in (0, 40] for float32 safety")
        if not (0.0 <= self.actor_lr_final <= self.actor_lr and 0.0 <= self.critic_lr_final <= self.critic_lr):
            raise ValueError("final learning rates must be in [0, initial learning rate]")
        adaptive_kl_valid = (
            self.target_kl is not None
            and 0.0 < self.kl_lr_decrease_factor < 1.0
            and self.kl_lr_increase_factor >= 1.0
            and 0.0 < self.kl_lr_low_threshold < 1.5 * self.target_kl
            and self.kl_lr_low_patience >= 1
        )
        if self.adaptive_kl_lr and not adaptive_kl_valid:
            raise ValueError("invalid adaptive KL learning-rate settings")
        if self.epochs < 1 or self.minibatch_size < 1:
            raise ValueError("PPO epochs and minibatch_size must be positive")
        if self.max_grad_norm is not None and (
            not np.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0.0
        ):
            raise ValueError("max_grad_norm must be null or finite and positive")
        self.action_std_init = float(ppo.get("action_std_init", 1.0))
        self.lr_decay_enabled = bool(ppo.get("lr_decay", True))
        action_limit = float(
            config.get("thesis", {}).get("action_limit_rad", np.pi)
        )
        if action_limit <= 0.0:
            raise ValueError("action_limit_rad must be positive")
        self.action_limit = torch.full(
            (action_dim,), action_limit, dtype=torch.float32, device=self.device
        )
        if len(hidden) != 2:
            raise ValueError("reference PPO requires exactly two hidden layers")
        self.actor = mlp(obs_dim, hidden, action_dim, activation="tanh", orthogonal_init=True, output_gain=0.01).to(self.device)
        # Explore around the configured bent home rather than the singular zero pose.
        home = config.get("thesis", {}).get("fixed_orientation_goal", {}).get(
            "initial_joint_positions_rad", [0.0] * action_dim
        )
        ratio = torch.as_tensor(home, dtype=torch.float32, device=self.device) / self.action_limit
        if ratio.shape != (action_dim,) or bool((ratio.abs() >= 1.0).any()):
            raise ValueError("fixed home must be strictly inside absolute action bounds")
        with torch.no_grad():
            self.actor[-1].bias.copy_(torch.atanh(ratio))
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), np.log(self.action_std_init), device=self.device))
        self.value = mlp(obs_dim, hidden, 1, activation="tanh", orthogonal_init=True).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + [self.actor_log_std],
            lr=self.actor_lr,
        )
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.critic_lr)

    def _dist(self, obs: torch.Tensor) -> Normal:
        mean = torch.tanh(self.actor(obs)) * self.action_limit
        log_std = self.actor_log_std.clamp(-5.0, 1.0).expand_as(mean)
        return Normal(mean, log_std.exp())

    def _log_prob(self, obs: torch.Tensor, action: torch.Tensor, std_scale: torch.Tensor | None = None) -> torch.Tensor:
        dist = self._dist(obs)
        if std_scale is not None:
            dist = Normal(dist.mean, dist.stddev * std_scale)
        return dist.log_prob(action).sum(-1)

    @torch.no_grad()
    def predict_values(self, observations: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        return self.value(obs).squeeze(-1).cpu().numpy()

    @torch.no_grad()
    def act(
        self,
        observations: np.ndarray,
        deterministic: bool = False,
        episode_fraction: float = 0.0,
    ):
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        dist = self._dist(obs)
        sample_count = 1
        if deterministic:
            action = dist.mean
        else:
            # Match the reference main experiment's Gaussian action ensemble:
            # the number of samples grows from 1 to 13 while the averaging
            # probability decreases from 0.5 to 0.
            fraction = float(np.clip(episode_fraction, 0.0, 1.0))
            n = max(1, int(1.0 + 12.0 * fraction))
            p = float(np.clip(0.5 - 0.5 * fraction, 0.0, 1.0))
            sample_count = int(np.clip(np.random.binomial(n, p), 1, n))
            action = dist.sample((sample_count,)).mean(dim=0)
        # An average of k independent Normal samples has std / sqrt(k).
        # Store this conditional scale with each transition for correct PPO
        # ratios in the reference ensemble branch as well.
        std_scale = float(sample_count ** -0.5)
        sampling_dist = Normal(dist.mean, dist.stddev * std_scale)
        log_prob = sampling_dist.log_prob(action).sum(-1)
        raw_action = action.cpu().numpy().astype(np.float32)
        command = action.clamp(-self.action_limit, self.action_limit)
        value = self.value(obs).squeeze(-1)
        return command.cpu().numpy().astype(np.float32), log_prob.cpu().numpy(), value.cpu().numpy(), {
            "raw_actions": raw_action,
            "std_scale": np.full((*raw_action.shape[:-1], 1), std_scale, dtype=np.float32),
        }

    def validate_training_state(self) -> None:
        """Validate weights, log std, Adam moments/step and optimizer settings."""
        _require_finite(self.state_dict(), "training_state")

    @staticmethod
    def _linear_learning_rate(initial: float, final: float, progress: float) -> float:
        fraction = float(np.clip(progress, 0.0, 1.0))
        return final + (initial - final) * (1.0 - fraction)

    def _apply_learning_rate_schedule(self, progress: float | None) -> tuple[float, float]:
        if self.lr_decay_enabled and progress is not None:
            scheduled_actor = self._linear_learning_rate(self.actor_lr, self.actor_lr_final, progress)
            scheduled_critic = self._linear_learning_rate(self.critic_lr, self.critic_lr_final, progress)
        else:
            scheduled_actor = self.actor_lr
            scheduled_critic = self.critic_lr
        effective_actor = max(self.actor_lr_final, scheduled_actor * self.actor_lr_scale)
        for group in self.actor_optimizer.param_groups:
            group["lr"] = effective_actor
        for group in self.value_optimizer.param_groups:
            group["lr"] = scheduled_critic
        return scheduled_actor, scheduled_critic

    def _decrease_actor_learning_rate(self, progress: float | None) -> None:
        self.actor_lr_scale *= self.kl_lr_decrease_factor
        self.low_kl_streak = 0
        self._apply_learning_rate_schedule(progress)

    def _update_low_kl_adaptation(
        self, *, rejected: bool, accepted_kls: list[float], progress: float | None,
    ) -> None:
        if not self.adaptive_kl_lr:
            return
        if rejected or not accepted_kls or max(accepted_kls) >= self.kl_lr_low_threshold:
            self.low_kl_streak = 0
            return
        self.low_kl_streak += 1
        if self.low_kl_streak >= self.kl_lr_low_patience:
            self.actor_lr_scale = min(1.0, self.actor_lr_scale * self.kl_lr_increase_factor)
            self.low_kl_streak = 0
            self._apply_learning_rate_schedule(progress)

    def update(
        self, batch: dict[str, np.ndarray], progress: float | None = None,
        *, diagnostic_dir: str | Path | None = None,
        diagnostic_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_state = None
        self._update_context = {"phase": "before_update", **(diagnostic_context or {})}
        self._diagnostic_tensors = {}
        rng_state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
        if self.device.type == "cuda":
            rng_state["cuda"] = torch.cuda.get_rng_state_all()
        try:
            self.validate_training_state()
            clean_state = _cpu_copy(self.state_dict())
            metrics = self._update_impl(batch, progress)
            self._update_context["phase"] = "after_update"
            self.validate_training_state()
            _require_finite(metrics, "metrics")
            return metrics
        except Exception as error:
            # Capture the offending state BEFORE rollback. Never sanitize, skip
            # this batch, or checkpoint corrupt weights as a healthy model.
            payload = {
                "error": repr(error), "traceback": traceback.format_exc(),
                "context": _cpu_copy(self._update_context), "batch": _cpu_copy(batch),
                "intermediates": _cpu_copy(self._diagnostic_tensors),
                "failed_state": _cpu_copy(self.state_dict()), "clean_state": clean_state,
                "gradients": _cpu_copy({
                    **{f"actor.{k}": p.grad for k, p in self.actor.named_parameters()},
                    "actor_log_std": self.actor_log_std.grad,
                    **{f"value.{k}": p.grad for k, p in self.value.named_parameters()},
                }), "rng_state": rng_state,
                "ppo_settings": {key: getattr(self, key) for key in (
                    "epochs", "minibatch_size", "target_kl", "max_abs_log_ratio",
                    "normalize_advantage", "max_grad_norm", "entropy_coef", "value_coef",
                )},
            }
            directory = Path(diagnostic_dir) if diagnostic_dir is not None else Path(tempfile.mkdtemp(prefix="ppo-diagnostic-"))
            path = directory / f"ppo_failure_{uuid4().hex}.pt"
            try:
                directory.mkdir(parents=True, exist_ok=True)
                torch.save(payload, path.with_suffix(".tmp"))
                path.with_suffix(".tmp").replace(path)
            except Exception as save_error:
                raise RuntimeError(f"PPO failed ({error}); diagnostic save also failed: {save_error}; training stopped") from error
            if clean_state is not None:
                try:
                    self.load_state_dict(clean_state)
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    self.value_optimizer.zero_grad(set_to_none=True)
                except Exception as rollback_error:
                    raise PPONumericalError(f"{error}; rollback also failed: {rollback_error}", path, False) from error
            raise PPONumericalError(str(error), path, clean_state is not None) from error
        finally:
            self._diagnostic_tensors = {}

    def _update_impl(self, batch: dict[str, np.ndarray], progress: float | None) -> dict[str, Any]:
        _require_finite(batch, "batch")
        _require_finite(progress, "progress")
        scheduled_actor_lr, scheduled_critic_lr = self._apply_learning_rate_schedule(progress)
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(batch["logp"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        std_scale = torch.as_tensor(batch["std_scale"], dtype=torch.float32, device=self.device)
        _require_finite((obs, actions, old_logp, advantages, returns, std_scale), "float32_batch")
        n = obs.shape[0]
        if (n < 1 or obs.shape != (n, self.obs_dim)
                or actions.shape != (n, self.action_dim) or std_scale.shape != (n, 1)
                or any(t.shape != (n,) for t in (old_logp, advantages, returns))):
            raise ValueError("invalid PPO batch shapes")
        if not bool((std_scale > 0.0).all()):
            raise ValueError("sampling std scales must be positive")
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        _require_finite(advantages, "normalized_advantages")
        actor_losses, value_losses, entropies, ratios = [], [], [], []
        attempted_kls, accepted_kls, clip_fractions = [], [], []
        actor_norms, value_norms = [], []
        actor_clips, value_clips = [], []
        actor_parameters = list(self.actor.parameters()) + [self.actor_log_std]
        early_stopped = False
        actor_epochs_completed = 0
        max_log_ratio = 0.0
        actor_attempts = 0
        rejected_actor_steps = 0
        kl_limit = None if self.target_kl is None else 1.5 * self.target_kl

        # A rollout must initially belong to the loaded policy. This also
        # catches stale or mismatched batches before any Actor mutation.
        self._update_context["phase"] = "before_ratio_exp"
        with torch.no_grad():
            initial_logp = self._log_prob(obs, actions, std_scale)
            initial_log_ratio = initial_logp - old_logp
            self._diagnostic_tensors = {
                "new_logp": initial_logp, "old_logp": old_logp,
                "log_ratio": initial_log_ratio,
            }
            _require_finite(self._diagnostic_tensors, "log_probabilities")
            initial_largest = float(initial_log_ratio.abs().max())
            max_log_ratio = max(max_log_ratio, initial_largest)
            if initial_largest > self.max_abs_log_ratio:
                raise FloatingPointError(
                    f"abnormal abs(logp_new-logp_old)={initial_largest:.6g} "
                    f"> {self.max_abs_log_ratio}"
                )
            initial_kl = float((torch.expm1(initial_log_ratio) - initial_log_ratio).mean())
        _require_finite(initial_kl, "initial_full_rollout_kl")
        if kl_limit is not None and initial_kl > kl_limit:
            attempted_kls.append(initial_kl)
            early_stopped = True

        # Actor loop. Every proposed step is checked on the complete rollout.
        # If it crosses the KL limit, restore both parameters and Adam state,
        # reduce only the Actor LR, and stop Actor work for this rollout.
        for epoch in range(self.epochs):
            if early_stopped:
                break
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, self.minibatch_size):
                idx = indices[start:start + self.minibatch_size]
                self._update_context.update(phase="before_actor_minibatch", epoch=epoch, minibatch_start=start, indices=idx)
                self.validate_training_state()
                logp = self._log_prob(obs[idx], actions[idx], std_scale[idx])
                log_ratio = logp - old_logp[idx]
                self._diagnostic_tensors = {"new_logp": logp, "old_logp": old_logp[idx], "log_ratio": log_ratio}
                # Crucial: exp(inf) can yield NaN gradients even if PPO's
                # clipped minimum happens to make the scalar loss finite.
                self._update_context["phase"] = "before_ratio_exp"
                _require_finite(self._diagnostic_tensors, "log_probabilities")
                largest = float(log_ratio.detach().abs().max())
                max_log_ratio = max(max_log_ratio, largest)
                if largest > self.max_abs_log_ratio:
                    raise FloatingPointError(f"abnormal abs(logp_new-logp_old)={largest:.6g} > {self.max_abs_log_ratio}")
                ratio = torch.exp(log_ratio)
                _require_finite(ratio, "probability_ratio")
                with torch.no_grad():
                    clip_fraction = float(((ratio - 1.0).abs() > self.clip_ratio).float().mean())
                clip_fractions.append(clip_fraction)
                ratios.append(float(ratio.detach().mean()))
                dist = self._dist(obs[idx])
                entropy = Normal(dist.mean, dist.stddev * std_scale[idx]).entropy().sum(-1).mean()
                _require_finite(entropy, "entropy")
                entropies.append(float(entropy.detach()))
                surrogate = torch.min(ratio * advantages[idx], torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages[idx])
                actor_loss = -surrogate.mean() - self.entropy_coef * entropy
                self._diagnostic_tensors.update(ratio=ratio, advantages=advantages[idx], actor_loss=actor_loss)
                self._update_context["phase"] = "actor_backward"
                _require_finite((surrogate, actor_loss), "actor_objective")
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                if self.max_grad_norm is None:
                    actor_norm = _gradient_norm(actor_parameters)
                else:
                    actor_norm = nn.utils.clip_grad_norm_(
                        actor_parameters, self.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                _require_finite(actor_norm, "actor_gradient_norm")
                _require_finite([p.grad for p in actor_parameters], "actor_gradients")
                self.validate_training_state()
                actor_snapshot = {
                    "actor": _device_copy(self.actor.state_dict()),
                    "actor_log_std": self.actor_log_std.detach().clone(),
                    "optimizer": _device_copy(self.actor_optimizer.state_dict()),
                }
                self._update_context["phase"] = "actor_optimizer_step"
                self.actor_optimizer.step()
                actor_attempts += 1
                self.validate_training_state()

                self._update_context["phase"] = "post_actor_full_rollout_kl"
                with torch.no_grad():
                    full_logp = self._log_prob(obs, actions, std_scale)
                    full_log_ratio = full_logp - old_logp
                    self._diagnostic_tensors.update(
                        full_new_logp=full_logp, full_old_logp=old_logp,
                        full_log_ratio=full_log_ratio,
                    )
                    _require_finite(full_log_ratio, "full_rollout_log_ratio")
                    full_largest = float(full_log_ratio.abs().max())
                    max_log_ratio = max(max_log_ratio, full_largest)
                    if full_largest > self.max_abs_log_ratio:
                        raise FloatingPointError(
                            f"abnormal abs(full_logp_new-full_logp_old)={full_largest:.6g} "
                            f"> {self.max_abs_log_ratio}"
                        )
                    full_kl = float((torch.expm1(full_log_ratio) - full_log_ratio).mean())
                _require_finite(full_kl, "full_rollout_kl")
                attempted_kls.append(full_kl)
                if kl_limit is not None and full_kl > kl_limit:
                    self.actor.load_state_dict(actor_snapshot["actor"])
                    self.actor_log_std.data.copy_(actor_snapshot["actor_log_std"])
                    self.actor_optimizer.load_state_dict(actor_snapshot["optimizer"])
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    rejected_actor_steps += 1
                    early_stopped = True
                    if self.adaptive_kl_lr:
                        self._decrease_actor_learning_rate(progress)
                    self.validate_training_state()
                    break

                accepted_kls.append(full_kl)
                actor_losses.append(float(actor_loss.detach().cpu()))
                actor_norms.append(float(actor_norm))
                actor_clips.append(float(
                    self.max_grad_norm is not None
                    and float(actor_norm) > self.max_grad_norm
                ))
            if early_stopped:
                break
            actor_epochs_completed += 1

        self._update_low_kl_adaptation(
            rejected=bool(rejected_actor_steps), accepted_kls=accepted_kls, progress=progress,
        )

        # Critic is independent of Actor KL. It consumes the complete rollout
        # for every configured epoch even when an Actor proposal was rejected.
        critic_epochs_completed = 0
        for epoch in range(self.epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, self.minibatch_size):
                idx = indices[start:start + self.minibatch_size]
                self._update_context.update(phase="critic_backward", epoch=epoch, minibatch_start=start, indices=idx)
                value_loss = (self.value(obs[idx]).squeeze(-1) - returns[idx]).square().mean()
                self._diagnostic_tensors = {"value_loss": value_loss, "returns": returns[idx]}
                _require_finite(value_loss, "value_loss")
                self.value_optimizer.zero_grad(set_to_none=True)
                (self.value_coef * value_loss).backward()
                value_parameters = list(self.value.parameters())
                if self.max_grad_norm is None:
                    value_norm = _gradient_norm(value_parameters)
                else:
                    value_norm = nn.utils.clip_grad_norm_(
                        value_parameters, self.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                _require_finite(value_norm, "critic_gradient_norm")
                _require_finite([p.grad for p in value_parameters], "critic_gradients")
                self.validate_training_state()
                self._update_context["phase"] = "critic_optimizer_step"
                self.value_optimizer.step()
                self.validate_training_state()
                value_losses.append(float(value_loss.detach().cpu()))
                value_norms.append(float(value_norm))
                value_clips.append(float(
                    self.max_grad_norm is not None
                    and float(value_norm) > self.max_grad_norm
                ))
            critic_epochs_completed += 1
        with torch.no_grad():
            predictions = self.value(obs).squeeze(-1)
            variance = returns.var(unbiased=False)
            explained = float((1.0 - (returns - predictions).var(unbiased=False) / variance).cpu()) if float(variance) > 1e-8 else None
        mean_or_zero = lambda xs: float(np.mean(xs)) if xs else 0.0
        return {
            "actor_loss": mean_or_zero(actor_losses), "critic_loss": mean_or_zero(value_losses),
            "entropy": mean_or_zero(entropies), "ratio": mean_or_zero(ratios),
            "approx_kl": mean_or_zero(attempted_kls),
            "max_approx_kl": max(attempted_kls, default=0.0),
            "accepted_max_approx_kl": max(accepted_kls, default=0.0),
            "clip_fraction": mean_or_zero(clip_fractions),
            "actor_grad_norm": mean_or_zero(actor_norms), "critic_grad_norm": mean_or_zero(value_norms),
            "actor_gradient_clip_fraction": mean_or_zero(actor_clips),
            "critic_gradient_clip_fraction": mean_or_zero(value_clips),
            "explained_variance": explained,
            "action_std_mean": float(self.actor_log_std.clamp(-5.0, 1.0).exp().mean().detach().cpu()),
            "actor_lr": self.actor_optimizer.param_groups[0]["lr"],
            "critic_lr": self.value_optimizer.param_groups[0]["lr"],
            "scheduled_actor_lr": scheduled_actor_lr,
            "scheduled_critic_lr": scheduled_critic_lr,
            "actor_lr_scale": self.actor_lr_scale,
            "kl_early_stopped": early_stopped,
            "actor_update_rejected": bool(rejected_actor_steps),
            "rejected_actor_steps": rejected_actor_steps,
            "actor_optimizer_attempts": actor_attempts,
            "optimizer_steps": len(actor_losses),
            "critic_optimizer_steps": len(value_losses),
            "epochs_completed": actor_epochs_completed,
            "critic_epochs_completed": critic_epochs_completed,
            "stop_epoch": actor_epochs_completed + 1 if early_stopped else None,
            "low_kl_streak": self.low_kl_streak,
            "max_abs_log_ratio": max_log_ratio,
        }

    def state_dict(self) -> dict[str, Any]:
        return {"policy_contract": self.policy_contract, "obs_dim": self.obs_dim, "reference_ppo": self.reference_ppo, "actor": self.actor.state_dict(), "actor_log_std": self.actor_log_std.detach().cpu(), "value": self.value.state_dict(), "actor_optimizer": self.actor_optimizer.state_dict(), "value_optimizer": self.value_optimizer.state_dict(), "actor_lr_scale": self.actor_lr_scale, "low_kl_streak": self.low_kl_streak}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("policy_contract") != self.policy_contract or state.get("obs_dim") != self.obs_dim or state.get("reference_ppo") != self.reference_ppo:
            raise ValueError("incompatible checkpoint: expected the 25-D reference PPO contract")
        _require_finite(state, "checkpoint")
        self.actor.load_state_dict(state["actor"])
        self.actor_log_std.data.copy_(state["actor_log_std"].to(self.device)); self.value.load_state_dict(state["value"])
        if "actor_optimizer" in state: self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        if "value_optimizer" in state: self.value_optimizer.load_state_dict(state["value_optimizer"])
        self.actor_lr_scale = float(state.get("actor_lr_scale", 1.0))
        self.low_kl_streak = int(state.get("low_kl_streak", 0))
        self.validate_training_state()
