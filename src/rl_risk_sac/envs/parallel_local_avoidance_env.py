"""Process-isolated local passage environments; one GPU learner stays in the parent."""
from __future__ import annotations

import copy
import os

import numpy as np
import torch

from rl_risk_sac.envs.local_avoidance_env import LocalAvoidanceEnv
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool, _error_payload
from rl_risk_sac.tasks.local_avoidance import actor_digest


class FrozenCPUReachingPolicy:
    """Actor-only S0 copy; workers neither build critics nor initialize CUDA."""

    def __init__(self, actor):
        self.actor = actor.cpu().eval().requires_grad_(False)

    def select_action(self, observation, deterministic=True):
        if not deterministic:
            raise ValueError("recovery requires deterministic frozen reaching actions")
        state = np.asarray(observation, dtype=np.float32)[:241]
        with torch.no_grad():
            return self.actor.deterministic(torch.from_numpy(state[None]))[0].numpy()


def _local_worker(connection, config, seed, cpu_id, step_info_keys, actor):
    if cpu_id is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {int(cpu_id)})
    torch.set_num_threads(1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = copy.deepcopy(config)
    config["device"] = "cpu"
    env = None
    command = "initialize"
    try:
        reaching = FrozenCPUReachingPolicy(actor)
        env = LocalAvoidanceEnv(config, reaching)
        connection.send(("ready", env.observation_space.shape[0], env.action_space.shape[0]))
        failure = None
        while True:
            command, payload = connection.recv()
            if command == "close":
                connection.send(None)
                break
            if failure is not None:
                connection.send(failure)
                continue
            try:
                if command == "reset":
                    sample, reset_seed = payload
                    result = env.reset(seed=reset_seed, options={"sample": sample})
                elif command == "step":
                    result = env.step(payload)
                elif command == "state":
                    result = dict(environment=env.state_dict(), torch_rng=torch.get_rng_state(),
                                  numpy_rng=np.random.get_state())
                elif command == "restore":
                    if "sample" in payload:  # Legacy serial local environment state.
                        result = env.load_state_dict(payload)
                    else:
                        result = env.load_state_dict(payload["environment"])
                        torch.set_rng_state(payload["torch_rng"].cpu())
                        np.random.set_state(payload["numpy_rng"])
                elif command == "policy_info":
                    result = dict(device=str(next(reaching.actor.parameters()).device),
                                  actor_sha256=actor_digest(reaching.actor),
                                  trainable=any(p.requires_grad for p in reaching.actor.parameters()),
                                  cuda_initialized=torch.cuda.is_initialized())
                else:
                    raise ValueError(f"unknown local environment command {command!r}")
            except Exception as error:
                failure = _error_payload(command, error)
                connection.send(failure)
            else:
                connection.send(result)
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        connection.send(_error_payload(command, error))
    finally:
        if env is not None:
            env.close()
        connection.close()


class ParallelLocalAvoidanceEnvPool(ParallelThesisEnvPool):
    """Reuse the existing spawn/pipe transport and structured worker-error handling."""

    def __init__(self, config, seeds, reaching_actor, *, cpu_ids=None, start_method="spawn"):
        actor = copy.deepcopy(reaching_actor).cpu().eval().requires_grad_(False)
        super().__init__(config, seeds, cpu_ids=cpu_ids, start_method=start_method,
                         _worker_target=_local_worker, _worker_extra_args=(actor,))

    def reset_many(self, requests, *, seeds=None):
        if self._pending_step_indices is not None:
            raise RuntimeError("cannot reset local environments while steps are pending")
        if seeds is not None and set(seeds) != set(requests):
            raise ValueError("reset seeds must match reset request workers")
        if any(index < 0 or index >= len(self) for index in requests):
            raise IndexError("local reset worker index is out of range")
        for index, sample in requests.items():
            reset_seed = None if seeds is None else int(seeds[index])
            self.connections[index].send(("reset", (sample, reset_seed)))
        return self._recv_many(requests, "reset")

    def states(self, indices=None):
        if self._pending_step_indices is not None:
            raise RuntimeError("cannot snapshot local environments while steps are pending")
        return super().states(indices)

    def restore_many(self, states):
        if self._pending_step_indices is not None:
            raise RuntimeError("cannot restore local environments while steps are pending")
        return super().restore_many(states)

    def policy_info(self):
        if self._pending_step_indices is not None:
            raise RuntimeError("cannot inspect policies while steps are pending")
        for connection in self.connections:
            connection.send(("policy_info", None))
        return self._recv_many(range(len(self)), "policy_info")
