"""Parallel real-obstacle handoffs using the evaluation controller unchanged."""
from __future__ import annotations

import copy
import os

import numpy as np
import torch

from rl_risk_sac.control.joint_policy import JointControlFailure, JointPolicyController
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool, _error_payload
from rl_risk_sac.envs.parallel_local_avoidance_env import FrozenCPUReachingPolicy
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.tasks.local_avoidance import actor_digest
from rl_risk_sac.tasks.local_samples import simulation_config
from rl_risk_sac.tasks.obstacle_observation import observation_tensors, stack_observations


CONTROLLER_FIELDS = ("role", "sample", "local_steps", "steps", "attempts", "passages",
                     "reference_calls", "policy_steps", "events", "finished")


class FrozenJointPolicy:
    def __init__(self, reaching_actor, avoidance_actor):
        self.reaching = FrozenCPUReachingPolicy(reaching_actor)
        self.avoidance_actor = avoidance_actor.eval().requires_grad_(False)

    def select_action(self, observation, deterministic=True, *, policy="reaching"):
        if policy == "reaching":
            return self.reaching.select_action(observation, deterministic)
        if not deterministic:
            raise ValueError("avoidance policy must remain frozen and deterministic")
        state, features, mask = observation_tensors(stack_observations([observation]), "cpu")
        with torch.no_grad():
            return self.avoidance_actor.deterministic(state, features, mask)[0].numpy()


class JointHandoffEnvironment:
    def __init__(self, config, reaching_actor, avoidance_actor, max_steps):
        self.base = ThesisHomotopyEnv(simulation_config(config), cache_geometry_queries=True)
        try:
            self.controller = JointPolicyController(config, FrozenJointPolicy(reaching_actor, avoidance_actor))
        except BaseException:
            self.base.close()
            raise
        self.max_steps, self.observation, self.prepared = max_steps, None, None

    def reset(self, contract, seed):
        self.base.configure_episode(**contract)
        self.observation, info = self.base.reset(seed=seed)
        self.base.horizon = self.max_steps
        self.controller.reset()
        self.prepared = None
        return self.observation, info

    def restore(self, state):
        self.prepared = None
        self.controller.reset()
        if "environment" in state:
            self.observation = self.base.restore_episode_state(copy.deepcopy(state["environment"]))
            for name, value in state["controller"].items():
                setattr(self.controller, name, copy.deepcopy(value))
        else:
            self.observation = self.base.restore_episode_state(copy.deepcopy(state))
        self.base.horizon = self.max_steps
        return self.observation

    def prepare(self, reaching_action=None):
        if self.prepared is not None:
            raise RuntimeError("a prepared action must be stepped before preparing again")
        task_decision = self.controller.role == "reaching"
        try:
            self.prepared = self.controller.prepare_action(self.base, self.observation, reaching_action=reaching_action)
        except JointControlFailure as error:
            return dict(role=None, failure=f"reference_unavailable:{error}", observation=self.observation,
                        task_decision=task_decision)
        return dict(role=self.prepared[2], failure=None, observation=self.observation, task_decision=task_decision)

    def step(self, reaching_action):
        if self.prepared is None:
            raise RuntimeError("prepare the controller before stepping")
        if self.prepared[2] == "reaching" and reaching_action is None:
            raise ValueError("missing reaching action")
        self.observation, terminated, truncated, info = self.controller.step(
            self.base, self.observation, prepared=self.prepared, reaching_action=reaching_action)
        self.prepared = None
        if info["mesh_only_collision"]:
            # The controller also catches substep raw-mesh contacts that the
            # live environment's ordinary contact event can miss. Apply its
            # existing failure penalty once, without changing reward weights.
            info["environment_reward"] -= self.base.reward_parameters["hard_failure_penalty"]
        info.update(avoidance_attempts=self.controller.attempts, verified_passages=self.controller.passages)
        return (self.observation, info["environment_reward"], info["environment_cost"],
                terminated, truncated, info)

    def state(self):
        if self.prepared is not None:
            raise RuntimeError("checkpoint only at a completed physical step boundary")
        return dict(environment=self.base.episode_state_dict(),
                    controller={name: copy.deepcopy(getattr(self.controller, name)) for name in CONTROLLER_FIELDS})

    def close(self):
        self.controller.close()
        self.base.close()


def _joint_worker(connection, config, seed, cpu_id, step_info_keys, reaching_actor, avoidance_actor, max_steps):
    if cpu_id is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {int(cpu_id)})
    torch.set_num_threads(1)
    env, command = None, "initialize"
    try:
        env = JointHandoffEnvironment(config, reaching_actor, avoidance_actor, max_steps)
        env.base.rng = np.random.default_rng(seed)
        connection.send(("ready", 241, 6))
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
                    result = env.reset(*payload)
                elif command == "restore":
                    result = env.restore(payload)
                elif command == "prepare":
                    result = env.prepare(payload)
                elif command == "step":
                    result = env.step(payload)
                elif command == "state":
                    result = env.state()
                elif command == "policy_info":
                    result = dict(reaching=actor_digest(env.controller.agent.reaching.actor),
                                  avoidance=actor_digest(env.controller.agent.avoidance_actor),
                                  reaching_obstacle_aware=bool(getattr(env.controller.agent.reaching.actor,
                                                                      "reaching_obstacle_aware", False)),
                                  cuda_initialized=torch.cuda.is_initialized())
                else:
                    raise ValueError(f"unknown joint handoff command {command!r}")
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


class ParallelJointHandoffEnvPool(ParallelThesisEnvPool):
    def __init__(self, config, seeds, reaching_actor, avoidance_actor, *, max_steps=1000, start_method="spawn"):
        # Parent updates this CPU snapshot only while every worker is idle.
        # The learner may then update its separate GPU model during simulation.
        self.reaching_snapshot = copy.deepcopy(reaching_actor).cpu().eval().requires_grad_(False).share_memory()
        avoidance = copy.deepcopy(avoidance_actor).cpu().eval().requires_grad_(False).share_memory()
        worker_config = copy.deepcopy(config)
        worker_config["device"] = "cpu"
        super().__init__(worker_config, seeds, start_method=start_method, _worker_target=_joint_worker,
                         _worker_extra_args=(self.reaching_snapshot, avoidance, max_steps))

    def sync_reaching(self, actor):
        if self._pending_step_indices is not None:
            raise RuntimeError("cannot change the reaching snapshot during simulation")
        self.reaching_snapshot.load_state_dict(actor.state_dict())

    def prepare_many(self, indices, *, reaching_actions=None):
        if self._pending_step_indices is not None:
            raise RuntimeError("cannot prepare while a step is pending")
        for index in indices:
            self.connections[index].send(("prepare", None if reaching_actions is None else reaching_actions[index]))
        return self._recv_many(indices, "prepare")

    def send_steps(self, actions):
        if self._pending_step_indices is not None:
            raise RuntimeError("parallel environment step is already in flight")
        if not actions:
            raise ValueError("no selected workers")
        for index, action in actions.items():
            self.connections[index].send(("step", action))
        self._pending_step_indices = tuple(actions)

    def policy_info(self):
        if self._pending_step_indices is not None:
            raise RuntimeError("cannot inspect policies during simulation")
        for connection in self.connections:
            connection.send(("policy_info", None))
        return self._recv_many(range(len(self)), "policy_info")
