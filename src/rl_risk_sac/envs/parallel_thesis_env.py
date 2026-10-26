from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from multiprocessing.connection import Connection
from typing import Any, Collection

import numpy as np

from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv


_ERROR_MARKER = "__thesis_worker_error__"


class ParallelEnvWorkerError(RuntimeError):
    """A worker reported its original exception instead of a bare pipe EOF."""

    def __init__(self, index: int, payload: dict[str, Any]) -> None:
        self.worker_index = index
        self.command = payload["command"]
        self.diagnostics = payload.get("diagnostics")
        self.remote_traceback = payload.get("traceback", "")
        super().__init__(
            f"thesis-env-{index} failed during {self.command}: "
            f"{payload['type']}: {payload['message']}\n{self.remote_traceback}"
        )


def _error_payload(command: str, error: Exception) -> dict[str, Any]:
    return {
        _ERROR_MARKER: True, "command": command,
        "type": type(error).__name__, "message": str(error),
        "traceback": traceback.format_exc(),
        "diagnostics": getattr(error, "diagnostics", None),
    }


def _execute(env, command, payload, step_info_keys):
    if command == "reset":
        contract, reset_seed = payload
        env.configure_episode(**contract)
        return env.reset(seed=reset_seed)
    if command == "step":
        profile_started = time.perf_counter() if os.environ.get("THESIS_PROFILE") == "1" else None
        observation, reward, cost, terminated, truncated, info = env.step(payload)
        if profile_started is not None:
            info = dict(info)
            info["_profile_env_step_s"] = time.perf_counter() - profile_started
        if step_info_keys is not None:
            info = {key: info[key] for key in step_info_keys}
        return observation, reward, cost, terminated, truncated, info
    if command == "state":
        return env.episode_state_dict()
    if command == "pose_crossing_diagnostics":
        return env.pose_crossing_diagnostics()
    if command == "restore":
        return env.restore_episode_state(payload)
    if command == "reset_bad_state":
        center, contract = payload
        return env.reset_bad_state(center, contract)
    if command == "set_rng":
        env.rng.bit_generator.state = payload
        return None
    raise ValueError(f"unknown parallel environment command {command!r}")


def _worker(
    connection: Connection,
    config: dict[str, Any],
    seed: int,
    cpu_id: int | None,
    step_info_keys: tuple[str, ...] | None,
) -> None:
    if cpu_id is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {int(cpu_id)})
    env = None
    command = "initialize"
    try:
        env = ThesisHomotopyEnv(config)
        env.rng = np.random.default_rng(seed)
        connection.send(("ready", env.observation_space.shape[0], env.action_space.shape[0]))
        failure = None
        while True:
            command, payload = connection.recv()
            if command == "close":
                connection.send(None)
                break
            # After an unrecoverable command failure, keep the channel alive
            # for shutdown/error reporting, but never snapshot a half-reset env.
            if failure is not None:
                connection.send(failure)
                continue
            try:
                result = _execute(env, command, payload, step_info_keys)
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


class ParallelThesisEnvPool:
    """Process-isolated PyBullet environments controlled by one learner process."""

    def __init__(
        self,
        config: dict[str, Any],
        seeds: list[int],
        *,
        cpu_ids: list[int] | None = None,
        step_info_keys: Collection[str] | None = None,
        start_method: str = "spawn",
    ) -> None:
        if not seeds:
            raise ValueError("parallel environment pool requires at least one seed")
        if cpu_ids is not None and len(cpu_ids) != len(seeds):
            raise ValueError("cpu_ids must match the number of environment seeds")
        if start_method not in mp.get_all_start_methods():
            raise ValueError(f"unsupported multiprocessing start method {start_method!r}")
        context = mp.get_context(start_method)
        selected_info_keys = (
            None if step_info_keys is None else tuple(dict.fromkeys(step_info_keys))
        )
        self.connections: list[Connection] = []
        self.processes: list[mp.Process] = []
        self._pending_step_indices: tuple[int, ...] | None = None
        for index, seed in enumerate(seeds):
            parent, child = context.Pipe()
            cpu_id = None if cpu_ids is None else int(cpu_ids[index])
            process = context.Process(
                target=_worker,
                args=(child, config, int(seed), cpu_id, selected_info_keys),
                name=f"thesis-env-{index}",
            )
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)
        try:
            ready = list(self._recv_many(range(len(self.connections)), "initialize").values())
        except Exception:
            self.close()
            raise
        if any(message[0] != "ready" for message in ready):
            self.close()
            raise RuntimeError(f"parallel environment failed to initialize: {ready}")
        dimensions = {(int(message[1]), int(message[2])) for message in ready}
        if len(dimensions) != 1:
            self.close()
            raise RuntimeError(f"parallel environments disagree on dimensions: {dimensions}")
        self.observation_dim, self.action_dim = dimensions.pop()

    def __len__(self) -> int:
        return len(self.connections)

    def _recv_many(self, indices, command):
        results, errors = {}, []
        # Drain every requested reply before raising. Otherwise another worker's
        # old reset/step reply could be mistaken for a checkpoint state reply.
        for index in indices:
            try:
                result = self.connections[index].recv()
            except (EOFError, OSError) as error:
                errors.append(ParallelEnvWorkerError(index, {
                    "command": command, "type": type(error).__name__,
                    "message": f"worker connection lost (exitcode={self.processes[index].exitcode})",
                }))
                continue
            if isinstance(result, dict) and result.get(_ERROR_MARKER) is True:
                errors.append(ParallelEnvWorkerError(index, result))
            else:
                results[index] = result
        if errors:
            raise errors[0]
        return results

    def reset_many(
        self, requests: dict[int, dict[str, Any]],
        *, seeds: dict[int, int] | None = None,
    ) -> dict[int, tuple[np.ndarray, dict[str, Any]]]:
        if seeds is not None and set(seeds) != set(requests):
            raise ValueError("reset seeds must match reset request workers")
        for index, contract in requests.items():
            reset_seed = None if seeds is None else int(seeds[index])
            self.connections[index].send(("reset", (contract, reset_seed)))
        return self._recv_many(requests, "reset")

    def step_many(
        self, actions: dict[int, np.ndarray],
    ) -> dict[int, tuple[np.ndarray, float, float, bool, bool, dict[str, Any]]]:
        self.send_steps(actions)
        return self.recv_steps()

    def pose_crossing_diagnostics_many(
        self, indices: Collection[int],
    ) -> dict[int, dict[str, object]]:
        """Collect on-demand kinematic diagnostics from selected workers."""
        selected = tuple(dict.fromkeys(int(index) for index in indices))
        if self._pending_step_indices is not None:
            raise RuntimeError(
                "cannot request diagnostics while an environment step is pending"
            )
        for index in selected:
            self.connections[index].send(("pose_crossing_diagnostics", None))
        return self._recv_many(selected, "pose_crossing_diagnostics")

    def send_steps(self, actions: dict[int, np.ndarray]) -> None:
        """Start one simulation step on every selected worker without waiting."""
        if self._pending_step_indices is not None:
            raise RuntimeError("parallel environment step is already in flight")
        indices = tuple(actions)
        if not indices:
            raise ValueError("parallel environment step requires at least one action")
        if len(set(indices)) != len(indices) or any(
            index < 0 or index >= len(self.connections) for index in indices
        ):
            raise IndexError("parallel environment action index is out of range")
        for index, action in actions.items():
            self.connections[index].send(("step", np.asarray(action, dtype=np.float32)))
        self._pending_step_indices = indices

    def recv_steps(
        self,
    ) -> dict[int, tuple[np.ndarray, float, float, bool, bool, dict[str, Any]]]:
        """Collect the simulation step most recently started by ``send_steps``."""
        if self._pending_step_indices is None:
            raise RuntimeError("no parallel environment step is in flight")
        indices = self._pending_step_indices
        try:
            return self._recv_many(indices, "step")
        finally:
            self._pending_step_indices = None

    def states(self, indices: list[int] | None = None) -> dict[int, dict[str, Any]]:
        selected = list(range(len(self))) if indices is None else list(indices)
        for index in selected:
            self.connections[index].send(("state", None))
        return self._recv_many(selected, "state")

    def restore_many(self, states: dict[int, dict[str, Any]]) -> dict[int, np.ndarray]:
        for index, state in states.items():
            self.connections[index].send(("restore", state))
        return self._recv_many(states, "restore")

    def reset_bad_states_many(self, requests: dict[int, tuple[dict, dict]]):
        for index, request in requests.items():
            self.connections[index].send(("reset_bad_state", request))
        return self._recv_many(requests, "reset_bad_state")

    def set_rng_states(self, states: dict[int, dict[str, Any]]) -> None:
        for index, state in states.items():
            self.connections[index].send(("set_rng", state))
        self._recv_many(states, "set_rng")

    def close(self) -> None:
        if self._pending_step_indices is not None:
            try:
                self.recv_steps()
            except (OSError, EOFError, ParallelEnvWorkerError):
                self._pending_step_indices = None
        for connection, process in zip(self.connections, self.processes):
            if process.is_alive():
                try:
                    connection.send(("close", None))
                except (OSError, EOFError):
                    pass
        for connection, process in zip(self.connections, self.processes):
            if process.is_alive():
                try:
                    if connection.poll(5.0):
                        connection.recv()
                except (OSError, EOFError):
                    pass
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
            connection.close()

    def __enter__(self) -> "ParallelThesisEnvPool":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
