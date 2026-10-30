"""Reuse the existing asynchronous process pool; workers simulate, learner infers."""
from __future__ import annotations

import os
import signal

import numpy as np

from rl_risk_sac.envs.paper_reaching_env import PaperReachingEnv
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool, _error_payload


def _paper_worker(connection, config, seed, cpu_id, step_info_keys):
    # A terminal Ctrl+C targets the whole process group. Let the learner drain
    # the batch and save; workers must not raise KeyboardInterrupt mid-step.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if cpu_id is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {int(cpu_id)})
    env, command = None, "initialize"
    try:
        env = PaperReachingEnv(config)
        env.rng = np.random.default_rng(seed)
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
                    options, reset_seed = payload
                    result = env.reset(seed=reset_seed, options=options)
                elif command == "step":
                    result = env.step(payload)
                    if step_info_keys is not None:
                        result = (*result[:-1], {key: result[-1][key] for key in step_info_keys})
                elif command == "state":
                    result = env.episode_state_dict()
                elif command == "restore":
                    result = env.restore_episode_state(payload)
                elif command == "reference":
                    result = env.set_reference(*payload)
                else:
                    raise ValueError(f"unknown paper worker command {command}")
                connection.send(result)
            except Exception as error:
                failure = _error_payload(command, error)
                connection.send(failure)
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        connection.send(_error_payload(command, error))
    finally:
        if env is not None:
            env.close()
        connection.close()


class ParallelPaperEnvPool(ParallelThesisEnvPool):
    def __init__(self, config, seeds, *, step_info_keys=None):
        # Spawn imports NumPy/Torch before entering the worker. Inherit a
        # one-thread BLAS setting so eight small IK solves do not oversubscribe
        # the machine; restore the learner's environment after initialization.
        variables = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        previous = {key: os.environ.get(key) for key in variables}
        try:
            for key in variables:
                os.environ[key] = "1"
            super().__init__(config, seeds, step_info_keys=step_info_keys, _worker_target=_paper_worker)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def set_references(self, references):
        if self._pending_step_indices is not None:
            raise RuntimeError("cannot change references during an in-flight step")
        for index, reference in references.items():
            self.connections[index].send(("reference", reference))
        return self._recv_many(references, "reference")
