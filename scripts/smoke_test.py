from __future__ import annotations

import copy

import numpy as np

from rl_risk_sac.algorithms import SACAgent
from rl_risk_sac.algorithms.replay_buffer import ReplayBuffer
from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config
from rl_risk_sac.utils.device import resolve_device
from rl_risk_sac.utils.seeding import set_seed


def main() -> None:
    config = load_config("configs/default.yaml")
    smoke_cfg = config["smoke"]
    config["seed"] = int(smoke_cfg["seed"])
    config["env"]["max_episode_steps"] = int(smoke_cfg["max_episode_steps"])
    config["device"] = resolve_device(config)
    set_seed(int(smoke_cfg["seed"]))

    env = UR5DynamicObstacleEnv(config, method=smoke_cfg["method"])
    observation, info = env.reset(seed=int(smoke_cfg["seed"]))
    assert observation.shape == env.observation_space.shape
    assert np.isfinite(observation).all()
    assert "risk_global" in info

    agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], config, method=smoke_cfg["method"])
    replay = ReplayBuffer(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        capacity=int(smoke_cfg["replay_capacity"]),
        device=config.get("device", "cpu"),
    )

    for _ in range(int(smoke_cfg["rollout_steps"])):
        action = env.action_space.sample()
        next_observation, reward, cost, terminated, truncated, info = env.step(action)
        replay.add(observation, action, reward, cost, next_observation, terminated or truncated)
        observation = next_observation
        if terminated or truncated:
            observation, _ = env.reset()

    batch = replay.sample(int(smoke_cfg["batch_size"]))
    update_info = agent.update(batch)
    env.close()

    no_obstacle_config = copy.deepcopy(config)
    no_obstacle_config["env"]["obstacle"]["enabled"] = False
    no_obstacle_env = UR5DynamicObstacleEnv(no_obstacle_config, method=smoke_cfg["method"])
    no_obstacle_observation, no_obstacle_info = no_obstacle_env.reset(seed=int(smoke_cfg["no_obstacle_seed"]))
    assert no_obstacle_observation.shape == no_obstacle_env.observation_space.shape
    assert np.isfinite(no_obstacle_observation).all()
    assert no_obstacle_info["obstacle_enabled"] is False
    assert no_obstacle_info["risk_global"] == 0.0
    no_obstacle_env.close()

    print(
        {
            "observation_dim": int(observation.shape[0]),
            "buffer_size": len(replay),
            "risk_global": float(info["risk_global"]),
            "d_min": float(info["d_min"]),
            "no_obstacle_risk": float(no_obstacle_info["risk_global"]),
            "actor_loss": update_info["loss/actor"],
        }
    )


if __name__ == "__main__":
    main()
