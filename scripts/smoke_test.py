from __future__ import annotations

import numpy as np

from rl_risk_sac.algorithms import SACAgent
from rl_risk_sac.algorithms.replay_buffer import ReplayBuffer
from rl_risk_sac.envs import UR5DynamicObstacleEnv
from rl_risk_sac.utils.config import load_config
from rl_risk_sac.utils.seeding import set_seed


def main() -> None:
    config = load_config("configs/default.yaml")
    config["seed"] = 7
    config["env"]["max_episode_steps"] = 8
    set_seed(7)

    env = UR5DynamicObstacleEnv(config, method="ldrc_adaptive")
    observation, info = env.reset(seed=7)
    assert observation.shape == env.observation_space.shape
    assert np.isfinite(observation).all()
    assert "risk_global" in info

    agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], config, method="ldrc_adaptive")
    replay = ReplayBuffer(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        capacity=64,
        device=config.get("device", "cpu"),
    )

    for _ in range(12):
        action = env.action_space.sample()
        next_observation, reward, cost, terminated, truncated, info = env.step(action)
        replay.add(observation, action, reward, cost, next_observation, terminated or truncated)
        observation = next_observation
        if terminated or truncated:
            observation, _ = env.reset()

    batch = replay.sample(8)
    update_info = agent.update(batch)
    env.close()
    print(
        {
            "observation_dim": int(observation.shape[0]),
            "buffer_size": len(replay),
            "risk_global": float(info["risk_global"]),
            "d_min": float(info["d_min"]),
            "actor_loss": update_info["loss/actor"],
        }
    )


if __name__ == "__main__":
    main()
