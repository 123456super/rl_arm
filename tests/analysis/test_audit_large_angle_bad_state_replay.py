from types import SimpleNamespace

import numpy as np

from scripts.replay.audit_large_angle_bad_state_replay import (
    describe, neighborhood, partition_metrics, select_centers,
)


def test_describe_accepts_generator_and_empty_values():
    assert describe(x for x in (1, 3))["mean"] == 2
    assert describe([]) == {"count": 0, "mean": None, "median": None}


def test_neighborhood_requires_joint_velocity_and_jacobian_not_only_pose():
    obs = np.zeros((4, 162), np.float32)
    obs[1, :6] = .3
    obs[2, 6:12] = .4
    obs[3, 21:75] = .2
    part = SimpleNamespace(size=4, obs=obs, raw=np.tile([.1, .1, .8, .8], (4, 1)),
                           step=np.full(4, 40))
    center = {"rho_orientation": .8, "rho_position": .1, "step": 40,
              "observation": np.zeros(162, np.float32)}
    pose, local, counts = neighborhood(part, [center, center])
    assert pose.tolist() == [True] * 4
    assert local.tolist() == [True, False, False, False]
    assert counts == [1, 1]  # Union does not double count repeated centers.
    assert neighborhood(part, [])[2] == []


def test_events_and_good_descent_use_pre_action_time_and_first_windows():
    episodes = [{"episode": 7, "success": False, "crossing_0p6_step": 10},
                {"episode": 8, "success": True, "crossing_0p6_step": 5}]
    rows = [{"episode": 7, "step": step, "rho_orientation": .8,
             "orientation_progress_rad_s": -.1, "actor_command_alignment": -.2}
            for step in range(5)]
    rows.extend([{"episode": 7, "step": 10, "rho_orientation": .59,
                  "orientation_progress_rad_s": -.1, "actor_command_alignment": -.2},
                 {"episode": 7, "step": 11, "rho_orientation": .61,
                  "orientation_progress_rad_s": -.1, "actor_command_alignment": -.2}])
    rows.extend({"episode": 8, "step": step, "rho_orientation": .75,
                 "orientation_progress_rad_s": progress}
                for step, progress in ((0, -.1), (1, .3), (6, .5)))
    bad, good = select_centers(episodes, rows)
    assert bad["low_alignment"][0]["step"] == 0
    assert bad["stagnation"][0]["step"] == 0
    assert bad["post_crossing_rebound"][0]["step"] == 11
    assert good[0]["step"] == 1


def test_measured_alignment_uses_next_velocity_and_excludes_clipping(monkeypatch):
    obs = np.zeros((3, 162), np.float32)
    obs[:, 78] = .2
    next_obs = np.zeros_like(obs)
    next_obs[:, 86] = [-.2, .2, 1.]
    part = SimpleNamespace(obs=obs, next_obs=next_obs,
                           raw=np.zeros((3, 29)), done=np.zeros((3, 1)),
                           episode=np.zeros(3), step=np.zeros(3))
    seen = []

    def rewards(raw, done, xi, lambda_self, gamma, config):
        seen.append(lambda_self)
        return np.zeros(3), np.zeros(3)

    monkeypatch.setattr("scripts.replay.audit_large_angle_bad_state_replay._vectorized_replay_rewards", rewards)
    result = partition_metrics(part, np.ones(3, bool), {0: "success"}, {}, .99, 240, .2)
    assert seen == [.2]
    assert result["measured_endpoint_angular_alignment"]["mean"] == 0
    assert result["measured_endpoint_negative_alignment_fraction"] == .5
    assert result["measured_endpoint_alignment_excluded"] == 1
