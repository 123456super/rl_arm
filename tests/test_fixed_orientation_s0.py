"""Shared fixed-pose task for keypoint and reference PPO observations/rewards."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pybullet as p

from rl_risk_sac.envs.thesis_homotopy_env_reduced_observation import (
    REFERENCE_GOAL_EULER_RAD,
    ThesisHomotopyEnv,
)
from rl_risk_sac.tasks.thesis_reaching_reduced_observation import pose_error
from rl_risk_sac.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class FixedOrientationS0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "configs/experiments/paper_full.yaml")
        cls.config["thesis"]["simple_reaching_mode"] = True

    def setUp(self):
        self.env = ThesisHomotopyEnv(deepcopy(self.config))
        self.addCleanup(self.env.close)
        self.env.configure_episode("none", xi=1.0, strict=False)

    def test_resets_have_random_valid_positions_fixed_orientation_and_configured_home(self):
        targets = []
        expected = np.asarray(p.getQuaternionFromEuler(REFERENCE_GOAL_EULER_RAD))
        for _ in range(12):
            obs, info = self.env.reset()
            targets.append(info["goal_position"])
            np.testing.assert_allclose(
                self.env.robot.joint_state_exact()[0],
                self.env.fixed_initial_joint_positions, atol=1e-7,
            )
            np.testing.assert_array_equal(self.env.robot.joint_state_exact()[1], np.zeros(6))
            np.testing.assert_allclose(info["goal_quaternion"], expected, atol=1e-7)
            self.assertEqual(info["goal_sampling_mode"], "fixed_orientation")
            self.assertEqual(obs.shape, (38,))
            self.assertTrue(self.env.observation_space.contains(obs))
            self.assertIsNone(self.env.obstacle_id)
            x, y, z = info["goal_position"]
            self.assertLessEqual(x * x + y * y + z * z, 0.5625 + 1e-7)
            self.assertGreaterEqual(x * x + y * y, 0.09 - 1e-7)
            self.assertGreaterEqual(y, 0.0)
            self.assertGreaterEqual(z, 0.05)
            validation = info["goal_ik_validation"]
            self.assertTrue(validation["reachable"])
            self.assertTrue(validation["collision_free"])
            self.assertLessEqual(validation["position_error_m"], 0.01)
            self.assertLessEqual(validation["orientation_error_rad"], 0.10)
            self.assertGreaterEqual(validation["self_clearance_m"], self.env.d_self_safe)
        self.assertGreater(np.max(np.ptp(targets, axis=0)), 0.1)

    def test_seeded_reset_reproduces_target_and_ik(self):
        obs1, info1 = self.env.reset(seed=11001)
        self.env.step(np.full(6, 0.05))
        obs2, info2 = self.env.reset(seed=11001)
        np.testing.assert_array_equal(obs1, obs2)
        np.testing.assert_array_equal(info1["goal_position"], info2["goal_position"])
        np.testing.assert_array_equal(info1["goal_joint_positions"], info2["goal_joint_positions"])

    def test_ik_endpoint_is_real_and_requires_five_holds(self):
        _, info = self.env.reset(seed=11001)
        self.env._set_joint_state(info["goal_joint_positions"])
        p.performCollisionDetection(physicsClientId=self.env.client_id)
        self.assertFalse(any(self.env._collision_events().values()))
        pos, quat = self.env.robot.end_effector_pose()
        ep, er = pose_error(self.env.goal_position, self.env.goal_quaternion, pos, quat)
        self.assertLessEqual(np.linalg.norm(ep), 0.01)
        self.assertLessEqual(np.linalg.norm(er), 0.10)
        for step in range(5):
            obs, reward, cost, terminated, truncated, info = self.env.step(self.env.goal_joint_positions)
            self.assertTrue(np.isfinite(reward))
            self.assertTrue(np.isfinite(obs).all())
            self.assertFalse(truncated)
            self.assertEqual(terminated, step == 4)
            self.assertEqual(bool(info["success"]), step == 4)

    def test_keypoint_action_is_an_absolute_target(self):
        self.env.reset(seed=11001)
        q = np.asarray([0.1, -0.15, 0.2, -0.1, 0.05, -0.1])
        self.env._set_joint_state(q)
        action = np.asarray([0.01, -0.02, 0.03, -0.01, 0.02, 0.01])
        with patch.object(p, "setJointMotorControlArray", wraps=p.setJointMotorControlArray) as motor:
            self.env.step(action)
        np.testing.assert_allclose(motor.call_args.kwargs["targetPositions"], action, atol=1e-6)

    def test_failed_solver_restores_home_and_does_not_accept_fake_target(self):
        self.env.reset(seed=11001)
        q, qdot = self.env.robot.joint_state_exact()
        self.env.goal_sample_max_attempts = 3

        def fail(*args, **kwargs):
            self.env._set_joint_state(np.full(6, 0.2))
            raise RuntimeError("solver failure")

        with patch.object(self.env, "_propose_reference_goal_position", return_value=np.array([0.4, 0.3, 0.2])):
            with patch.object(self.env.robot, "check_pose_ik", side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, "solver failure"):
                    self.env._sample_fixed_orientation_goal_pose()
        np.testing.assert_array_equal(self.env.robot.joint_state_exact()[0], q)
        np.testing.assert_array_equal(self.env.robot.joint_state_exact()[1], qdot)

    def test_unreachable_proposals_exhaust_budget(self):
        self.env.reset(seed=11001)
        self.env.goal_sample_max_attempts = 3
        invalid = {"reachable": False, "finite": True, "within_limits": True}
        with patch.object(self.env, "_propose_reference_goal_position", return_value=np.array([0.4, 0.3, 0.2])):
            with patch.object(self.env.robot, "check_pose_ik", return_value=invalid) as solver:
                with self.assertRaisesRegex(RuntimeError, "fixed-orientation goal"):
                    self.env._sample_fixed_orientation_goal_pose()
        self.assertEqual(solver.call_count, 3)
        np.testing.assert_allclose(
            self.env.robot.joint_state_exact()[0], self.env.fixed_initial_joint_positions,
            atol=1e-7,
        )

    def test_colliding_ik_endpoint_is_rejected(self):
        _, info = self.env.reset(seed=11001)
        validation = info["goal_ik_validation"]
        clear = {"self_collision": False, "obstacle_collision": False, "environment_collision": False}
        colliding = {**clear, "self_collision": True}
        with patch.object(self.env, "_propose_reference_goal_position", return_value=info["goal_position"]):
            with patch.object(self.env.robot, "check_pose_ik", return_value=validation):
                with patch.object(self.env, "_collision_events", side_effect=[colliding, clear]):
                    self.env._sample_fixed_orientation_goal_pose()
        self.assertEqual(self.env.goal_sample_attempts_last, 2)
        np.testing.assert_allclose(
            self.env.robot.joint_state_exact()[0], self.env.fixed_initial_joint_positions,
            atol=1e-7,
        )

    def test_ready_home_is_bent_full_rank_and_close_to_target_orientation(self):
        self.env.reset(seed=11001)
        singular_values = np.linalg.svd(self.env.robot.end_effector_jacobian(), compute_uv=False)
        self.assertGreater(float(singular_values[-1]), 0.05)
        self.assertLess(float(singular_values[0] / singular_values[-1]), 20.0)
        self.assertFalse(any(self.env._collision_events().values()))
        self.assertGreaterEqual(self.env._self_geometry().distance_min, self.env.d_self_safe)
        position, quaternion = self.env.robot.end_effector_pose()
        self.assertGreater(float(position[2]), 0.20)
        _, orientation_error = pose_error(position, self.env.fixed_goal_quaternion, position, quaternion)
        self.assertLess(float(np.linalg.norm(orientation_error)), 0.01)
        np.testing.assert_allclose(
            np.rad2deg(self.env.fixed_initial_joint_positions),
            [45.0, -90.0, 90.0, 0.0, 45.0, -90.0], atol=1e-7,
        )

    def test_default_contract_uses_configured_success_tolerances(self):
        config = deepcopy(self.config)
        config["thesis"]["position_tolerance"] = 0.007
        config["thesis"]["orientation_tolerance"] = 0.08
        env = ThesisHomotopyEnv(config)
        self.addCleanup(env.close)
        self.assertEqual(env.contract.position_tolerance, 0.007)
        self.assertEqual(env.contract.orientation_tolerance, 0.08)

    def test_fixed_goal_ignores_legacy_orientation_sampling(self):
        self.env.configure_episode(
            "none", xi=1.0, strict=False,
            target_distance_min_m=0.03, target_distance_max_m=0.5,
            target_orientation_min_rad=2.0, target_orientation_max_rad=3.0,
        )
        with patch.object(self.env, "_sample_task_space_goal_pose", side_effect=AssertionError("legacy sampler")):
            _, info = self.env.reset(seed=11001)
        np.testing.assert_allclose(info["goal_quaternion"], self.env.fixed_goal_quaternion)

    def reference_env(self, config=None):
        config = deepcopy(self.config if config is None else config)
        config["thesis"]["reward"]["reward_mode"] = "reference_ppo"
        env = ThesisHomotopyEnv(config)
        self.addCleanup(env.close)
        env.configure_episode("none", xi=1.0, strict=False)
        return env

    def test_reference_mode_shares_home_validated_targets_and_keeps_25_inputs(self):
        env = self.reference_env()
        self.assertEqual(env.reference_keypoint_offset_m, 0.001)
        for seed in [11001, 31001, 51001, None, None, None]:
            keypoint_obs, keypoint_info = self.env.reset(seed=seed)
            with patch.object(env.robot, "check_pose_ik", wraps=env.robot.check_pose_ik) as solver:
                obs, info = env.reset(seed=seed)
            self.assertGreater(solver.call_count, 0)
            self.assertEqual(obs.shape, (25,))
            self.assertEqual(keypoint_obs.shape, (38,))
            self.assertTrue(env.observation_space.contains(obs))
            np.testing.assert_array_equal(env.robot.joint_state_exact()[0], self.env.robot.joint_state_exact()[0])
            np.testing.assert_allclose(env.robot.joint_state_exact()[0], self.env.fixed_initial_joint_positions, atol=1e-7)
            np.testing.assert_array_equal(env.robot.joint_state_exact()[1], np.zeros(6))
            for key in ["goal_position", "goal_quaternion", "goal_joint_positions"]:
                np.testing.assert_array_equal(info[key], keypoint_info[key])
            self.assertEqual(info["goal_sample_attempts"], keypoint_info["goal_sample_attempts"])
            self.assertEqual(info["goal_sampling_mode"], "fixed_orientation")
            np.testing.assert_allclose(
                info["keypoint_distances"], info["goal_error_norm"], atol=1e-7,
            )
            self.assertTrue(info["goal_ik_validation"]["reachable"])
            self.assertTrue(info["goal_ik_validation"]["collision_free"])
            self.assertLessEqual(info["goal_ik_validation"]["position_error_m"], .01)
            self.assertLessEqual(info["goal_ik_validation"]["orientation_error_rad"], .10)

    def test_reference_success_uses_weighted_error_and_one_step_threshold(self):
        env = self.reference_env()
        env.reset(seed=11001)
        original_observe = env._observe

        def precise_observe():
            obs, info, geom, self_geom, _, _ = original_observe()
            info["keypoint_distances"] = np.full(3, 1.0e-5, dtype=np.float32)
            return obs, info, geom, self_geom, np.zeros(3), np.array([1.0e-5, 0., 0.])

        with patch.object(env, "_observe", side_effect=precise_observe):
            obs, reward, _, terminated, truncated, info = env.step(env.goal_joint_positions)
        self.assertEqual(obs.shape, (25,))
        self.assertEqual(info["success_hold_count"], 1)
        self.assertEqual(info["success_hold_steps_required"], 1)
        self.assertTrue(terminated)
        self.assertTrue(info["success"])
        self.assertFalse(truncated)
        error = info["reference_error"]
        self.assertLess(error, 1.0e-4)
        expected = -.001 * error**2 - np.log(error**2 + 1e-4)
        self.assertAlmostEqual(reward, expected, places=6)

    def test_both_modes_enforce_both_error_thresholds_and_reset_hold_on_exit(self):
        for env in [self.env]:
            env.reset(seed=11001)
            env._set_joint_state(env.goal_joint_positions)
            original_observe = env._observe

            def observe_with_errors(position, orientation):
                def observe():
                    obs, info, geom, self_geom, _, _ = original_observe()
                    return obs, info, geom, self_geom, np.array([position, 0., 0.]), np.array([orientation, 0., 0.])
                return observe

            # Position-only or orientation-only satisfaction must not pass.
            cases = [(.0101, .09, 0), (.009, .1001, 0),
                     (.01, .10, 1), (.01, .10, 2), (.0101, .09, 0)]
            cases += [(.009, .09, hold) for hold in range(1, 6)]
            for pos, ori, hold in cases:
                with patch.object(env, "_observe", side_effect=observe_with_errors(pos, ori)):
                    _, _, _, terminated, truncated, info = env.step(env.goal_joint_positions)
                self.assertEqual(info["success_hold_count"], hold)
                self.assertEqual(info["strict_pose_reached"], pos <= .01 and ori <= .10)
                self.assertEqual(terminated, hold == 5)
                self.assertFalse(truncated)

    def test_reference_rejects_unreachable_targets_instead_of_faking_ik_success(self):
        env = self.reference_env()
        env.goal_sample_max_attempts = 3
        with patch.object(env, "_propose_reference_goal_position", return_value=np.array([.4, .3, .2])):
            with patch.object(env.robot, "check_pose_ik", return_value={"reachable": False}) as solver:
                with self.assertRaisesRegex(RuntimeError, "fixed-orientation goal"):
                    env.reset(seed=11001)
        self.assertEqual(solver.call_count, 3)
        np.testing.assert_allclose(env.robot.joint_state_exact()[0], env.fixed_initial_joint_positions)

    def test_reference_uses_configured_orientation_and_tolerances_not_legacy_constants(self):
        config = deepcopy(self.config)
        config["thesis"]["fixed_orientation_goal"]["euler_rad"] = [.2, 1.4, 1.2]
        config["thesis"].update(position_tolerance=.007, orientation_tolerance=.08)
        env = self.reference_env(config)
        _, info = env.reset(seed=11001)
        np.testing.assert_allclose(info["goal_quaternion"], p.getQuaternionFromEuler([.2, 1.4, 1.2]), atol=1e-7)
        self.assertEqual(env.contract.position_tolerance, .007)
        self.assertEqual(env.contract.orientation_tolerance, .08)
        self.assertLessEqual(info["goal_ik_validation"]["position_error_m"], .007)
        self.assertLessEqual(info["goal_ik_validation"]["orientation_error_rad"], .08)


if __name__ == "__main__":
    unittest.main()
