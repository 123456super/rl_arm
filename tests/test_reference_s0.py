"""Regression tests for the sole supported FTP4RM reference-PPO path."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pybullet as p
import torch

from rl_risk_sac.algorithms.networks import ObstacleSetEncoder
from rl_risk_sac.algorithms.thesis_ppo import ThesisPPOAgent
from rl_risk_sac.envs.thesis_homotopy_env_reduced_observation import (
    ThesisHomotopyEnv,
)
from rl_risk_sac.envs.joint_motion_controller import BoundedJointMotionController
from rl_risk_sac.envs.thesis_obstacles import _swept_obb_envelope
from rl_risk_sac.envs.thesis_static_obstacles import ObstacleSamplingError
from rl_risk_sac.prediction.obstacle_lstm import CHECKPOINT_FORMAT, ObstacleMotionLSTM
from rl_risk_sac.tasks.reference_reaching import (
    DEFAULT_MAX_OBSTACLES,
    OBSTACLE_CLEARANCE_SCALE_M,
    OBSTACLE_FEATURE_DIM,
    REFERENCE_BASE_OBSERVATION_DIM,
    REFERENCE_PPO_OBSERVATION_DIM,
    closest_points_segment_obb,
    reference_ppo_reward,
    residual_action_penalty,
    compose_base_residual_action,
    residual_handoff_authority,
)
from rl_risk_sac.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class ReferenceS0Tests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/experiments/paper_full.yaml")
        self.config["device"] = "cpu"
        self.config["device_selection"]["print_summary"] = False
        self.env = ThesisHomotopyEnv(self.config)
        self.env.configure_episode("none")

    def tearDown(self):
        self.env.close()

    def test_reset_uses_fixed_orientation_and_empty_obstacle_set(self):
        observation, info = self.env.reset(seed=11001)
        self.assertEqual(observation.shape, (REFERENCE_PPO_OBSERVATION_DIM,))
        self.assertTrue(self.env.observation_space.contains(observation))
        np.testing.assert_allclose(
            info["goal_quaternion"], self.env.fixed_goal_quaternion, atol=1.0e-7
        )
        distances = np.asarray(info["reference_point_distances"])
        np.testing.assert_allclose(
            distances,
            np.full(3, info["goal_error_norm"]),
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            observation[22:25], info["position_error_vector"], atol=1.0e-7
        )
        np.testing.assert_allclose(
            observation[25:28], info["orientation_error_vector"], atol=1.0e-7
        )
        np.testing.assert_array_equal(observation[28:34], 0.0)
        np.testing.assert_allclose(
            np.linalg.norm(observation[22:25]), info["goal_error_norm"], atol=1.0e-7
        )
        np.testing.assert_allclose(
            np.linalg.norm(observation[25:28]), info["orientation_error_norm"], atol=1.0e-7
        )
        feature_end = (
            REFERENCE_BASE_OBSERVATION_DIM
            + DEFAULT_MAX_OBSTACLES * OBSTACLE_FEATURE_DIM
        )
        np.testing.assert_array_equal(
            observation[REFERENCE_BASE_OBSERVATION_DIM:feature_end], 0.0
        )
        np.testing.assert_array_equal(observation[feature_end:], 0.0)

    def test_action_is_a_joint_angle_increment(self):
        self.env.reset(seed=11001)
        q_before, _ = self.env.robot.joint_state()
        # Values above the historical 0.25 rad policy limit must pass through
        # unchanged; only absolute robot joint-position limits remain.
        delta = np.asarray([0.35, -0.35, 0.30, -0.30, 0.28, -0.28], dtype=np.float32)
        _, _, _, _, _, info = self.env.step(delta)
        expected = np.clip(
            q_before + delta,
            np.maximum(self.env.robot.joint_lower_limits, -self.env.action_limit),
            np.minimum(self.env.robot.joint_upper_limits, self.env.action_limit),
        )
        np.testing.assert_allclose(info["policy_joint_delta"], delta, atol=1.0e-7)
        np.testing.assert_allclose(info["target_joint_positions"], expected, atol=1.0e-7)
        # The restored reference-PPO route intentionally uses Bullet
        # POSITION_CONTROL and therefore does not apply the bounded CB31
        # bounded velocity controller.
        self.assertFalse(self.env.bounded_joint_motion_enabled)
        self.assertEqual(info["executed_substeps"], 100)
        np.testing.assert_array_equal(info["commanded_joint_velocity"], 0.0)
        self.assertEqual(info["joint_velocity_limiter_fraction"], 0.0)
        self.assertEqual(info["joint_acceleration_limiter_fraction"], 0.0)
        self.assertEqual(info["tcp_velocity_limiter_fraction"], 0.0)
        self.assertLess(
            float(np.max(np.abs(info["commanded_joint_delta"]))),
            float(np.max(np.abs(delta))),
        )

    def test_motion_controller_accelerates_and_stops_without_overshoot(self):
        controller = BoundedJointMotionController(
            joint_velocity_limits=np.full(6, 0.25),
            joint_acceleration_limits=np.full(6, 0.5),
            position_gain=4.0,
            position_tolerance=5.0e-4,
        )
        q = np.zeros(6)
        goal = np.full(6, 0.1)
        previous = np.zeros(6)
        for _ in range(500):
            command = controller.command(q, goal, 1.0 / 240.0)
            self.assertTrue(np.all(np.abs(command.velocity) <= 0.25 + 1.0e-12))
            self.assertTrue(np.all(
                np.abs(command.velocity - previous) <= 0.5 / 240.0 + 1.0e-7
            ))
            q = q + command.velocity / 240.0
            previous = command.velocity
        self.assertTrue(np.all(q <= goal + 5.0e-4))
        self.assertTrue(np.all(np.abs(goal - q) <= 5.0e-4))

    def test_cb31_scheduler_updates_at_125hz_and_holds_at_240hz(self):
        config = load_config(
            ROOT / "configs/experiments/paper_s0_cb31_10deg_horizon150.yaml"
        )
        config["device"] = "cpu"
        config["device_selection"]["print_summary"] = False
        env = ThesisHomotopyEnv(config)
        try:
            env.configure_episode("none")
            env.reset(seed=11001)
            _, _, _, _, _, info = env.step(
                np.full(6, 0.02, dtype=np.float32)
            )
            self.assertEqual(info["executed_substeps"], 96)
            self.assertEqual(info["controller_updates"], 50)
            self.assertEqual(info["controller_updates_expected"], 50)
            self.assertAlmostEqual(info["real_control_frequency_hz"], 125.0)
            self.assertAlmostEqual(info["physics_frequency_hz"], 240.0)
            self.assertLessEqual(
                info["joint_speed_peak_rad_s"], np.deg2rad(10.0) + 2.0e-3
            )
        finally:
            env.close()

    def test_static_obstacle_uses_six_clearances_and_nearest_direction(self):
        self.env.configure_episode("static")
        observation, info = self.env.reset(seed=11001)
        np.testing.assert_allclose(
            self.env.obstacle_half_extents,
            np.full(3, 0.075, dtype=np.float32),
            atol=0.0,
        )
        start = REFERENCE_BASE_OBSERVATION_DIM
        rows = observation[
            start:start + DEFAULT_MAX_OBSTACLES * OBSTACLE_FEATURE_DIM
        ].reshape(DEFAULT_MAX_OBSTACLES, OBSTACLE_FEATURE_DIM)
        mask = observation[-DEFAULT_MAX_OBSTACLES:]
        np.testing.assert_array_equal(mask, [1.0] + [0.0] * 7)
        geometry = self.env._geometry()
        np.testing.assert_allclose(
            rows[0, :6],
            geometry.distances / OBSTACLE_CLEARANCE_SCALE_M,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            rows[0, 6:], geometry.nearest_direction, atol=1.0e-6
        )
        self.assertAlmostEqual(float(np.linalg.norm(rows[0, 6:])), 1.0, places=6)
        np.testing.assert_array_equal(rows[1:], 0.0)

        collision_data = p.getCollisionShapeData(
            self.env.obstacle_id,
            -1,
            physicsClientId=self.env.client_id,
        )
        self.assertEqual(collision_data[0][2], p.GEOM_BOX)

    def test_reset_resamples_goal_after_infeasible_obstacle_scene(self):
        self.env.configure_episode("static")
        create_obstacle = self.env._create_obstacle
        call_count = 0

        def fail_first_scene():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                self.env.obstacle_sampling_proposals = 16000
                self.env.obstacle_sampling_time_s = 1.0
                raise ObstacleSamplingError("forced infeasible scene")
            return create_obstacle()

        with patch.object(
            self.env, "_create_obstacle", side_effect=fail_first_scene
        ):
            _, info = self.env.reset(seed=11001)

        self.assertEqual(call_count, 2)
        self.assertEqual(info["scene_sampling_attempts"], 2)
        self.assertGreaterEqual(info["obstacle_sampling_proposals"], 16000)
        self.assertGreaterEqual(info["obstacle_sampling_time_s"], 1.0)

    def test_segment_obb_distance_and_direction_are_exact(self):
        segment_point, box_point, direction, distance, fraction = (
            closest_points_segment_obb(
                np.asarray([-2.0, 2.0, 0.0]),
                np.asarray([2.0, 2.0, 0.0]),
                np.zeros(3),
                np.ones(3),
                np.eye(3),
            )
        )
        self.assertAlmostEqual(distance, 1.0, places=7)
        self.assertGreaterEqual(fraction, 0.25)
        self.assertLessEqual(fraction, 0.75)
        np.testing.assert_allclose(segment_point[1], 2.0, atol=1.0e-7)
        np.testing.assert_allclose(box_point[1], 1.0, atol=1.0e-7)
        np.testing.assert_allclose(direction, [0.0, -1.0, 0.0], atol=1.0e-7)

    def test_swept_obb_envelope_contains_all_translated_boxes(self):
        quaternion = np.asarray(
            p.getQuaternionFromEuler([0.0, 0.0, 0.63]), dtype=np.float32
        )
        positions = np.asarray(
            [[0.1, -0.2, 0.3], [0.25, 0.05, 0.35], [0.0, 0.2, 0.25]],
            dtype=np.float32,
        )
        original_half_extents = np.asarray([0.05, 0.08, 0.04], dtype=np.float32)
        center, half_extents = _swept_obb_envelope(
            positions, original_half_extents, quaternion
        )
        rotation = np.asarray(
            p.getMatrixFromQuaternion(quaternion), dtype=np.float64
        ).reshape(3, 3)
        local_centres = (positions - center) @ rotation
        self.assertTrue(np.all(
            np.abs(local_centres) + original_half_extents
            <= half_extents + 1.0e-6
        ))

    def test_s2_lstm_prediction_is_one_swept_set_element(self):
        config = load_config(
            ROOT / "configs/experiments/paper_s2_lstm_sweep_single.yaml"
        )
        config["device"] = "cpu"
        config["device_selection"]["print_summary"] = False
        model_config = {
            "input_size": 7, "hidden_size": 8, "num_layers": 1,
            "horizon_steps": 3,
        }
        model = ObstacleMotionLSTM(**model_config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.head[-1].bias.copy_(torch.tensor([
                0.01, 0.0, 0.0, 0.02, 0.0, 0.0, 0.03, 0.0, 0.0,
            ]))
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "lstm.pt"
            torch.save({
                "format": CHECKPOINT_FORMAT,
                "model_config": model_config,
                "model_state_dict": model.state_dict(),
                "history_length": 10,
                "history_hz": 30.0,
                "horizon_times_s": [0.4 / 3.0, 0.8 / 3.0, 0.4],
                "input_scale": [1.0] * 7,
                "output_scale": 1.0,
            }, checkpoint)
            config["thesis"]["dynamic_prediction"]["checkpoint"] = str(checkpoint)
            env = ThesisHomotopyEnv(config)
            try:
                env.simple_reaching_mode = False
                env.blocking_obstacle_mode = True
                env.configure_episode("dynamic")
                observation, reset_info = env.reset(seed=11001)
                self.assertTrue(reset_info["dynamic_prediction_enabled"])
                self.assertEqual(
                    reset_info["dynamic_predicted_positions"].shape, (1, 4, 3)
                )
                self.assertEqual(
                    len(env.dynamic_position_history[0]),
                    env.dynamic_history_length,
                )
                self.assertGreater(
                    float(np.linalg.norm(reset_info["dynamic_tracked_velocities"][0])),
                    1.0e-4,
                )
                self.assertTrue(np.any(
                    reset_info["dynamic_envelope_half_extents"][0]
                    > env.dynamic_obstacle_half_extents + 1.0e-5
                ))
                _, _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
                predicted = info["dynamic_predicted_positions"][0]
                tracked = info["dynamic_tracked_velocities"][0]
                self.assertTrue(np.isfinite(predicted).all())
                self.assertGreater(
                    float(np.linalg.norm(predicted[-1] - predicted[0])), 1.0e-5
                )
                self.assertGreater(float(np.linalg.norm(tracked)), 1.0e-4)
                self.assertTrue(np.all(
                    info["dynamic_envelope_half_extents"][0]
                    >= env.dynamic_obstacle_half_extents
                ))
                feature_start = REFERENCE_BASE_OBSERVATION_DIM
                rows = observation[
                    feature_start:
                    feature_start + DEFAULT_MAX_OBSTACLES * OBSTACLE_FEATURE_DIM
                ].reshape(DEFAULT_MAX_OBSTACLES, OBSTACLE_FEATURE_DIM)
                mask = observation[-DEFAULT_MAX_OBSTACLES:]
                np.testing.assert_array_equal(mask, [1.0] + [0.0] * 7)
                self.assertTrue(np.isfinite(rows[0]).all())
                np.testing.assert_array_equal(rows[1:], 0.0)
            finally:
                env.close()

    def test_hybrid_scene_moves_only_dynamic_obstacle(self):
        config = load_config(
            ROOT / "configs/experiments/paper_s2_lstm_sweep_hybrid.yaml"
        )
        config["device"] = "cpu"
        config["device_selection"]["print_summary"] = False
        config["thesis"]["dynamic_prediction"]["enabled"] = False
        config["thesis"]["dynamic_prediction"]["checkpoint"] = None
        env = ThesisHomotopyEnv(config)
        try:
            env.configure_episode("hybrid")
            observation, info = env.reset(seed=11001)
            self.assertEqual(info["obstacle_count"], 2)
            self.assertEqual(info["dynamic_obstacle_count"], 1)
            self.assertEqual(info["static_obstacle_count"], 1)
            np.testing.assert_array_equal(
                env.obstacle_dynamic_mask, [True, False]
            )
            mask = observation[-DEFAULT_MAX_OBSTACLES:]
            np.testing.assert_array_equal(mask, [1.0, 1.0] + [0.0] * 6)
            before = env.obstacle_positions.copy()
            _, _, _, _, _, step_info = env.step(
                np.zeros(6, dtype=np.float32)
            )
            motion = np.linalg.norm(env.obstacle_positions - before, axis=1)
            self.assertGreater(float(motion[0]), 1.0e-5)
            self.assertEqual(float(motion[1]), 0.0)
            self.assertEqual(
                step_info["dynamic_predicted_positions"].shape, (1, 1, 3)
            )
        finally:
            env.close()

    def test_obstacle_encoder_is_permutation_invariant_and_empty_is_zero(self):
        torch.manual_seed(7)
        encoder = ObstacleSetEncoder(9, 8, embedding_dim=16, output_dim=12)
        features = torch.randn(2, 8, 9)
        mask = torch.tensor(
            [[1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=torch.float32,
        )
        permutation = torch.tensor([2, 0, 1, 7, 5, 3, 6, 4])
        encoded = encoder(features, mask)
        permuted = encoder(features[:, permutation], mask[:, permutation])
        torch.testing.assert_close(encoded, permuted)
        torch.testing.assert_close(encoded[1], torch.zeros(12))

    def test_presence_switch_bypasses_residual_only_for_empty_set(self):
        config = deepcopy(self.config)
        config["thesis"]["stage"] = "s1"
        agent = ThesisPPOAgent(REFERENCE_PPO_OBSERVATION_DIM, 6, config)
        observation = torch.randn(4, REFERENCE_PPO_OBSERVATION_DIM)
        observation[:, -DEFAULT_MAX_OBSTACLES:] = 0.0
        with torch.no_grad():
            agent.residual_actor[-1].weight.zero_()
            agent.residual_actor[-1].bias.fill_(0.40)
        empty_mean, empty_base, empty_residual = agent._policy_components(observation)
        torch.testing.assert_close(
            empty_residual, torch.zeros_like(empty_residual), rtol=0, atol=0
        )
        torch.testing.assert_close(empty_mean, empty_base, rtol=0, atol=0)

        obstacle_observation = observation.clone()
        obstacle_observation[:, -DEFAULT_MAX_OBSTACLES] = 1.0
        mean, base, residual = agent._policy_components(obstacle_observation)
        torch.testing.assert_close(residual, torch.full_like(residual, 0.40))
        torch.testing.assert_close(mean, base + residual)
        self.assertGreater(float(mean.max().detach()), 0.25)
        self.assertFalse(any(p.requires_grad for p in agent.actor.parameters()))
        self.assertTrue(any(p.requires_grad for p in agent.residual_actor.parameters()))

        agent.actor_optimizer.zero_grad(set_to_none=True)
        mean.square().mean().backward()
        self.assertFalse(any(p.grad is not None for p in agent.actor.parameters()))
        self.assertGreater(float(agent.residual_actor[-1].bias.grad.norm()), 0.0)

    def test_s0_base_override_preserves_avoidance_modules(self):
        s0_config = deepcopy(self.config)
        s0_config["thesis"]["stage"] = "s0"
        s0_agent = ThesisPPOAgent(REFERENCE_PPO_OBSERVATION_DIM, 6, s0_config)
        with torch.no_grad():
            for parameter in s0_agent.actor.parameters():
                parameter.fill_(0.0123)

        s2_config = deepcopy(self.config)
        s2_config["thesis"]["stage"] = "s2"
        s2_agent = ThesisPPOAgent(REFERENCE_PPO_OBSERVATION_DIM, 6, s2_config)
        residual_before = {
            key: value.detach().clone()
            for key, value in s2_agent.residual_actor.state_dict().items()
        }
        encoder_before = {
            key: value.detach().clone()
            for key, value in s2_agent.actor_obstacle_encoder.state_dict().items()
        }
        value_before = {
            key: value.detach().clone()
            for key, value in s2_agent.value.state_dict().items()
        }

        s2_agent.replace_base_actor_from_s0_checkpoint(s0_agent.state_dict())

        for key, value in s0_agent.actor.state_dict().items():
            torch.testing.assert_close(s2_agent.actor.state_dict()[key], value)
        for key, value in residual_before.items():
            torch.testing.assert_close(
                s2_agent.residual_actor.state_dict()[key], value
            )
        for key, value in encoder_before.items():
            torch.testing.assert_close(
                s2_agent.actor_obstacle_encoder.state_dict()[key], value
            )
        for key, value in value_before.items():
            torch.testing.assert_close(s2_agent.value.state_dict()[key], value)
        self.assertFalse(any(
            parameter.requires_grad for parameter in s2_agent.actor.parameters()
        ))

        with self.assertRaisesRegex(ValueError, "S0 checkpoint"):
            s2_agent.replace_base_actor_from_s0_checkpoint(s2_agent.state_dict())

    def test_residual_has_no_clearance_gate_or_output_saturation(self):
        config = deepcopy(self.config)
        config["thesis"]["stage"] = "s1"
        agent = ThesisPPOAgent(REFERENCE_PPO_OBSERVATION_DIM, 6, config)
        observations = torch.zeros(4, REFERENCE_PPO_OBSERVATION_DIM)
        feature_start = REFERENCE_BASE_OBSERVATION_DIM
        mask_start = feature_start + DEFAULT_MAX_OBSTACLES * OBSTACLE_FEATURE_DIM
        observations[:, mask_start] = 1.0
        observations[0, feature_start:feature_start + 6] = 10.0
        observations[1, feature_start:feature_start + 6] = 0.12
        observations[2, feature_start:feature_start + 6] = 0.05
        observations[3, feature_start:feature_start + 6] = -0.01
        with torch.no_grad():
            agent.residual_actor[-1].weight.zero_()
            agent.residual_actor[-1].bias.fill_(1.25)
        mean, base, residual = agent._policy_components(observations)
        torch.testing.assert_close(residual, torch.full_like(residual, 1.25))
        torch.testing.assert_close(mean, base + residual)

    def test_blocking_static_obstacle_intersects_nominal_tcp_path(self):
        self.env.simple_reaching_mode = False
        self.env.blocking_obstacle_mode = True
        self.env.configure_episode("static")
        self.env.reset(seed=11001)
        initial_q, initial_qdot = self.env.robot.joint_state_exact()
        minimum_robot_clearance = float("inf")
        minimum_tcp_center_error = float("inf")
        try:
            for fraction in np.linspace(0.20, 0.80, 13):
                self.env._set_joint_state(
                    initial_q
                    + float(fraction)
                    * (self.env.goal_joint_positions - initial_q)
                )
                minimum_robot_clearance = min(
                    minimum_robot_clearance, self.env._geometry().distance_min
                )
                tcp_position, _ = self.env.robot.end_effector_pose()
                minimum_tcp_center_error = min(
                    minimum_tcp_center_error,
                    float(np.linalg.norm(
                        tcp_position - self.env.obstacle_position
                    )),
                )
        finally:
            self.env._set_joint_state(initial_q, initial_qdot)
        self.assertLessEqual(minimum_robot_clearance, 0.0)
        self.assertLessEqual(minimum_tcp_center_error, 1.0e-6)

    def test_fast_reset_clearance_matches_full_obstacle_geometry(self):
        self.env.simple_reaching_mode = False
        self.env.blocking_obstacle_mode = True
        self.env.configure_episode("static")
        self.env.reset(seed=11001)
        capsules = self.env.robot.capsules()
        for offset in (
            np.zeros(3, dtype=np.float32),
            np.asarray([0.03, -0.02, 0.01], dtype=np.float32),
            np.asarray([-0.04, 0.01, 0.05], dtype=np.float32),
        ):
            position = self.env.obstacle_position + offset
            fast = self.env._minimum_obstacle_clearance_for(capsules, position)
            full = self.env._obstacle_geometry_for(
                capsules, position, np.zeros(3, dtype=np.float32)
            ).distance_min
            self.assertAlmostEqual(fast, full, places=7)

    def test_s1_geometry_diagnostics_are_decimated_but_contacts_remain_physical(self):
        self.env.simple_reaching_mode = False
        self.env.configure_episode("static")
        self.env.reset(seed=11001)
        with (
            patch.object(
                self.env,
                "_obstacle_geometries",
                wraps=self.env._obstacle_geometries,
            ) as obstacle_geometry,
            patch.object(
                self.env, "_self_geometry", wraps=self.env._self_geometry
            ) as self_geometry,
            patch.object(
                self.env, "_collision_events", wraps=self.env._collision_events
            ) as collision_events,
        ):
            self.env.step(np.zeros(6, dtype=np.float32))
        # Intermediate physical contact detection remains at 240 Hz, while
        # OBB/capsule geometry is computed once for the exact action endpoint.
        self.assertEqual(obstacle_geometry.call_count, 1)
        self.assertEqual(self_geometry.call_count, 1)
        self.assertEqual(collision_events.call_count, self.env.substeps)

    def test_robot_wide_contact_query_preserves_obstacle_detection(self):
        self.env.simple_reaching_mode = False
        self.env.configure_episode("static")
        self.env.reset(seed=11001)
        capsule = self.env.robot.capsules()[3]
        contact_position = 0.5 * (capsule.start + capsule.end)
        p.resetBasePositionAndOrientation(
            self.env.obstacle_id,
            contact_position.tolist(),
            self.env.obstacle_quaternion.tolist(),
            physicsClientId=self.env.client_id,
        )
        p.performCollisionDetection(physicsClientId=self.env.client_id)
        filtered_contacts = p.getContactPoints(
            bodyA=self.env.robot_id,
            bodyB=self.env.obstacle_id,
            physicsClientId=self.env.client_id,
        )
        self.assertTrue(any(point[8] <= 0.0 for point in filtered_contacts))
        self.assertTrue(self.env._collision_events()["obstacle_collision"])

    def test_reference_reward_matches_closed_form(self):
        reward, fields = reference_ppo_reward(
            reference_error=0.2,
            obstacle_collision=True,
            obstacle_clearances=(),
            obstacle_collision_penalty=20.0,
        )
        expected_dense = -0.001 * 0.2**2 - np.log(0.2**2 + 1.0e-4)
        self.assertAlmostEqual(fields["r_goal"], expected_dense)
        self.assertAlmostEqual(reward, expected_dense - 20.0)

    def test_reference_avoidance_reward_uses_linear_clearance_and_fixed_collisions(self):
        reward, fields = reference_ppo_reward(
            reference_error=0.2,
            obstacle_collision=True,
            self_collision=True,
            obstacle_clearances=(0.08, 0.06, 0.04, 0.02, 0.0, -0.01),
        )
        expected_goal = -0.001 * 0.2**2 - np.log(0.2**2 + 1.0e-4)
        expected_proximity = 0.0 + 0.25 + 0.5 + 0.75 + 1.0 + 1.0
        self.assertAlmostEqual(fields["c_proximity"], expected_proximity)
        self.assertAlmostEqual(fields["reference_obstacle_penalty"], expected_proximity)
        self.assertAlmostEqual(fields["reference_obstacle_collision_penalty"], 20.0)
        self.assertAlmostEqual(fields["reference_self_collision_penalty"], 10.0)
        self.assertAlmostEqual(reward, expected_goal - expected_proximity - 30.0)

    def test_residual_penalty_is_goal_and_clearance_conditioned(self):
        residual = np.asarray([0.1, -0.2, 0.0, 0.0, 0.0, 0.0])
        far_goal_cost, far_goal_weight = residual_action_penalty(
            residual, 0.03, 0.12
        )
        unsafe_cost, unsafe_weight = residual_action_penalty(
            residual, 0.02, 0.08
        )
        middle_cost, middle_weight = residual_action_penalty(
            residual, 0.025, 0.10
        )
        near_safe_cost, near_safe_weight = residual_action_penalty(
            residual, 0.02, 0.12
        )
        self.assertEqual((far_goal_cost, far_goal_weight), (0.0, 0.0))
        self.assertEqual((unsafe_cost, unsafe_weight), (0.0, 0.0))
        self.assertAlmostEqual(middle_weight, 0.25)
        self.assertAlmostEqual(middle_cost, 0.625)
        self.assertAlmostEqual(near_safe_weight, 1.0)
        self.assertAlmostEqual(near_safe_cost, 2.5)

    def test_residual_handoff_requires_both_near_goal_and_safe_clearance(self):
        self.assertEqual(residual_handoff_authority(0.025, 0.10), 0.0)
        self.assertEqual(residual_handoff_authority(0.035, 0.10), 1.0)
        self.assertEqual(residual_handoff_authority(0.025, 0.08), 1.0)
        self.assertAlmostEqual(
            residual_handoff_authority(0.030, 0.09), 0.5
        )

    def test_deployment_action_hands_control_to_base_inside_25_mm(self):
        base = np.asarray([[0.1, -0.2], [0.3, -0.4]], dtype=np.float32)
        residual = np.asarray([[0.5, 0.6], [-0.1, 0.2]], dtype=np.float32)
        action, authority = compose_base_residual_action(
            base, residual, np.asarray([0.025, 0.025001])
        )
        np.testing.assert_allclose(authority, [0.0, 1.0])
        np.testing.assert_allclose(action[0], base[0])
        np.testing.assert_allclose(action[1], base[1] + residual[1])

    def test_deployment_agent_uses_shared_handoff_composer(self):
        config = deepcopy(self.config)
        config["thesis"]["stage"] = "s1"
        agent = ThesisPPOAgent(REFERENCE_PPO_OBSERVATION_DIM, 6, config)
        observation = np.zeros((2, REFERENCE_PPO_OBSERVATION_DIM), dtype=np.float32)
        with patch.object(agent, "act") as act:
            total = np.full((2, 6), 0.3, dtype=np.float32)
            residual = np.full((2, 6), 0.2, dtype=np.float32)
            act.return_value = (
                total, np.zeros(2), np.zeros(2),
                {"residual_actions": residual},
            )
            executed, _logp, _values, policy = agent.act_for_deployment(
                observation, np.asarray([0.01, 0.10], dtype=np.float32)
            )
        np.testing.assert_allclose(executed[0], 0.1)
        np.testing.assert_allclose(executed[1], 0.3)
        np.testing.assert_allclose(policy["residual_authority"], [0.0, 1.0])

    def test_s1_obstacle_collision_terminates_immediately(self):
        self.env.simple_reaching_mode = False
        self.env.terminate_on_collision = True
        self.env.configure_episode("static")
        self.env.reset(seed=11001)
        collision = {
            "obstacle_collision": True,
            "self_collision": False,
            "environment_collision": False,
        }
        with patch.object(self.env, "_collision_events", return_value=collision):
            _, _, _, terminated, truncated, info = self.env.step(
                np.zeros(6, dtype=np.float32)
            )
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "obstacle_collision")
        self.assertEqual(info["executed_substeps"], 1)

    def test_nonterminal_obstacle_collision_keeps_full_control_period(self):
        self.env.simple_reaching_mode = False
        self.env.terminate_on_collision = False
        self.env.configure_episode("static")
        self.env.reset(seed=11001)
        collision = {
            "obstacle_collision": True,
            "self_collision": False,
            "environment_collision": False,
        }
        with patch.object(self.env, "_collision_events", return_value=collision):
            _, _, _, terminated, truncated, info = self.env.step(
                np.zeros(6, dtype=np.float32)
            )
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["executed_substeps"], self.env.substeps)
        self.assertTrue(info["obstacle_collision"])

    def test_s0_collision_keeps_reference_nonterminal_semantics(self):
        self.env.simple_reaching_mode = True
        self.env.terminate_on_collision = True
        self.env.configure_episode("none")
        self.env.reset(seed=11001)
        collision = {
            "obstacle_collision": False,
            "self_collision": True,
            "environment_collision": False,
        }
        with patch.object(self.env, "_collision_events", return_value=collision):
            _, _, _, terminated, truncated, info = self.env.step(
                np.zeros(6, dtype=np.float32)
            )
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "running")

    def test_non_reference_mode_is_rejected(self):
        config = deepcopy(self.config)
        config["thesis"]["reward"]["reward_mode"] = "removed_mode"
        with self.assertRaisesRegex(ValueError, "only"):
            ThesisPPOAgent(REFERENCE_PPO_OBSERVATION_DIM, 6, config)


if __name__ == "__main__":
    unittest.main()
