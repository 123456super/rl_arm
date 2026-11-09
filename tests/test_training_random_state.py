from __future__ import annotations

import random
import unittest

import numpy as np
import torch

from scripts.core.train_thesis_ppo import capture_random_state, restore_random_state


class FakePool:
    def __init__(self) -> None:
        self.states = {0: {"state": 11}, 1: {"state": 22}}

    def __len__(self) -> int:
        return 2

    def rng_states(self):
        return self.states.copy()

    def set_rng_states(self, states):
        self.states = states.copy()


class TrainingRandomStateTests(unittest.TestCase):
    def test_all_random_streams_round_trip(self):
        pool = FakePool()
        random.seed(17)
        np.random.seed(23)
        torch.manual_seed(31)
        state = capture_random_state(pool)
        expected = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
        )
        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)
        pool.states = {0: {"state": -1}, 1: {"state": -1}}
        restore_random_state(state, pool)
        actual = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
        )
        self.assertEqual(actual, expected)
        self.assertEqual(pool.states, {0: {"state": 11}, 1: {"state": 22}})

    def test_environment_count_mismatch_is_rejected(self):
        pool = FakePool()
        state = capture_random_state(pool)
        state["environments"] = {0: {"state": 11}}
        with self.assertRaisesRegex(ValueError, "num-envs"):
            restore_random_state(state, pool)


if __name__ == "__main__":
    unittest.main()
