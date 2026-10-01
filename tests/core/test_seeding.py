from __future__ import annotations

import pytest

from rl_risk_sac.utils.seeding import derive_episode_seed


def test_episode_seeds_do_not_overlap_across_consecutive_eval_seeds() -> None:
    seeds = {
        derive_episode_seed(base_seed, episode)
        for base_seed in (1004, 1005, 1006)
        for episode in range(100)
    }
    assert len(seeds) == 300


def test_episode_seed_is_deterministic() -> None:
    assert derive_episode_seed(1004, 17) == derive_episode_seed(1004, 17)
    assert derive_episode_seed(1004, 17) != derive_episode_seed(1005, 16)


@pytest.mark.parametrize(("base_seed", "episode"), [(-1, 0), (0, -1)])
def test_episode_seed_rejects_negative_inputs(base_seed: int, episode: int) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        derive_episode_seed(base_seed, episode)
