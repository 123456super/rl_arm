from __future__ import annotations
import copy
import csv
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from rl_risk_sac.envs.thesis_homotopy_env import EpisodeContract
from rl_risk_sac.tasks.local_avoidance import actor_digest
from rl_risk_sac.tasks.reaching_handoff import (
    BANK_PROTOCOL, TRAIN_PROTOCOL, RecoveryReplay, load_reaching_override,
    recovery_state, sample_digest, validate_handoff_bank, recovery_config,
)
from rl_risk_sac.utils.config import load_config
from rl_risk_sac.utils.training_logs import CsvLog
from rl_risk_sac.algorithms.dual_sac import ReachingSACAgent
from rl_risk_sac.algorithms.replay_buffer import Batch
from scripts.core.train_reaching_handoff import progress_fieldnames


def sample():
    state=dict(q=np.arange(6)*.1,qdot=np.ones(6)*.07,goal_position=np.array([.2,.3,.4]),
               goal_quaternion=np.array([0.,0.,0.,1.]),obstacles=[dict(identity=1)],
               predictor_history={1:[2]},step_count=73,physics_step_count=876,success_hold_count=2,
               obstacle_contact_seen=False,contract=EpisodeContract(terminate_on_success=False))
    result=dict(state=state,source_seed=123,source_scene_id="scene",handoff_index=1,
                clearance_m=.08,self_clearance_m=.01)
    result["sample_id"]=sample_digest(state,"scene",1)
    return result


def bank():
    return dict(protocol=BANK_PROTOCOL,split="train",samples=[sample()],excluded_trace_seeds=[789],source_checkpoint_sha256="source")


def test_recovery_reset_preserves_motion_goal_and_physics_clock_without_touching_source():
    original=sample()["state"]
    restored=recovery_state(original)
    for key in ("q","qdot","goal_position","goal_quaternion"):
        np.testing.assert_array_equal(restored[key],original[key])
    assert restored["physics_step_count"]==876
    assert restored["step_count"]==restored["success_hold_count"]==0
    assert restored["obstacles"]==[] and restored["predictor_history"]=={}
    assert restored["contract"].terminate_on_success
    assert original["step_count"]==73 and original["obstacles"]
    assert not original["contract"].terminate_on_success


@pytest.mark.parametrize("bad",["split","source","overlap","duplicate","state","clearance"])
def test_bad_handoff_banks_are_rejected(bad):
    b=bank()
    if bad=="split": b["split"]="test"
    if bad=="source": b["source_checkpoint_sha256"]="other"
    if bad=="overlap": b["excluded_trace_seeds"]=[123]
    if bad=="duplicate": b["samples"].append(copy.deepcopy(b["samples"][0]))
    if bad=="state": b["samples"][0]["state"]["q"][0]+=.1
    if bad=="clearance": b["samples"][0]["clearance_m"]=.01
    with pytest.raises(ValueError): validate_handoff_bank(b,split=("train",),source_sha="source")


def test_valid_bank_and_obstacle_free_configuration_keep_s0_objective():
    assert len(validate_handoff_bank(bank(),split=("train",),source_sha="source"))==1
    cfg=load_config("configs/experiments/paper_local_avoidance_geometry.yaml")
    result=recovery_config(cfg)
    assert result["thesis"]["reward"]==cfg["thesis"]["reward"]
    assert not result["thesis"]["obstacles"]["enabled"]
    assert not result["local_avoidance"]["enabled"] and not result["dual_sac"]["enabled"]
    assert result["sac"]["actor_lr"]==1e-5
    assert cfg["local_avoidance"]["enabled"]


def test_recovery_replay_roundtrip_and_time_limit_bootstrap():
    replay=RecoveryReplay(8,"cpu",7)
    for i in range(5): replay.add(np.full(241,i),np.zeros(6),i,np.full(241,i+1),False)
    state=copy.deepcopy(replay.state_dict())
    restored=RecoveryReplay(8,"cpu",99);restored.load_state_dict(state)
    a,b=replay.sample(5),restored.sample(5)
    torch.testing.assert_close(a.observations,b.observations)
    assert not a.dones.any()  # Time limit resets do not become Bellman terminals.
    with pytest.raises(ValueError): restored.load_state_dict(dict(state,protocol="local_passage"))
    with pytest.raises(ValueError): replay.add(np.zeros(268),np.zeros(6),0,np.zeros(268),False)


@pytest.mark.parametrize("start_steps", [0, 50000])
def test_progress_accepts_learning_metrics_after_a_row_without_updates(tmp_path, start_steps):
    # Reproduce the real startup/resume boundary: the first progress row has
    # no collected metrics; the next contains all metrics from an SAC update.
    cfg = recovery_config(load_config("configs/experiments/paper_local_avoidance_geometry.yaml"))
    cfg["device"] = "cpu"
    cfg["sac"]["hidden_dims"] = [16, 16]
    agent = ReachingSACAgent(241, 6, cfg)
    path = tmp_path / "progress.csv"
    fields = progress_fieldnames(agent)
    assert len(fields) == len(set(fields))
    with CsvLog(path, fields) as progress:
        progress.writerow(dict(steps=start_steps + 1000, updates=0, replay_size=1000))
        batch = Batch(torch.randn(8, 241), torch.zeros(8, 6), torch.zeros(8, 1),
                      torch.zeros(8, 1), torch.randn(8, 241), torch.zeros(8, 1))
        metrics = agent.update(batch, reference_observations=batch)
        assert metrics and all(np.isfinite(value) for value in metrics.values())
        progress.writerow(dict(steps=start_steps + 2000, updates=1, replay_size=2000, **metrics))
        progress.writerow(dict(steps=start_steps + 3000, updates=2, replay_size=3000))
    with path.open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == list(fields)
    assert len(rows) == 3 and all(None not in row for row in rows)
    for name, value in metrics.items():
        assert rows[0][name] == rows[2][name] == ""
        assert float(rows[1][name]) == value


@pytest.mark.parametrize("bad",[None,"source","overlap","avoidance"])
def test_joint_override_is_bound_to_frozen_avoidance_and_disjoint_training(bad):
    avoid=torch.nn.Linear(2,2)
    reach=torch.nn.Linear(2,2)
    loaded=[]
    agent=SimpleNamespace(avoidance=SimpleNamespace(actor=avoid),
        reaching=SimpleNamespace(actor=reach,load_state_dict=lambda state,actor_only:loaded.append(state)))
    state=dict(protocol=TRAIN_PROTOCOL,source_checkpoint_sha256="source",training_trace_seeds=[123],
               frozen_avoidance_actor_sha256=actor_digest(avoid),agent={"new":"weights"})
    if bad=="source": state["source_checkpoint_sha256"]="other"
    if bad=="avoidance": state["frozen_avoidance_actor_sha256"]="other"
    if bad:
        with pytest.raises(ValueError): load_reaching_override(agent,state,"source",[123] if bad=="overlap" else [789])
        assert loaded==[]
    else:
        before=actor_digest(avoid)
        load_reaching_override(agent,state,"source",[789])
        assert loaded==[{"new":"weights"}] and actor_digest(avoid)==before
        assert not any(p.requires_grad for p in reach.parameters())
