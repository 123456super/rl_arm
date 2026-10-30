#!/usr/bin/env python3
"""Compare standalone reaching after held-out handoffs, with optional real obstacles."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
import torch
from rl_risk_sac.algorithms.dual_sac import ReachingSACAgent, checkpoint_policy_state
from rl_risk_sac.envs.thesis_homotopy_env import ThesisHomotopyEnv
from rl_risk_sac.tasks.local_samples import configure_nominal_episode, simulation_config
from rl_risk_sac.tasks.local_avoidance import MeshClearance
from rl_risk_sac.tasks.reaching_handoff import (
    HANDOFF_TRAIN_PROTOCOLS, file_digest, joint_handoff_state, recovery_config, recovery_state, validate_handoff_bank,
)


def rollout(env, agent, observation, *, check_mesh=False):
    length=0
    while True:
        margins=MeshClearance(env) if check_mesh else None
        action=agent.select_action(observation,deterministic=True)
        if margins is None:
            observation,_,_,terminated,truncated,info=env.step(action)
        else:
            observation,_,_,terminated,truncated,info=env.step(action,substep_observer=margins)
        length+=1
        collision=bool(info["collision"] or (margins is not None and margins.collision))
        if terminated or truncated or collision:
            return dict(success=bool(info["safe_success"] and not collision and not info["joint_limit"]),collision=collision,
                        joint_limit=bool(info["joint_limit"]),length=length,
                        position_error_m=info["next_rho_position"],orientation_error_rad=info["next_rho_orientation"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--bank",required=True,help="validation/test handoff bank, never training starts")
    parser.add_argument("--output",required=True)
    parser.add_argument("--normal-episodes",type=int,default=100)
    parser.add_argument("--seed",type=int,default=27103)
    parser.add_argument("--keep-obstacles",action="store_true",help="evaluate real handoffs with obstacles and no backup re-entry")
    parser.add_argument("--max-episode-steps",type=int,help="live reaching control steps; default 1000 with --keep-obstacles")
    args=parser.parse_args()
    if args.normal_episodes<1 or args.seed<0: parser.error("positive episodes/nonnegative seed required")
    if args.max_episode_steps is not None and args.max_episode_steps<1: parser.error("positive episode budget required")
    output=Path(args.output)
    if output.exists(): raise FileExistsError(output)
    torch.set_num_threads(1)
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    if state.get("protocol") not in HANDOFF_TRAIN_PROTOCOLS: parser.error("requires a handoff fine-tuning checkpoint")
    bank=torch.load(args.bank,map_location="cpu",weights_only=False)
    validate_handoff_bank(bank,split=("validation","test"),source_sha=state["source_checkpoint_sha256"])
    if {s["source_seed"] for s in bank["samples"]}.intersection(state["training_trace_seeds"]):
        raise ValueError("held-out handoff traces overlap with fine-tuning")
    if file_digest(state["source_checkpoint"])!=state["source_checkpoint_sha256"]:
        raise ValueError("source checkpoint changed")
    source=torch.load(state["source_checkpoint"],map_location="cpu",weights_only=False)
    config=recovery_config(state["config"]);config["device"]="cpu"
    policies={}
    for name,weights in (("baseline",checkpoint_policy_state(source,"reaching")),("finetuned",state["agent"])):
        policies[name]=ReachingSACAgent(241,6,config)
        policies[name].load_state_dict(weights,actor_only=True)
        policies[name].actor.eval().requires_grad_(False)
    rng=np.random.default_rng(args.seed)
    seeds=[]
    protected=set(state["training_trace_seeds"])|{s["source_seed"] for s in bank["samples"]}
    while len(seeds)<args.normal_episodes:
        seed=int(rng.integers(2**32))
        if seed not in protected and seed not in seeds: seeds.append(seed)
    environment_config=simulation_config(source["config"]) if args.keep_obstacles else config
    env=ThesisHomotopyEnv(environment_config,cache_geometry_queries=True)
    rows=[]
    try:
        for kind,items in (("handoff",bank["samples"]),("original",seeds)):
            for index,item in enumerate(items):
                row=dict(kind=kind,index=index)
                for name,agent in policies.items():
                    if kind=="handoff":
                        restore=joint_handoff_state if args.keep_obstacles else recovery_state
                        observation=env.restore_episode_state(restore(item["state"]))
                    else:
                        configure_nominal_episode(env,config)
                        observation,_=env.reset(seed=item)
                    if args.max_episode_steps is not None or args.keep_obstacles:
                        env.horizon=args.max_episode_steps or 1000
                    row[name]=rollout(env,agent,observation,check_mesh=args.keep_obstacles)
                rows.append(row)
                print(f"{kind} {index+1}/{len(items)}: baseline={row['baseline']['success']} finetuned={row['finetuned']['success']}",flush=True)
    finally:
        env.close()
    summary={kind:{name:dict(samples=sum(r['kind']==kind for r in rows),
        success_rate=float(np.mean([r[name]['success'] for r in rows if r['kind']==kind])),
        collision_rate=float(np.mean([r[name]['collision'] for r in rows if r['kind']==kind])))
        for name in policies} for kind in ("handoff","original")}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as stream:
        json.dump(dict(checkpoint=args.checkpoint,bank=args.bank,validation_mode=state['settings']['validation_mode'],
                       obstacles_removed=not args.keep_obstacles,backup_interventions=False,
                       reaching_obstacle_aware={name:agent.obstacle_aware for name,agent in policies.items()},
                       max_episode_steps=args.max_episode_steps or (1000 if args.keep_obstacles else None),
                       summary=summary,outcomes=rows),stream,indent=2)
    print(json.dumps(summary),flush=True)


if __name__=="__main__": main()
