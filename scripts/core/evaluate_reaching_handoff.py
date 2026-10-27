#!/usr/bin/env python3
"""Compare original/fine-tuned S0 on held-out handoffs and identical original S0 tasks."""
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
from rl_risk_sac.tasks.local_samples import configure_nominal_episode
from rl_risk_sac.tasks.reaching_handoff import TRAIN_PROTOCOL, file_digest, recovery_state, validate_handoff_bank


def rollout(env, agent, observation):
    length=0
    while True:
        observation,_,_,terminated,truncated,info=env.step(agent.select_action(observation,deterministic=True))
        length+=1
        if terminated or truncated:
            return dict(success=bool(info["safe_success"]),collision=bool(info["collision"]),
                        joint_limit=bool(info["joint_limit"]),length=length,
                        position_error_m=info["next_rho_position"],orientation_error_rad=info["next_rho_orientation"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--bank",required=True,help="validation/test handoff bank, never training starts")
    parser.add_argument("--output",required=True)
    parser.add_argument("--normal-episodes",type=int,default=100)
    parser.add_argument("--seed",type=int,default=27103)
    args=parser.parse_args()
    if args.normal_episodes<1 or args.seed<0: parser.error("positive episodes/nonnegative seed required")
    output=Path(args.output)
    if output.exists(): raise FileExistsError(output)
    torch.set_num_threads(1)
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    if state.get("protocol")!=TRAIN_PROTOCOL: parser.error("requires a handoff fine-tuning checkpoint")
    bank=torch.load(args.bank,map_location="cpu",weights_only=False)
    validate_handoff_bank(bank,split=("validation","test"),source_sha=state["source_checkpoint_sha256"])
    if {s["source_seed"] for s in bank["samples"]}.intersection(state["training_trace_seeds"]):
        raise ValueError("held-out handoff traces overlap with fine-tuning")
    if file_digest(state["source_checkpoint"])!=state["source_checkpoint_sha256"]:
        raise ValueError("source checkpoint changed")
    source=torch.load(state["source_checkpoint"],map_location="cpu",weights_only=False)
    config=copy.deepcopy(state["config"]);config["device"]="cpu"
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
    env=ThesisHomotopyEnv(config,cache_geometry_queries=True)
    rows=[]
    try:
        for kind,items in (("handoff",bank["samples"]),("original",seeds)):
            for index,item in enumerate(items):
                row=dict(kind=kind,index=index)
                for name,agent in policies.items():
                    if kind=="handoff": observation=env.restore_episode_state(recovery_state(item["state"]))
                    else:
                        configure_nominal_episode(env,config)
                        observation,_=env.reset(seed=item)
                    row[name]=rollout(env,agent,observation)
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
                       obstacles_removed=True,summary=summary,outcomes=rows),stream,indent=2)
    print(json.dumps(summary),flush=True)


if __name__=="__main__": main()
