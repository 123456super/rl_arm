#!/usr/bin/env python3
"""Fine-tune only S0 on obstacle-free handoff copies mixed with original S0 resets."""
from __future__ import annotations
import argparse
from collections import deque
import copy
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
import torch
from rl_risk_sac.algorithms.dual_sac import ReachingSACAgent, checkpoint_policy_state
from rl_risk_sac.envs.parallel_thesis_env import ParallelThesisEnvPool
from rl_risk_sac.tasks.local_samples import nominal_episode_contract
from rl_risk_sac.tasks.reaching_handoff import (
    TRAIN_PROTOCOL, RecoveryReplay, bank_fingerprint, file_digest, recovery_config, recovery_state,
    validate_handoff_bank,
)
from rl_risk_sac.utils.training_logs import CsvLog, format_training_progress
from scripts.core.train_thesis_homotopy import accrue_update_credit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", default="outputs/reaching_handoff")
    parser.add_argument("--resume")
    parser.add_argument("--steps", type=int, default=500000, help="additional aggregate environment steps")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--save-interval", type=int, default=50000)
    parser.add_argument("--handoff-fraction", type=float, default=.5, help="fraction of episode starts, not replay transitions")
    parser.add_argument("--actor-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=3e-5)
    parser.add_argument("--alpha-lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=27102)
    parser.add_argument("--device", choices=("cpu", "cuda:auto"), default="cuda:auto")
    parser.add_argument("--validation-mode", action="store_true")
    args = parser.parse_args()
    if (min(args.steps,args.num_envs,args.save_interval)<1 or args.seed<0
            or not 0 < args.handoff_fraction < 1 or min(args.actor_lr,args.critic_lr,args.alpha_lr)<=0):
        parser.error("invalid budgets, mixture or learning rates")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    source_sha = file_digest(args.source_checkpoint)
    source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    validate_handoff_bank(bank, split=("train",), source_sha=source_sha)
    config = recovery_config(source["config"], actor_lr=args.actor_lr, critic_lr=args.critic_lr, alpha_lr=args.alpha_lr)
    config["device"] = args.device
    agent = ReachingSACAgent(241, 6, config)
    settings = dict(num_envs=args.num_envs, handoff_fraction=args.handoff_fraction,
                    actor_lr=args.actor_lr, critic_lr=args.critic_lr, alpha_lr=args.alpha_lr,
                    seed=args.seed, validation_mode=args.validation_mode)
    capacity = 512 if args.validation_mode else 50000
    batch_size = 8 if args.validation_mode else 1024
    replay = RecoveryReplay(capacity, agent.device, args.seed+1)
    rng = np.random.default_rng(args.seed)
    counters = dict(steps=0, episodes=0, updates=0, update_credit=0.)
    recent = {kind:deque(maxlen=100) for kind in ("all","original","handoff")}
    workers, observations = [None]*args.num_envs, {}
    normal_seeds = set()
    loaded = None
    if args.resume:
        loaded = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (loaded.get("protocol")!=TRAIN_PROTOCOL or loaded["source_checkpoint_sha256"]!=source_sha
                or loaded["bank_fingerprint"]!=bank_fingerprint(bank) or loaded["settings"]!=settings):
            raise ValueError("resume source/bank/settings differ")
        agent.load_state_dict(loaded["agent"])
        replay.load_state_dict(loaded["replay"])
        counters.update(loaded["counters"])
        rng.bit_generator.state=loaded["rng"]
        torch.set_rng_state(loaded["torch_rng"].cpu())
        if torch.cuda.is_available() and loaded["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in loaded["cuda_rng"]])
        normal_seeds.update(loaded["normal_reset_seeds"])
        for kind in recent: recent[kind].extend(loaded["recent"][kind])
    else:
        agent.load_state_dict(checkpoint_policy_state(source,"reaching"))
        agent.reset_chain_reference_actor()
    # Optimizer loading restores old LR; explicitly apply the fine-tuning rates afterwards.
    for optimizer,rate in ((agent.actor_optimizer,args.actor_lr),(agent.q_optimizer,args.critic_lr),(agent.alpha_optimizer,args.alpha_lr)):
        for group in optimizer.param_groups: group["lr"]=rate
    frozen_avoidance_sha = bank["source_actor_sha256"]["avoidance"]
    output=Path(args.output_dir)/args.run_name
    output.mkdir(parents=True,exist_ok=False)
    checkpoints=output/"checkpoints"; checkpoints.mkdir()
    (output/"manifest.json").write_text(json.dumps(dict(protocol=TRAIN_PROTOCOL,config=config,settings=settings,
        source_checkpoint=str(Path(args.source_checkpoint).resolve()),source_checkpoint_sha256=source_sha,
        frozen_avoidance_actor_sha256=frozen_avoidance_sha,bank=str(Path(args.bank).resolve()),
        bank_fingerprint=bank_fingerprint(bank),obstacles_removed=True,original_s0_reward=True,
        resume=args.resume),indent=2)+"\n")
    pool=ParallelThesisEnvPool(config,list(range(args.seed+10,args.seed+10+args.num_envs)),
        step_info_keys=("safe_success","collision","joint_limit","next_rho_position","next_rho_orientation"))
    start=counters["steps"]; end=start+args.steps
    started=time.perf_counter(); last_metrics={}; due=[]
    protected=set(bank["excluded_trace_seeds"])
    handoff_seeds={s["source_seed"] for s in bank["samples"]}
    cursor=int(loaded.get("worker_cursor",0)) if loaded else 0
    try:
        if loaded:
            workers=loaded["workers"]
            observations.update(pool.restore_many(loaded["environment_states"]))
        with CsvLog(output/"episodes.csv", ("steps","worker","kind","length","success","collision","joint_limit","position_error_m","orientation_error_rad")) as episodes, \
             CsvLog(output/"progress.csv") as progress:
            def update():
                nonlocal last_metrics
                for _ in due:
                    batch=replay.sample(batch_size)
                    counters["updates"]+=1
                    collect=counters["updates"]==1 or counters["updates"]%10==0
                    metrics=agent.update(batch,reference_observations=batch,collect_diagnostics=collect,check_finite=True)
                    if not all(np.isfinite(x) for x in metrics.values()): raise RuntimeError("nonfinite recovery update")
                    if collect: last_metrics=metrics
                due.clear()

            def save():
                env_states=pool.states([i for i,w in enumerate(workers) if w is not None])
                torch.save(dict(protocol=TRAIN_PROTOCOL,agent=agent.state_dict(),config=config,settings=settings,
                    replay=replay.state_dict(),counters=dict(counters),source_checkpoint_sha256=source_sha,
                    source_checkpoint=str(Path(args.source_checkpoint).resolve()),
                    frozen_avoidance_actor_sha256=frozen_avoidance_sha,bank_fingerprint=bank_fingerprint(bank),
                    training_trace_seeds=sorted(handoff_seeds|normal_seeds),normal_reset_seeds=sorted(normal_seeds),
                    handoff_sample_ids=sorted(s["sample_id"] for s in bank["samples"]),
                    workers=copy.deepcopy(workers),environment_states=env_states,worker_cursor=cursor,
                    recent={k:list(v) for k,v in recent.items()},rng=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                    checkpoints/f"step_{counters['steps']:07d}.pt")

            while counters["steps"]<end:
                boundary=min(end,(counters["steps"]//1000+1)*1000,(counters["steps"]//args.save_interval+1)*args.save_interval)
                count=min(args.num_envs,boundary-counters["steps"])
                indices=[(cursor+i)%args.num_envs for i in range(count)]
                cursor=(cursor+count)%args.num_envs
                normal,normal_reset,handoffs={},{},{}
                for i in indices:
                    if workers[i] is None or not workers[i]["active"]:
                        if rng.random()<args.handoff_fraction:
                            sample=bank["samples"][int(rng.integers(len(bank["samples"])))]
                            handoffs[i]=recovery_state(sample["state"]); kind="handoff"
                        else:
                            seed=int(rng.integers(2**32))
                            while seed in protected or seed in handoff_seeds: seed=int(rng.integers(2**32))
                            normal_seeds.add(seed)
                            normal[i]=nominal_episode_contract(config); normal_reset[i]=seed; kind="original"
                        workers[i]=dict(active=True,kind=kind,length=0)
                for i,(obs,_) in pool.reset_many(normal,seeds=normal_reset).items(): observations[i]=obs
                observations.update(pool.restore_many(handoffs))
                actions=agent.select_actions([observations[i] for i in indices])
                pool.send_steps(dict(zip(indices,actions)))
                update()
                results=pool.recv_steps()
                for i in indices:
                    following,reward,_,terminated,truncated,info=results[i]
                    replay.add(observations[i],actions[indices.index(i)],reward,following,terminated)
                    observations[i]=following; counters["steps"]+=1; workers[i]["length"]+=1
                    if replay.size>=batch_size: due.extend([None]*accrue_update_credit(counters,.25))
                    if terminated or truncated:
                        worker=workers[i]; worker["active"]=False; counters["episodes"]+=1
                        success=bool(info["safe_success"])
                        recent["all"].append(success); recent[worker["kind"]].append(success)
                        episodes.writerow(dict(steps=counters["steps"],worker=i,kind=worker["kind"],length=worker["length"],
                            success=success,collision=info["collision"],joint_limit=info["joint_limit"],
                            position_error_m=info["next_rho_position"],orientation_error_rad=info["next_rho_orientation"]))
                log=counters["steps"]%1000==0 or counters["steps"]==end
                checkpoint=counters["steps"]%args.save_interval==0 or counters["steps"]==end
                if log or checkpoint: update()
                if log:
                    elapsed=time.perf_counter()-started
                    rates={f"{k}_success_rate":float(np.mean(v)) if v else "" for k,v in recent.items()}
                    progress.writerow(dict(**counters,replay_size=replay.size,elapsed_s=elapsed,**rates,**last_metrics)); progress.flush(); episodes.flush()
                    print(format_training_progress(counters["steps"],end,elapsed,recent["all"],label="到达交接微调",completed_steps=counters["steps"]-start)
                          +f" | 原始={rates['original_success_rate']} 交接={rates['handoff_success_rate']}",flush=True)
                if checkpoint: save()
    finally:
        pool.close()
    (output/"summary.json").write_text(json.dumps(dict(**counters,frozen_avoidance_actor_sha256=frozen_avoidance_sha),indent=2)+"\n")


if __name__=="__main__":
    main()
