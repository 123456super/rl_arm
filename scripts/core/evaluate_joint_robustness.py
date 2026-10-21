#!/usr/bin/env python3
"""Paired position-measurement noise plus physical tool-force evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.core.evaluate_mechanical_disturbance import SOURCES as MECHANICAL_SOURCES

SOURCES = MECHANICAL_SOURCES + ('scripts/core/evaluate_joint_robustness.py',)
FIELDS = ('policy', 'training_seed', 'position_std_mm', 'force_n', 'episodes',
          'success_rate', 'collision_rate', 'timeout_rate', 'joint_limit_rate',
          'mean_episode_steps', 'p95_position_error_mm', 'p95_orientation_error_rad',
          'checkpoint_sha256', 'source_config_sha256', 'result_file')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grouped(rows):
    return {v: [r for r in rows if r['position_std_mm'] == v]
            for v in sorted({r['position_std_mm'] for r in rows})}


def avg(rows, key):
    return sum(r[key] for r in rows) / len(rows)


def render(output, rows, run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for i, policy in enumerate(run['labels']):
        selected = sorted([r for r in rows if r['policy'] == policy], key=lambda r: r['position_std_mm'])
        ax.plot([r['position_std_mm'] for r in selected], [100*r['success_rate'] for r in selected],
                marker=('o', 's')[i % 2], lw=1.6, ms=4,
                label=f"Full, train seed {selected[0]['training_seed']}")
    groups = grouped(rows)
    ax.plot(list(groups), [100*avg(v,'success_rate') for v in groups.values()],
            color='black', ls='--', marker='D', ms=3, lw=1.2, label='Two-seed mean')
    ax.set(xlabel='Position noise std per axis (mm)', ylabel='Coupled-pose success (%)',
           title=f"Joint perturbations: {run['force_n']:g} N tool force", ylim=(-2, 103))
    ax.set_xticks(run['position_std_mm'], [f'{s:g}' for s in run['position_std_mm']])
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(output/'joint_robustness_compact.png', dpi=300)
    fig.savefig(output/'joint_robustness_compact.svg')
    plt.close(fig)
    table = ['| 位置噪声每轴 σ (mm) | seed 11001 (%) | seed 11002 (%) | 平均成功率 (%) |',
             '|---:|---:|---:|---:|']
    for sigma, selected in groups.items():
        rates = [100*r['success_rate'] for r in selected]
        table.append(f"| {sigma:g} | {rates[0]:.2f} | {rates[1]:.2f} | {100*avg(selected,'success_rate'):.2f} |")
    methods = (
        f"Two frozen Full policies were evaluated at 1 mm / 0.025 rad on {run['episodes']} "
        'paired episodes per condition under simultaneous position-measurement noise and physical tool forces. '
        f"A {run['force_n']:g}-N world-frame force acted throughout each episode, with its isotropic "
        'direction resampled every 0.25 s and reapplied at every 240-Hz physics substep. '
        'Independent Gaussian position-measurement noise was sampled at 20 Hz, with per-axis standard '
        'deviations of ' + ', '.join(f'{s:g}' for s in run['position_std_mm']) + ' mm. '
        'The same noisy pose was used for pose/keypoint features and precision-dependent action scaling; '
        'orientation and joint measurements remained exact. Success and terminal errors used the true '
        'simulated state, with the original five-sample success hold and 500-step horizon. '
        'Tasks and standardized noise/force-direction sequences were paired across levels. '
    )
    if run['phase'] == 'formal':
        methods += ('The nonzero noise levels were specified before formal evaluation. An independent pilot study explored the transition '
                    'in success rate; its tasks and random-stream seeds were excluded from the formal evaluation. '
                    'No zero-perturbation conditions were rerun.')
    results = ('Mean success rates across the two training seeds were '
               + ', '.join(f"{100*avg(v,'success_rate'):.2f}% at {s:g} mm" for s,v in groups.items())
               + '. The full evaluated sweep is reported without selecting outcomes for monotonicity. '
               'This experiment characterizes the policy–servo system under combined perturbations; '
               'a fixed force magnitude does not identify a statistical interaction or an advantage over other algorithms.')
    caption = (f"Success under simultaneous position-measurement noise and {run['force_n']:g}-N randomly redirected "
               f"tool force at 1 mm / 0.025 rad. Each frozen policy was evaluated on {run['episodes']} paired episodes "
               'per noise level. Noise magnitude is the per-axis standard deviation. The dashed curve is the two-seed mean.')
    (output/'manuscript_insert.md').write_text('# Methods\n\n'+methods+'\n\n# Results\n\n'+results+'\n\n# Figure caption\n\n'+caption+'\n')
    lines = ['# 位置测量噪声与机械外力联合实验', '', f"阶段：{run['phase']}；共 {len(rows)} 个条件 / {sum(r['episodes'] for r in rows)} 回合。", '',
             f"各条件外力均为 {run['force_n']:g} N，仅扫描非零位置噪声。姿态测量准确。", '', *table, '',
             '先导实验与正式实验分离。正式评估完整报告用户指定的七个等级，不强制成功率单调，不按正式测试结果反复调参或排除点。',
             '外力较小的影响可能由底层速度伺服补偿；该结果不隔离学习策略的抗扰作用。两训练种子不代表充分的训练方差估计。', '',
             '## English methods', '', methods, '', '## English results', '', results, '', '## Caption', '', caption]
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', nargs=2, required=True)
    p.add_argument('--labels', nargs=2, default=['Full_seed11001', 'Full_seed11002'])
    p.add_argument('--phase', choices=['pilot','formal'], required=True)
    p.add_argument('--position-std-mm', type=float, nargs='+', required=True)
    p.add_argument('--force-n', type=float, default=20)
    p.add_argument('--episodes', type=int, required=True)
    p.add_argument('--test-seed', type=int, required=True)
    p.add_argument('--sensor-noise-seed', type=int, required=True)
    p.add_argument('--disturbance-seed', type=int, required=True)
    p.add_argument('--num-envs', type=int, default=40)
    p.add_argument('--selection-record', help='required before the formal run; frozen pilot-to-formal decision JSON')
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    if (args.episodes < 100 or args.episodes % 100 or args.num_envs < 1
            or len(set(args.labels)) != 2 or min(args.test_seed,args.sensor_noise_seed,args.disturbance_seed) < 0):
        p.error('positive workers, unique labels, nonnegative seeds and a multiple of 100 episodes required')
    if not math.isfinite(args.force_n) or args.force_n <= 0:
        p.error('force must be positive and finite; no zero-force conditions')
    if (len(set(args.position_std_mm)) != len(args.position_std_mm)
            or any(not math.isfinite(x) or x <= 0 for x in args.position_std_mm)):
        p.error('noise levels must be unique, positive and finite')
    if args.phase == 'formal':
        if not args.selection_record:
            p.error('formal evaluation requires a selection record')
        record = json.loads(Path(args.selection_record).read_text())
        if record['formal_position_std_mm'] != args.position_std_mm:
            p.error('formal noise levels disagree with frozen selection record')
        for arg, key in [('episodes','formal_episodes'),('test_seed','formal_test_seed'),
                         ('sensor_noise_seed','formal_sensor_noise_seed'),('disturbance_seed','formal_disturbance_seed')]:
            if getattr(args,arg) != record[key]:
                p.error(f'{arg} disagrees with frozen selection record')
        if args.force_n != record['force_n']:
            p.error('force disagrees with frozen selection record')
    for checkpoint in args.checkpoint:
        if not (ROOT/checkpoint).is_file():
            p.error(f'missing checkpoint: {checkpoint}')
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    run = {**vars(args), 'status':'running', 'protocol':'joint_position_noise_tool_force_L13_v1',
           'source_sha256':{s:sha(ROOT/s) for s in SOURCES},
           'checkpoint_sha256':[sha(ROOT/c) for c in args.checkpoint]}
    for source in SOURCES:
        target=out/'source_at_launch'/source
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/source).read_bytes())
    if args.selection_record:
        run['selection_record_sha256']=sha(args.selection_record)
        (out/'selection_record.json').write_bytes(Path(args.selection_record).read_bytes())
    (out/'run.json').write_text(json.dumps(run,indent=2)+'\n')
    rows=[]
    started=time.perf_counter()
    total=len(args.checkpoint)*len(args.position_std_mm)
    try:
        for index,(label,checkpoint) in enumerate(zip(args.labels,args.checkpoint)):
            for sigma in args.position_std_mm:
                stem=f'policy{index+1}_position{sigma:g}mm_force{args.force_n:g}N_seed{args.test_seed}'
                result=out/f'{stem}.json'
                cmd=[sys.executable,str(ROOT/'scripts/core/evaluate_thesis_homotopy.py'),
                     '--checkpoint',checkpoint,'--checkpoint-config','--level-index','13',
                     '--episodes',str(args.episodes),'--seed',str(args.test_seed),
                     '--position-noise-std-mm',str(sigma),'--sensor-noise-seed',str(args.sensor_noise_seed),
                     '--external-force-n',str(args.force_n),'--force-interval-s','.25',
                     '--disturbance-seed',str(args.disturbance_seed),'--num-envs',str(args.num_envs),
                     '--no-cpu-affinity','--output',str(result)]
                print(f'[{len(rows)+1}/{total}] {args.phase}: {label}, sigma={sigma:g} mm, force={args.force_n:g} N',flush=True)
                with (out/f'{stem}.log').open('w') as log:
                    subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                m=json.loads(result.read_text())
                assert m['episodes']==args.episodes and m['bin_episode_counts']==[args.episodes//100]*100
                assert m['position_tolerance_m']==.001 and m['orientation_tolerance_rad']==.025
                assert m['checkpoint_sha256']==run['checkpoint_sha256'][index]
                assert m['mechanical_disturbance']['force_n']==args.force_n
                assert m['sensor_noise']['position_std_m']==sigma/1000 and m['sensor_noise']['orientation_std_rad']==0
                rows.append({'policy':label,'position_std_mm':sigma,'force_n':args.force_n,
                             'p95_position_error_mm':1000*m['p95_position_error_m'],'result_file':result.name,
                             **{k:m[k] for k in FIELDS if k in m}})
                with (out/'summary.csv').open('w',newline='') as handle:
                    writer=csv.DictWriter(handle,fieldnames=FIELDS)
                    writer.writeheader();writer.writerows(rows)
                print(f"  success={100*m['success_rate']:.2f}% elapsed={m['elapsed_seconds']:.1f}s",flush=True)
        assert {s:sha(ROOT/s) for s in SOURCES}==run['source_sha256'],'source changed during evaluation'
        for label in args.labels:
            assert len({r['source_config_sha256'] for r in rows if r['policy']==label})==1
        render(out,rows,run)
        run.update(status='complete',unique_conditions=len(rows),total_episodes=sum(r['episodes'] for r in rows),
                   elapsed_seconds=time.perf_counter()-started,source_unchanged=True)
    except BaseException as exc:
        run.update(status='failed',error=str(exc),completed_conditions=len(rows))
        raise
    finally:
        (out/'run.json').write_text(json.dumps(run,indent=2)+'\n')
    print(f'Complete: {out}',flush=True)


if __name__=='__main__':
    main()
