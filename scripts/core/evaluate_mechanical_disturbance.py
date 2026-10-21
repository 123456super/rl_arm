#!/usr/bin/env python3
"""Compact paired L13 physical tool-force benchmark for frozen policies."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    'scripts/core/evaluate_mechanical_disturbance.py',
    'scripts/core/evaluate_thesis_homotopy.py',
    'src/rl_risk_sac/envs/mechanical_disturbance.py',
    'src/rl_risk_sac/envs/thesis_homotopy_env.py',
    'src/rl_risk_sac/envs/thesis_homotopy_env_reduced_observation.py',
    'src/rl_risk_sac/envs/position_sensor_noise.py',
    'src/rl_risk_sac/envs/orientation_sensor_noise.py',
    'src/rl_risk_sac/envs/parallel_thesis_env.py',
    'src/rl_risk_sac/robots/pybullet_robot.py',
    'src/rl_risk_sac/algorithms/homotopy_curriculum.py',
    'scripts/core/train_thesis_homotopy.py',
    'src/rl_risk_sac/tasks/thesis_reaching.py',
)
FIELDS = ('policy', 'training_seed', 'force_n', 'episodes', 'success_rate',
          'collision_rate', 'timeout_rate', 'joint_limit_rate', 'mean_episode_steps',
          'p95_position_error_mm', 'p95_orientation_error_rad',
          'checkpoint_sha256', 'source_config_sha256', 'result_file')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def average(rows, key):
    return sum(r[key] for r in rows) / len(rows)


def render(output, rows, run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for i, policy in enumerate(run['labels']):
        selected = sorted([r for r in rows if r['policy'] == policy], key=lambda r: r['force_n'])
        ax.plot([r['force_n'] for r in selected], [100*r['success_rate'] for r in selected],
                marker=('o', 's')[i % 2], lw=1.6, ms=5,
                label=f"Full, train seed {selected[0]['training_seed']}")
    ax.set(xlabel='External tool force magnitude (N)', ylabel='Coupled-pose success (%)',
           title='Mechanical disturbance at 1 mm / 0.025 rad', ylim=(-2, 103))
    ax.set_xticks(run['force_n'])
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(output / 'mechanical_disturbance_compact.png', dpi=300)
    fig.savefig(output / 'mechanical_disturbance_compact.svg')
    plt.close(fig)
    grouped = {f: [r for r in rows if r['force_n'] == f] for f in run['force_n']}
    table = ['| 外力幅值 (N) | seed 11001 (%) | seed 11002 (%) | 平均成功率 (%) | 平均超时率 (%) | 平均终点位置 P95 (mm) |',
             '|---:|---:|---:|---:|---:|---:|']
    for f, selected in grouped.items():
        rates = [100*r['success_rate'] for r in selected]
        table.append(f"| {f:g} | {rates[0]:.2f} | {rates[1]:.2f} | {average(selected,'success_rate')*100:.2f} | {average(selected,'timeout_rate')*100:.2f} | {average(selected,'p95_position_error_mm'):.3f} |")
    methods = (
        'Two frozen Full policies were evaluated at the strict tolerance of 1 mm / 0.025 rad '
        f"on {run['episodes']} paired episodes per condition. A world-frame external force "
        'of fixed magnitude (0, 5, 10, or 20 N) was applied at the current tool origin throughout '
        'each episode. Its direction was sampled uniformly on the sphere and held for 0.25 s '
        'before resampling. The force was reapplied at every 240-Hz physics substep, while the '
        'policy operated at 20 Hz. Initial tasks and direction sequences were paired across '
        'force magnitudes, using a disturbance random stream independent of task sampling. '
        'Measurements remained exact; the policy, velocity servo, motor limits, and success '
        'criterion (five consecutive control samples) were unchanged.'
    )
    results = ('Mean success rates across the two training seeds were '
               + ', '.join(f"{average(selected,'success_rate')*100:.2f}% at {f:g} N" for f, selected in grouped.items())
               + '. These measurements characterize the closed-loop policy–servo system under the specified simulated external-force model; they do not isolate disturbance rejection by the learned policy or establish physical-robot performance.')
    caption = ('Success under randomly redirected external tool forces at 1 mm / 0.025 rad. '
               'Each curve represents one frozen training seed evaluated on 1000 paired episodes '
               'per force magnitude. The force acts throughout each episode and changes direction every 0.25 s.')
    (output / 'manuscript_insert.md').write_text('# Methods\n\n' + methods + '\n\n# Results\n\n' + results + '\n\n# Figure caption\n\n' + caption + '\n')
    report = ['# L13 机械外力扰动实验', '', f"共 {len(rows)} 个条件 / {sum(r['episodes'] for r in rows)} 回合。", '',
              '使用两份原 Full 冻结策略；传感器准确；外力从首个控制周期开始，贯穿整个回合，每 0.25 s 随机换向，幅值固定。',
              '方向各向同性，同一任务的方向序列在各强度和策略间共用。每个物理子步在当前 TCP 世界坐标位置重新施力，不修改动作或直接跳变状态。',
              '', *table, '',
              'P95 包含全部成功及失败回合；上表为两个策略各自终点 P95 的算术平均，并非合并分位数。', '',
              '这些结果包含原有底层速度伺服的抗扰作用。有限外力可能被高增益速度伺服补偿，不能把系统成功率全部归功于策略；也不能据此声称优于其他算法。',
              '两训练种子不能支撑充分的训练方差估计。本实验不是测量噪声、载荷变化、真机测试或达到目标后的长期保持测试。', '',
              '## English methods', '', methods, '', '## English results', '', results, '', '## Figure caption', '', caption]
    (output / 'report.md').write_text('\n'.join(report) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', nargs=2, required=True)
    p.add_argument('--labels', nargs=2, default=['Full_seed11001', 'Full_seed11002'])
    p.add_argument('--episodes', type=int, default=1000)
    p.add_argument('--num-envs', type=int, default=40)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    if args.episodes != 1000 or args.num_envs < 1 or len(set(args.labels)) != 2:
        p.error('formal protocol requires 1000 episodes, positive workers and unique labels')
    for checkpoint in args.checkpoint:
        if not (ROOT / checkpoint).is_file():
            p.error(f'missing checkpoint: {checkpoint}')
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    run = {**vars(args), 'status': 'running', 'protocol': 'compact_L13_external_force_v1',
           'force_n': [0, 5, 10, 20], 'interval_s': .25, 'test_seed': 41001,
           'disturbance_seed': 72001, 'source_sha256': {s: sha(ROOT/s) for s in SOURCES},
           'checkpoint_sha256': [sha(ROOT/c) for c in args.checkpoint]}
    for source in SOURCES:
        target = out / 'source_at_launch' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/source).read_bytes())
    (out / 'run.json').write_text(json.dumps(run, indent=2)+'\n')
    rows = []
    started = time.perf_counter()
    try:
        for index, (label, checkpoint) in enumerate(zip(args.labels, args.checkpoint)):
            for force in run['force_n']:
                stem = f'policy{index+1}_force{force:g}N_seed41001'
                result = out / f'{stem}.json'
                cmd = [sys.executable, str(ROOT/'scripts/core/evaluate_thesis_homotopy.py'),
                       '--checkpoint', checkpoint, '--checkpoint-config', '--level-index', '13',
                       '--episodes', str(args.episodes), '--seed', '41001',
                       '--external-force-n', str(force), '--force-interval-s', '.25',
                       '--disturbance-seed', '72001', '--num-envs', str(args.num_envs),
                       '--no-cpu-affinity', '--output', str(result)]
                print(f'[{len(rows)+1}/8] {label}, {force} N', flush=True)
                with (out/f'{stem}.log').open('w') as log:
                    subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                m = json.loads(result.read_text())
                assert m['position_tolerance_m'] == .001 and m['orientation_tolerance_rad'] == .025
                assert m['episodes'] == 1000 and m['bin_episode_counts'] == [10]*100
                assert m['checkpoint_sha256'] == run['checkpoint_sha256'][index]
                assert m['mechanical_disturbance']['force_n'] == force
                assert m['sensor_noise']['position_std_m'] == m['sensor_noise']['orientation_std_rad'] == 0
                rows.append({'policy': label, 'force_n': force, 'result_file': result.name,
                             'p95_position_error_mm': 1000*m['p95_position_error_m'],
                             **{k: m[k] for k in FIELDS if k in m}})
                with (out/'summary.csv').open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"  success={100*m['success_rate']:.2f}% timeout={100*m['timeout_rate']:.2f}% elapsed={m['elapsed_seconds']:.1f}s", flush=True)
        current = {s: sha(ROOT/s) for s in SOURCES}
        assert current == run['source_sha256'], 'source changed during evaluation'
        for label in args.labels:
            assert len({r['source_config_sha256'] for r in rows if r['policy']==label}) == 1
        render(out, rows, run)
        run.update(status='complete', unique_conditions=len(rows), total_episodes=sum(r['episodes'] for r in rows),
                   elapsed_seconds=time.perf_counter()-started, source_unchanged=True)
    except BaseException as exc:
        run.update(status='failed', error=str(exc), completed_conditions=len(rows))
        raise
    finally:
        (out/'run.json').write_text(json.dumps(run, indent=2)+'\n')
    print(f'Complete: {out}', flush=True)


if __name__ == '__main__':
    main()
