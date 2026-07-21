# Link-Level Dynamic Risk Constrained SAC

本项目根据 `docs/thesis_outline.md` 搭建一个可运行的 PyBullet 仿真原型，用于 UR5-like 机械臂在单动态球形障碍物场景下的连杆级风险约束 SAC 避障训练。

## 环境

推荐从项目根目录创建环境：

```bash
conda env create -f environment.yml
conda activate rl
```

如果需要指定 CUDA 版 PyTorch，可先按下面方式创建 Python 环境，再手动安装匹配本机 CUDA 的 `torch`。

```bash
conda env remove -n rl -y
conda create -n rl --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge -y python=3.13
conda install -n rl --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge -y pip
conda activate rl
```

逐个安装 Python 包：

```bash
python -m pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
python -m pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
python -m pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
python -m pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
python -m pip install pyyaml -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
python -m pip install gymnasium -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
python -m pip install pybullet -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e . --no-deps
```

如果 PyTorch 下载较慢，可先手动下载匹配当前 Python 版本的 wheel，再用 `python -m pip install ./文件名.whl` 本地安装。当前测试通过的组合是 Python 3.13、PyTorch `2.13.0+cu126`。

GPU 验证：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

如需自动选择 GPU，把 `configs/default.yaml` 中的 `device` 改为 `cuda:auto`。程序会综合候选显卡的空闲显存和算力评分选择一张卡；如果 CUDA 不可用且 `device_selection.fallback_to_cpu: true`，会自动回退到 CPU。

## 快速检查

```bash
conda run -n rl python scripts/smoke_test.py
```

## 训练

详细训练流程、过程观察和下一步决策见 [docs/training_guide.md](docs/training_guide.md)。

完整方法：

```bash
python scripts/train.py --config configs/default.yaml --method ldrc_adaptive --total-steps 100000
```

常用对比方法：

```bash
python scripts/train.py --method ee_fixed --total-steps 100000
python scripts/train.py --method link_fixed --total-steps 100000
python scripts/train.py --method ldrc_fixed --total-steps 100000
python scripts/train.py --method ldrc_adaptive --total-steps 100000
```

## 评估

```bash
python scripts/evaluate.py --checkpoint outputs/ldrc_adaptive/actor.pt --episodes 20
```

如需导出典型 episode 曲线数据：

```bash
python scripts/evaluate.py --checkpoint outputs/ldrc_adaptive/actor.pt --episodes 3 --trace-output outputs/ldrc_adaptive/traces
```

## 场景配置

配置入口是 `configs/default.yaml`，它通过 `includes` 组合多个子配置。脚本中的命令行参数只作为临时覆盖使用。

```text
configs/
  default.yaml          # 默认入口，组合下面几个配置
  robot/ur5_like.yaml   # URDF、关节 id、胶囊体、reset 初始状态
  environment/sim.yaml  # PyBullet、目标、障碍物、场景和可视化
  algo/sac.yaml         # 风险代价、平滑、奖励和 SAC 超参数
  run/dev.yaml          # train、eval、smoke 的运行参数
```

`env.obstacle.enabled` 可关闭动态障碍物，用于基础目标到达能力检查。`env.obstacle.scenario` 默认为 `random`，也可设置为 `upper_arm_crossing`、`elbow_crossing`、`forearm_crossing` 或 `wrist_crossing`，用于后续构造靠近不同非末端连杆区域的受控测试场景。

## 方法名称

- `ee_fixed`: SAC-EndEffectorRisk-FixedPenalty-FixedSmooth
- `link_fixed`: SAC-LinkDynamicRisk-FixedPenalty-FixedSmooth
- `ldrc_fixed`: LDRC-SAC-LinkDynamicRisk-FixedSmooth
- `ldrc_adaptive`: LDRC-SAC-LinkDynamicRisk-AdaptiveSmooth

当前代码定位为论文仿真实验原型。真实 UR5/RGB-D 部署需要接入实际机器人控制接口、相机标定和障碍物检测模块后再使用。
