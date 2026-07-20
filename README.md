# Link-Level Dynamic Risk Constrained SAC

本项目根据 `docs/thesis_outline.md` 搭建一个可运行的 PyBullet 仿真原型，用于 UR5-like 机械臂在单动态球形障碍物场景下的连杆级风险约束 SAC 避障训练。

## 环境

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

## 快速检查

```bash
python scripts/smoke_test.py
```

## 训练

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

## 方法名称

- `ee_fixed`: SAC-EndEffectorRisk-FixedPenalty-FixedSmooth
- `link_fixed`: SAC-LinkDynamicRisk-FixedPenalty-FixedSmooth
- `ldrc_fixed`: LDRC-SAC-LinkDynamicRisk-FixedSmooth
- `ldrc_adaptive`: LDRC-SAC-LinkDynamicRisk-AdaptiveSmooth

当前代码定位为论文仿真实验原型。真实 UR5/RGB-D 部署需要接入实际机器人控制接口、相机标定和障碍物检测模块后再使用。
