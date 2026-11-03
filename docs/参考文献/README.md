# 参考文献索引

全部 10 篇原文统一存放在本目录。本次整理仅移动和重命名文件，不修改 PDF 内容。以下编号 [1]～[5] 与 [研究方案](../设计方案/面向障碍布局变化的强化学习避障研究方案.md) 一致。

## 主要方法依据

| 编号 | 论文与全文 | 主要用途 |
| --- | --- | --- |
| [1] | [RoboBallet: Planning for multirobot reaching with graph neural networks and reinforcement learning](scirobotics.ads1204.pdf) | 关系化场景表征、布局多样化训练 |
| [2] | [Prioritized Level Replay](2010.03934v4.pdf) | 布局优先采样；重建场景并重新采集当前策略轨迹 |
| [3] | [Sim2real Learning of Obstacle Avoidance for Robotic Manipulators in Uncertain Environments](Sim2real_Learning_of_Obstacle_Avoidance_for_Robotic_Manipulators_in_Uncertain_Environments.pdf) | 紧凑统一几何表示 |
| [4] | [Adaptive Curriculum Generation from Demonstrations for Sim-to-Real Visuomotor Control](Adaptive_Curriculum_Generation_from_Demonstrations_for_Sim-to-Real_Visuomotor_Control.pdf) | 可选的演示轨迹起点课程 |
| [5] | [RoboArm-NMP: a Learning Environment for Neural Motion Planning](2405.16335v1.pdf) | 目标泛化与障碍布局泛化的区分、基线和评估设计；本地为预印本 |

## 相关研究与到达基础

| 论文与全文 | 阅读定位 |
| --- | --- |
| [Rapid Generalization of Motion Planning for Robot Manipulator Among Locally Changed Environments](Rapid_Generalization_of_Motion_Planning_for_Robot_Manipulator_Among_Locally_Changed_Environments.pdf) | 局部环境变化下的运动规划相关工作 |
| [Safe Reinforcement Learning of Robot Trajectories in the Presence of Moving Obstacles](Safe_Reinforcement_Learning_of_Robot_Trajectories_in_the_Presence_of_Moving_Obstacles.pdf) | 动态避障和安全控制相关工作 |
| [A trend-aware reinforcement learning approach for adaptive motion planning of robotic manipulators in dynamic environments](A_trend-aware_reinforcement_learning_approach_for_adaptive_motion_planning_of_robotic_manipulators_in_dynamic_environments.pdf) | 动态环境时序建模相关工作；原文件为 `新的主题/main.pdf` |
| [Application and Evaluation of Soft-Actor Critic Reinforcement Learning in Constrained Trajectory Planning for 6DOF Robotic Manipulators](Application_and_Evaluation_of_Soft-Actor_Critic_Reinforcement_Learning_in_Constrained_Trajectory_Planning_for_6DOF_Robotic_Manipulators.pdf) | 位置与姿态约束下的到达任务背景；不代表继续采用 SAC |
| [Continuous control actions learning and adaptation for robotic manipulation through reinforcement learning](Continuous_control_actions_learning_and_adaptation_for_robotic_manipulation_through_reinforcement_learning.pdf) | 到达与抓取、连续控制和任务适配背景；原文件为 `PPO到达抓取两步.pdf` |

## 使用边界

论文方法的借鉴不等于已在本项目复现。布局泛化、动态预测、安全过滤与高精度位姿到达是不同能力，引用时应核对对应实验条件，不把规划器或安全拦截的效果全部归于策略，也不把少量新目标测试等同于未见障碍布局测试。
