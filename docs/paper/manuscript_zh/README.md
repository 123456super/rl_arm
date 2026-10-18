# 中文会议论文初稿

本目录基于 `docs/Templates/Conference-LaTeX-template_10-17-19` 的 `IEEEtran` 双栏会议格式，正文暂用中文撰写，入口文件为 [`main.tex`](./main.tex)。作者、机构和基金信息仍需在投稿前替换；实验表格已根据 `outputs/evaluation_final/` 的 39 个冻结评估 JSON 汇总。

## 编译

模板本身是 LaTeX，中文稿使用 XeLaTeX 和 `ctex`：

```bash
cd docs/paper/manuscript_zh
xelatex main.tex
xelatex main.tex
```

编译需要 XeLaTeX 和 CTeX；若当前环境没有 `xelatex`，可在安装 TeX Live 后执行上述两次编译命令。提交前仍需检查页数、作者信息、图像分辨率和参考文献 DOI/链接。

## 初稿结构

- 引言：把论文问题固定为高精度联合位姿控制的局部收敛退化；
- 相关工作：SAC、课程学习、位姿表示和经验回放；
- 方法：关键点/Jacobian 局部表征、联合瓶颈塑形、精度边界回放和自动课程；
- 实验：Hybrid KP-SAC-Base、Full、M1、M2、M3；L0--L13；Full/Base 双种子、M1--M3 预指定单种子训练；
- 结果：共同 L6 与各方法实际最高档位的冻结评估，使用 31001/31002/31003 三个独立评估种子；
- 讨论：明确仿真结果不能直接等同于真实机器人精度。

图示和实验曲线应在新实验完成后补入。图中的文字需保持英文或无文字，避免违反模板关于图内中文字符的要求。
