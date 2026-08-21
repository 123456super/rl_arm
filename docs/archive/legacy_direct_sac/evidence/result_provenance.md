# 结果来源与完整性说明

## 归档方式

文档快照来自 Git 提交 `ab73e567035c194cc2a61563419aadf2e90aad5a`。结果副本来自 2026-08-19 归档时工作区中的 `outputs/reaching_incremental/`；这些产物不属于该提交的不可变内容，因此以归档副本及其 SHA-256 作为本次证据快照。

原始 `outputs/` 没有移动、重命名或删除。这样可保留 summary 内的 `eval_csv` 路径、训练恢复链和既有脚本兼容性。今后即使原路径被复跑覆盖，也应优先用本目录副本复核本次记录。

## 核心副本校验值

| 归档文件 | SHA-256 |
| --- | --- |
| `s1_static_obstacle_summary.json` | `308a06f24d8197b412f1901ffbe2ae80060059f32b1b8b1250511dc8e4f7f306` |
| `s1_static_obstacle_finetune_summary.json` | `82b73e474facaf6b1343335f6302ae9e1f7b2ff6df49ffb32ca1372e7f52475a` |
| `s1_static_extend100k_summary.json` | `60bbe150628fd918633f2a088b30a5d9a770056e7e5e0641c780d251e5f66e27` |
| `s1_static_candidate_terminal_refine_current_rerun_summary.json` | `ab0daf386a589d24af10bd92631e7ec859bafd318f15d319f2ee6f3a23ad8d54` |
| `s1_r3_blind_summary.json` | `725211b00515b3ce3757f85e59d7b489852468af153fbe7d17e5f0f72265e3f7` |
| `s1_r4_blind_summary.json` | `bf1c4ea83defdf09e43beba396b8ca8db290d207b313c926e547a67bb643e558` |
| `s1_static_failure_diagnostics_summary.json` | `85d042dc76a5718d895cf8aaeadb81cf647541b13b804ae07a6deac5be39b8ad` |
| `s1_terminal_refine_diagnostics_summary.json` | `fbdd90120ba1a208056fc8467a5bd714419dbd5216f79d7746631f937991b58e` |
| `s1_terminal_refine_detailed_failure_report.txt` | `0239a31cab650ebb8c5050923de351133e11b3ee4619edd184f89c4c8273f349` |

## 固定任务定义校验值

| 文件 | SHA-256 |
| --- | --- |
| `configs/experiments/reaching_recovery/manifests/v1_final.json` | `f6ec31e2391c219bc964200fd450c08fb169ee7993fb44b7e01dd046715796f4` |
| `configs/experiments/reaching_recovery/manifests/v1_validation.json` | `719f9de9502aec6621022503ac1d3a4a3a104f92aeb7ed2e29a56231d2435d93` |
| `configs/experiments/reaching_incremental/manifests/s1_blind_final_v1.json` | `3212d61b4010a83dc5ef4da115b2ce2d134feb5cf502365a7ddcf2b4aa2c80a2` |
| `configs/experiments/reaching_incremental/frozen_reaching_v2_actors.json` | `8b90fb922b41e9d1c348024e985322e9aa4b620d181c0a5c4a32782e1a3a37c6` |
| `outputs/reaching_incremental/s1_static_failure_diagnostics/feasibility/static_zero_speed_v1_final.json` | `5c1b8493bdcda568fcb41b4ac2d040820d394bd7160e02937d3a5df122b90a33` |

S0 没有独立 pooled summary，冻结结论由以下三个原始 final CSV 汇总。它们保留在原路径、不在文档目录重复复制：

| 原始文件 | SHA-256 |
| --- | --- |
| `outputs/reaching_recovery_v2/eval/seed_4301_final.csv` | `062e90308a78d3a748c27fc358388d1769501b140300e920ad4744d423a1479c` |
| `outputs/reaching_recovery_v2/eval/seed_4302_final.csv` | `9d8319a03671fc234788d21d780ba655bdeec4ca37b260fea634ba292488223b` |
| `outputs/reaching_recovery_v2/eval/seed_4303_final.csv` | `e9ecc0e5324aa13d2e3d611893246a4e70cca0dec22fd7ed35290e94f0348618` |

## 不可混用的两个 candidate terminal refine 记录

| 记录 | 成功率 | 可核对来源 |
| --- | ---: | --- |
| 历史最好 | `521/600=86.83%` | 归档旧协议与旧研究状态；原 summary 已不在当前磁盘 |
| 当前磁盘复跑 | `511/600=85.17%` | 本目录 `current_rerun` JSON，SHA-256 如上 |

论文表格可以同时呈现两者，但必须分成“历史记录”和“可复核复跑”，不能选择性把旧成功率与新诊断数据拼接成一行。
