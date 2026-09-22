# arc 测试 Runbook（精简版）

后续判断 selector 是否真正有效的语义验证流程见
[`docs/validation_runbook.md`](validation_runbook.md)，入口脚本为
`scripts/run_arc_validation.py`。该流程把最终 builder/agent 答案作为成功标准，
不会把 source ID 与 gold 不一致自动判为失败。

## 本轮执行范围（2026-09-19 用户指令）

本轮累计完成 **100 条 train/dev Grok 标注后停止**，包括已完成且通过核验的标注。
固定 QA 名单与 seed 7 保存在 `runs/arc_run_decisions.json`，其中 train 72 条、dev 28 条。
教师使用 Codex 当前 URL/key 调用 `grok-4.6`；Full builder 与实验模型使用
SiliconFlow 的 `Qwen/Qwen3.5-4B`。本轮不启动后续编译、训练或评估。

用户已决定将 `conv-42:qa:58`、`conv-42:qa:88`、`conv-47:qa:38` 从训练和校准
子集剔除，保留原数据。完整实验有效分母为 train **883**、dev **340**，合计 **1223**。
下文原始数量账本中的 885/341/1226 是剔除前数量；100 条子集不能代替完整实验验收。

后台启动或断点续跑：

```bash
python scripts/run_arc_pipeline.py --workers 2 --wait-for-quota-reset
```

脚本核验已有产物、增量补齐固定名单的输入，并复用成功 API 日志。若收到带重置时间的
额度错误，将 `BLOCKED` 和自动续跑时间写入 `runs/arc_run_state.json`，到时再继续；
凭证错误或调用结果不明时停止并保留证据。进程心跳见 `runs/arc_pipeline_job.json`，
日志见 `runs/arc_pipeline.log`，执行与验收报告见 `runs/arc_run_report.md`。

## 目标

arc 在冻结检索器和 builder `G` 的条件下，从检索候选集 `E(q)` 先选择结构 `g`，再选择更小的
来源子集 `S(q)`，让 `G` 仍生成可支持回答的 memory，同时降低 builder 的完整
输入 token 成本。

论文意义上的 work 需要同时满足：

- `our` 相对 `full_memory` 的质量损失单侧 95% CI 上界不超过 `0.02`
- `our` 的 builder 输入 token 有稳定节省，`Saving_build > 0`
- `our` 在相同预算下不输 `rank_pack`
- 所有 builder、selector、agent 调用和额外生成成本都被记录

主指标：`score`、`exact_match`、`construction_tokens`、`construction_calls`、
`memory_read_tokens`、`agent_total_tokens`、`total_latency_ms`。

## 当前资源

- 数据：10 个 conversation，1986 条 QA，5882 条 blocks
- Split：train 1157、dev 429、final 400；类别 1–4 的 train/dev/final 分别为 885/341/314 条
- 全量检索缓存：1986 行、14.7GB、hybrid RRF、平均 evidence recall `0.83715`
- Small 验证缓存：196 行、平均 evidence recall `0.69496`
- 模型：Agent/Builder/Auditor 使用 `Qwen/Qwen3.5-4B`；embedding 使用 `Qwen/Qwen3-Embedding-4B`
- 教师：`Grok-4.6`，当前 Codex endpoint/key 已验证可用
- 本地：Python 3.12、Claude Code 2.1.259、约 15GiB RAM、当前未检测到 NVIDIA GPU
- 当前已完成：3 条 Grok 标注、3 条候选域编译、selector 30 epoch 训练、1 条 `our` 真实使用验证

### 数量账本

| 项目（唯一 QA 行） | 当前可用 | 完整实验需要 | 还缺 |
|---|---:|---:|---:|
| Grok 标注（全数据，可选） | 3 条 | 1986 条 | 1983 条 |
| Grok 标注（train+dev，核心必需） | 0 条（现有 3 条来自 final） | 1226 条 | 1226 条 |
| 其中 train selector 子集 | 0 条 | 885 条 | 885 条 |
| 其中 dev 校准子集 | 0 条 | 341 条 | 341 条 |
| train+dev 候选域编译 | 0 条可用（已编译 3 条 final） | 同一 1226 条 | 1226 条 |
| 正式 final QA | 1 条 smoke | 400 条 | 399 条 |
| 正式 final 五臂记录 | 1 条 smoke | 2000 条 | 1999 条 |
| selector 随机 seed | 1 次 | 建议 3 次 | 2 次 |
| 外部 baseline | 0 个 | 15 个 | 15 个 |

这里的 train、dev、train+dev 是包含关系，不是三批相加：`885 + 341 = 1226`。
每条 QA 只需要一组标注，标注内容包括完整候选集上的 `Full(q)`、
`requirements[].id`、需求文本，以及每个需求的 1–3 个 `packages`。当前 3 条
标注全部来自 `conv-49` 的 final smoke，因此不能计入无泄漏 train+dev；核心实验
实际还缺 1226 组，其中 885 组用于训练 H，341 组用于选择 `b*`。Final 的 400 条
只需要做最终回答评估，不需要教师标注；除非要做 final 的离线候选域审计。

## 执行流程

### 1. 环境和 smoke

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=src
set -a; source .env; set +a
python scripts/check_credentials.py
python scripts/verify_real.py --config configs/verify_small.yaml --stage all
```

smoke 只验证真实 API 链路，不代表算法有效。

### 2. 准备数据和检索缓存

```bash
python -m arc.agent.data.locomo prepare --config configs/locomo.yaml
python -m arc.agent.data.retrieval refresh-evidence --config configs/locomo.yaml
python -m arc.agent.data.retrieval refresh-evidence \
  --config configs/verify_small.yaml \
  --input data/cache/verify_small/retrieval/locomo_topk.jsonl \
  --manifest data/cache/verify_small/retrieval/index_manifest.json
```

已有完整缓存时不要重建 index；只刷新 evidence recall 和 offset sidecar。若缓存缺失，
再执行 `python -m arc.agent.data.retrieval index --config configs/locomo.yaml`。当前已规范化
7 条 evidence 异常，剩余 3 条需要人工决定：`conv-42/qa_index=58 (D10:19)`、
`conv-42/qa_index=88 (D)`、`conv-47/qa_index=38 (D4:36)`。

### 3. 生成 compiler 原始输入

```bash
python -m arc.agent.data.requirements \
  --config configs/locomo.yaml \
  --output data/processed/requirements.raw.jsonl
```

该命令只整理 `question + sources + frozen vectors`，不调用模型，也不生成
requirements。

### 4. Grok 教师标注

每条教师输入必须包含：问题、固定编号的候选集 `E(q)`、builder 在完整候选集上
产生的 `Full(q)` 原始输出。教师只输出：

```json
{"requirements":[{"id":"r1","text":"...","packages":[[12],[12,18]]}]}
```

约束：最多 6 个 requirements；每项 1–3 个非空 packages；source ID 必须属于
当前候选集。当前小样本命令：

```bash
python scripts/annotate_with_grok.py \
  --input runs/verify_small/requirements.jsonl \
  --full runs/verify_small/requirements.full.jsonl \
  --output runs/verify_small/requirements.grok.jsonl \
  --limit 3
```

完整实验至少标注 train 类别 1–4 的 885 条和 dev 类别 1–4 的 341 条；final
不参与 selector 训练。先用 20–50 条试跑，再批量执行并断点续跑。

### 5. 编译候选域

```bash
python -m arc.algorithm.compiler \
  --config configs/locomo.yaml \
  --input data/processed/requirements.train-dev.jsonl \
  --output runs/locomo_arc/compilation.train-dev.jsonl \
  --limit 16 --d 8
```

检查 `annotation.valid`、`domain_complete`、`successful_sets`、`PASS/FAIL/UNKNOWN`
状态和冻结 tokenizer 计算的完整输入成本。

### 6. 训练 selector H

```bash
python -m arc.algorithm.selector \
  --config configs/locomo.yaml \
  --input runs/locomo_arc/compilation.train.jsonl \
  --output models/arc_selector.pt \
  --epochs 30
```

正式实验至少使用多个随机 seed。训练完成后，将
`configs/locomo.yaml` 的 `selector.checkpoint` 设置为 `models/arc_selector.pt`。

### 7. Dev 选择预算

分别测试 `1024/2048/4096`，选择满足质量约束的最小预算 `b*`。当前代码默认
使用 `budgets.builder_input_tokens: 4096`，需要补充预算循环、Dev 质量统计和
单侧 bootstrap 95% CI。

### 8. Final 核心评估

全量缓存已支持 offset sidecar 按 QA 懒加载；正式 checkpoint 可用后执行：

```bash
python -m arc.baseline.evaluate run \
  --config configs/locomo.yaml \
  --arms no_memory full_memory naive_rag rank_pack our \
  --split final
```

核心比较：`our` vs `full_memory`、`our` vs `rank_pack`、`our` vs `naive_rag`，
并分析类别 5 的拒答/误答、失败样本、构建成本、agent 成本和延迟。

### 9. 扩展 baseline

当前代码只有五个核心 arm。完整论文表还需补：`Top-k`、`MMR`、`PACMS`、
`Qwen-Rerank-pack`、`Qwen-Prompt-Pick`、`Provence`、`EXIT`、`LLMLingua-2`、
`LongLLMLingua`、`RECOMP-*`、`Prompt-SAW`、`GraphMemix-text`、
`Context-Picker-memory`、`RepoShapley-teacher` 和 `块释放＋前缀目标`。

## 当前阻塞项

1. 正式 train+dev 还缺 1226 条 Grok 标注：train 885、dev 341。
2. 这 1226 条还缺候选域编译，正式 selector checkpoint 尚未训练。
3. Dev 还缺 `b*` 选择、单侧 bootstrap 95% CI 和多 seed 稳定性。
4. Final 还缺 400 条正式五臂评估；外部 baseline 还缺 15 个。
5. Evidence 还剩 3 条无法自动推断，需要决定修正或剔除策略。
6. 需要确认完整标注/编译的 API quota、费用和运行时间；中断临时文件已清理。

## 验收记录

```text
真实链路：PASS / FAIL
全量 selector 训练：PASS / FAIL
Dev 预算 b*：?
our vs full_memory：质量差 = ?，构建 token 节省 = ?
our vs rank_pack：质量差 = ?，构建 token 差 = ?
单侧 95% CI 上界 <= 0.02：YES / NO
Saving_build > 0：YES / NO
多 seed 稳定：YES / NO
结论：work / not work / inconclusive
```
