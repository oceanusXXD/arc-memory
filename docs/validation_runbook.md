# arc 后续验证 Runbook

这个 runbook 用来回答一个问题：**selector 选出的 source 组合，是否能让固定 builder 生成可用 memory，并在减少 token 的同时保持最终问答质量。**

它是小规模验证流程，和完整论文实验分开。完整流程仍由 `scripts/run_arc_runbook.py` 管理；本流程的入口是 `scripts/run_arc_validation.py`。

## 评价原则

selector 不需要复现人工 gold evidence 的 source 编号。不同 source 组合可以触发不同的 builder 构造模式，只要最终结果正确，就算成功。

每个组合按下面的顺序评价：

1. builder 输出满足 JSON 结构并且没有越界 source 引用；
2. compiler auditor 判定 source 支持 memory、memory 满足 requirement；
3. 固定 agent 使用这个 memory 回答问题，记录 `score` 和 `exact_match`；
4. 记录 builder 完整输入 token、调用次数和延迟。

`gold source ID match` 只作为诊断字段，不能作为主要成功指标。

## 阶段和产物

| 阶段 | 目的 | 主要产物 | 是否调用模型 |
|---|---|---|---|
| `coverage` | 检查 gold evidence 是否进入候选池 | `runs/locomo_arc/validation/coverage.json` | 否 |
| `oracle` | 测试 gold 子集和 Full 的真实 builder/agent 质量 | `runs/locomo_arc/validation/oracle.json` | 是 |
| `make-train` | 只保留有 compiler PASS 集合的 train 行 | `compilation.train.validation.jsonl` | 否 |
| `train` | 用多个 PASS 集合训练 selector | `selector.seed-*.pt` | 否（读取已存特征） |
| `eval` | 在独立 dev 上比较 Full、rank_pack、our | `evaluation/budget-*/` | 是 |
| `report` | 计算 paired quality、token saving 和 bootstrap CI | `evaluation/budget-*/report.json` | 否 |

## 0. 环境检查

```bash
cd /teamspace/studios/this_studio/research
export PYTHONPATH=src
python -m py_compile scripts/run_arc_validation.py
```

模型阶段使用 `configs/locomo.yaml` 中配置的真实 builder、auditor 和 agent。先确认凭证和已有 retrieval cache 正常。

## 1. 检查候选覆盖率

```bash
python scripts/run_arc_validation.py --stage coverage
```

检查 `coverage.json` 中的：

- `qa_coverage`：一个 QA 的所有 gold evidence 是否都在候选池；
- `evidence_coverage`：按 evidence block 统计的召回率；
- `missing_examples`：候选池缺失的 QA。

候选池缺少正确 evidence 时，不把它计入 selector 训练失败；它属于 retrieval/data gate。问题、答案和 evidence 互相矛盾的样本要单独修正或排除，并保留记录。

## 2. 先做 gold oracle

只在 evidence 完整映射到候选池的样本上运行：

```bash
python scripts/run_arc_validation.py \
  --stage oracle \
  --split dev \
  --oracle-limit 6
```

这个阶段比较 `gold_oracle` 和 `full_memory` 的最终 agent 分数。它回答的是“压缩后的证据是否有机会工作”，不是 selector 是否已经学会选择。

判读方式：

- gold oracle 质量接近 Full 且 token 明显下降：压缩目标可行，可以进入 selector 训练；
- gold oracle 本身质量很低：先查 evidence、候选覆盖、builder 或 agent，暂时不要调 selector；
- 某个 gold 子集失败但额外上下文成功：说明 gold 不是 builder-aware 的最小集合，需要依赖 compiler 找成功组合。

## 3. 构造 selector 训练档案

先确保 `runs/locomo_arc/compilation.train-dev.jsonl` 中的行来自正确的 Grok annotation，并且每条行有可靠的 `successful_sets`。然后运行：

```bash
python scripts/run_arc_validation.py \
  --stage make-train \
  --train-limit 40
```

这个阶段只保留：

- `split == train`；
- `annotation.valid == true`；
- `domain_complete == true`；
- 至少一个 compiler `PASS` 集合。

它会保留每条记录在不同 budget 下的全部 `successful_sets`。不能把一条 gold evidence 强行替换成唯一训练标签，也不能把没有 PASS 集合的行当成 selector 负例。

正式训练前检查：

```text
total_pass_sets > train_rows
rows_with_multiple_budget_pass_sets > 0
没有 API error / UNKNOWN 被误写成 PASS
```

## 4. 训练 selector

先用一个 seed 做 smoke：

```bash
python scripts/run_arc_validation.py \
  --stage train \
  --train-input runs/locomo_arc/validation/compilation.train.validation.jsonl \
  --epochs 10 \
  --width 32 \
  --seeds 7
```

训练完成后必须能加载 checkpoint。正式实验使用多个 seed，例如：

```bash
python scripts/run_arc_validation.py \
  --stage train \
  --train-input runs/locomo_arc/validation/compilation.train.validation.jsonl \
  --epochs 30 \
  --width 128 \
  --seeds 7,17,27 \
  --resume
```

小数据 smoke 的 train 命中率只用于检查代码是否能学习，不代表算法效果；不能用 source ID 命中率替代最终答案评价。

## 5. 在独立 dev 上做语义评价

使用 train 得到的 primary checkpoint，比较同一批 QA、同一预算下的三个 arm：

```bash
python scripts/run_arc_validation.py \
  --stage eval \
  --checkpoint runs/locomo_arc/validation/selector.seed-7.pt \
  --dev-limit 20 \
  --budget 4096
```

三个 arm 的含义：

- `full_memory`：完整候选集交给 builder；
- `rank_pack`：按固定检索顺序，在同一 token budget 内贪心装包；
- `our`：selector 选择 source，再交给同一个 builder。

然后生成比较报告：

```bash
python scripts/run_arc_validation.py \
  --stage report \
  --budget 4096
```

报告中的主要指标是：

- `score` / `exact_match`：最终 agent 质量；
- `construction_tokens`：builder 完整输入 token；
- `construction_calls`：builder 调用次数；
- `one_sided_95_upper`：paired QA 质量损失的单侧 95% 上界；
- `saving_build`：相对 Full 的 builder token 节省。

## 6. 小规模结论门槛

小规模阶段只给出 `promising`、`not promising` 或 `inconclusive`，不宣称论文结论。建议按以下顺序判断：

1. gold oracle 是否证明压缩在语义上可行；
2. `our` 是否在 dev 上产生非空且语义正确的 memory；
3. `our` 的答案质量是否接近 `full_memory`；
4. `our` 是否比 `rank_pack` 更省 token 或质量更高；
5. 多 seed 结果是否方向一致。

正式 gate 才使用原论文阈值：

```text
our 相对 Full 的质量损失单侧 95% CI 上界 <= 0.02
saving_build > 0
our 在相同预算下不劣于 rank_pack
```

小样本不满足阈值时，只能说明需要更多数据或检查监督，不能单独证明模型结构失败。

## 7. 失败定位

| 现象 | 优先检查 |
|---|---|
| gold coverage 很低 | retrieval 候选池、source cap、evidence 数据 |
| gold oracle 也答不对 | evidence 冲突、builder、agent |
| compiler PASS 但 agent 常答错 | auditor 没检查最终可回答性，或 requirement 不完整 |
| oracle 好，selector 差 | PASS 集合太少、只给单一正例、train/dev 分布不一致 |
| selector 选了非 gold source 但答案正确 | 这是允许的成功，不应标为错误 |
| selector 总是 STOP | 训练档案缺 PASS 集合、budget/成本不一致，或正例覆盖不足 |

只有在候选覆盖、标注、compiler PASS 质量和训练样本量都通过后，selector 仍稳定低于 `rank_pack`，才进入模型结构或 loss 的修改。

## 当前 pilot 的已知边界

当前 100 条 Grok 标注中，train 的完整 gold 映射为 40/72，dev 为 20/28；这只能支持 smoke。当前编译文件只有极少量记录，不能把生成的 checkpoint 或 smoke 分数当成正式算法结论。正式验证必须先扩大 compiler PASS 档案，并使用独立 dev。
