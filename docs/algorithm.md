# ARC 算法实现说明

本文档是代码对应的独立算法说明。理论定义和论文表述以
[`docs/papers/3_method.md`](papers/3_method.md) 为准；本文档只说明当前仓库如何把这些定义执行出来。

ARC 的核心约束只有四条：

1. 写入策略只接收写入时可见的候选来源集合和预算，不接收当前问题或未来问题。
2. 策略按内容优先顺序先选来源集合 (S)，再在给定 (S) 上选架构 (g)。
3. 一次写入最多执行一个最终方案的 `Build` 和一个 `Update`。
4. 查询阶段只执行固定的 `Retrieve` 和回答模型，不重新调用策略、构建器或更新器。

## 1. 状态与候选来源

第 (i) 个写入单元包含写入前状态 (mathcal M_i^{-})、新历史批次 (X_i) 和输入预算 (b)。候选生成器实现为
`arc.agent.persistent.candidate_sources`：

```text
E_i = C(X_i, M_i^-)
```

新历史按输入顺序加入；已有状态中的来源只通过可观察的文本重叠和确定性的近期邻域加入。候选生成不读取问题、参考答案、需求标签或查询向量。每个候选保留：

- 本地整数 `id`，用于本次策略解码；
- 稳定的 `source_id`，用于持久化和后续更新；
- 原文、会话、时间、说话人、位置和冻结向量等元数据。

离线编译器也以同一边界接收候选集合。正式输入由会话历史和冻结来源向量构造；已经物化的候选行仍可通过 `candidate_sources`/`sources` 字段重放。

## 2. 架构与构建器

架构集合固定为：

| 架构 | 约束 |
|---|---|
| `Flat` | 独立的来源支持事实或事件，不建显式边 |
| `Chain` | 按规范来源顺序形成相邻路径 |
| `Tree` | 单根、无环、每个非根节点一个父节点 |
| `Graph` | 只建立同会话内相邻的可观察关系边 |
| `Cluster` | 按可观察会话分组，组之间不重叠且覆盖全部节点 |

`arc.algorithm.architecture.instantiate_structure` 根据来源元数据生成固定拓扑，
`validate_structure` 检查节点覆盖、重复节点、重复边、自环、树根和簇覆盖等约束。

对于选定来源集合 (S) 和架构 (g)，`serialize_input` 构造完整构建输入：

```text
Serialize_g(S) = instruction + schema + architecture_contract + structure + sources
```

来源文本、来源标识和架构字段都进入序列化；问题不会进入序列化结果。`_builder_content` 还明确要求模型不得使用 query 或 future query。构建输出必须是最多六个对象组成的 JSON 数组，每个对象严格包含：

```json
{"m": "source-supported memory", "src": [1, 2]}
```

`src` 只能引用当前 (S) 的本地来源 ID。`build_once` 对非空 (S) 执行一次冻结构建器调用，然后执行 JSON/schema 解析和三层认证：

1. **结构认证**：根类型、字段、来源 ID 和架构拓扑合法；
2. **来源忠实性认证**：生成记忆必须由输入来源支持；
3. **需求完整性认证**：记忆覆盖离线标注的全部信息需求。

认证状态只有 `PASS`、`FAIL`、`UNKNOWN`。解析或明确 schema 违反是 `FAIL`；API 错误、服务中断或信息不足是 `UNKNOWN`，不会被改写成训练负例。

## 3. 预算与成本

非空方案的精确构建输入成本为：

```text
T_g(S) = frozen_builder_tokenizer(Serialize_g(S))
```

空集合使用规范终态 `(Flat, ∅)`，成本为零。Tokenizer 通过配置中的冻结 builder tokenizer 和 chat template 计算，因此输入指令、schema 和模板都会计入。

来源阶段尚未知道架构，策略使用安全上界：

```text
T_hat(S) = max_g T_g(S)
```

最终架构选择仍使用对应架构的精确成本。`Reach_b` 对规范递增来源序列的每个前缀检查预算，保证策略解码和部署写入不会走到越界终态。

预算 (T_g) 与生命周期成本 (C_i(g,S)) 分开：

- (T_g) 只决定构建输入是否可执行；
- (C_i) 记录真实写入生成 tokens 除以未来查询数，再加上每个未来查询的回答生成 tokens。

训练成本项使用成功终态记录中的 `lifecycle_cost`；旧的试运行记录若没有该字段，训练代码才回退到精确构建输入成本。

## 4. 离线监督编译

离线更新单元包含候选来源、未来查询及其参考答案。教师模型只负责把查询需求映射为来源支持包，例如：

```text
D_i = {d_1, ..., d_R}
W_rk ⊆ E_i
```

每个需求最多三个替代支持包。支持包只用于缩小来源候选域，不会进入部署策略输入。

对每个候选 (S) 和五种架构 (g)，编译器执行：

```text
m        = Build_g(S; G)
M_g,S    = Update(M_i^-, m)
for q in Q_i:
    R_q    = Retrieve(q, M_g,S)
    y_hat  = Answer(q, R_q)
```

随后记录：

- 单查询效用 `query_scores`；
- 平均效用 `utility`；
- 查询覆盖率 `coverage`；
- 构建和回答 usage；
- 生命周期成本 `lifecycle_cost`；
- 三层认证状态和原因。

设查询效用阈值为 `tau_q`，平均效用阈值为 `tau_U`，覆盖率阈值为 `tau_coverage`。方案只有在三层认证均为 `PASS`、平均效用不低于 `tau_U`、覆盖率不低于 `tau_coverage` 时才进入成功档案：

```text
W_b(i) = {(S, g): status(S, g) = PASS and Reach_b(S, g) = 1}
```

同一来源集合在多个架构下成功时，所有成功配对都会保留。`UNKNOWN` 和未评估方案不进入成功档案，也不被当作失败标签。正式编译默认评估完整有限候选域；`limit` 仅是显式试跑上限，若候选域被截断，输出中的 `domain_complete` 为 `false`。

## 5. 内容优先选择器

选择器由 `arc.algorithm.selector.Selector` 实现，联合概率分解为：

```text
P(S, g | E, b) = p_phi(S | E, b) * p_psi(g | S, E, b)
```

### 5.1 写入阶段特征

`FeatureSchema` 使用冻结来源向量和可观察元数据生成特征。写入阶段不使用缓存 query vector；候选集合的向量中心只作为集合上下文表示。特征包含：

- 来源向量和候选集合表示；
- 来源与集合上下文的交互和差异；
- 位置、长度、时间、检索分数、会话和说话人；
- 当前预算特征。

因此同一个 (E,b) 在传入不同 query 字符串时，策略特征不改变。

### 5.2 来源解码 (S)

来源头维护规范递增前缀 (S_t)，合法动作是未选来源 `Pick(j)` 和 `STOP`：

```text
A_b(S_t) = {j > last(S_t): T_hat(S_t ∪ {j}) ≤ b - margin} ∪ {STOP}
```

束搜索保留宽度为 `beam_width` 的活动前缀，并保留最多 `top_l` 个完成集合。所有在线合法动作都进入 softmax 分母，训练时不会把动作空间缩减为成功档案的后继动作。立即 `STOP` 对应 `(Flat, ∅)`。

### 5.3 架构解码 (g)

对每个完成的 (S)，架构头只在精确预算可行的架构集合中归一化：

```text
G_b(S) = {g: T_g(S) ≤ b}
```

架构分数以完整 (S) 的编码状态、候选集合表示和预算特征为条件。最终比较完整联合分数：

```text
(S_hat, g_hat) = argmax_S,g [log p_phi(S | E,b) + log p_psi(g | S,E,b)]
```

解码期间不执行候选构建或审核。

## 6. 多正例训练

对成功档案非空的单元–预算对，定义成功终态概率质量：

```text
M_theta(i,b) = Σ_(S,g)∈W_b(i) P_theta(S,g | E_i,b)
```

训练损失由两部分组成：

1. `-log M_theta`，让全部成功终态获得概率质量；
2. 成功终态内部的生命周期成本偏好，相对预算内最便宜成功方案归一化。

代码使用 log-sum-exp 计算第一项，并沿每个完整 `(S,g)` 配对回传梯度。不会重组来源标签和架构标签，也不会给 `UNKNOWN` 或未评估方案制造负例。

优化器是 Adam；训练输入中的每个单元和每个非空预算档案等权。训练阶段不重新调用构建器、回答模型或审核器，只读取编译记录。

## 7. 部署流程

### 7.1 写入

```text
write(X_i, M_i^-):
    E_i ← candidate_sources(X_i, M_i^-)
    (S_hat, g_hat) ← content_first_selector(E_i, b)
    if S_hat = ∅:
        return clone(M_i^-)
    block ← Build_g_hat(S_hat; G)       # exactly one builder call
    return Update(M_i^-, block)          # exactly one state transition
```

`Update` 把 builder 的本地来源 ID 映射回稳定 `source_id`。同一来源 provenance 的记忆会被新版本替换；不同 provenance 的记忆并存，便于追溯。

### 7.2 查询

```text
query(q, M_i^+):
    R_q   ← Retrieve(q, M_i^+)
    answer ← Answer(q, R_q)
    return answer
```

`Memory.__call__` 的 query 分支只执行一次 `Retrieve`，并把已检索条目交给固定回答模型。该分支不会调用 selector、builder 或 `Update`。

## 8. 代码对应关系

| 算法部件 | 实现位置 |
|---|---|
| 候选生成、稳定来源和持久状态 | `src/arc/agent/persistent.py` |
| `Build` 序列化、解析和三层认证 | `src/arc/algorithm/memory.py` |
| 五种架构、拓扑和结构检查 | `src/arc/algorithm/architecture.py` |
| 候选域、预算可达性和三态编译 | `src/arc/algorithm/core.py`, `src/arc/algorithm/compiler.py` |
| query-independent 特征 | `src/arc/algorithm/features.py` |
| `S→g` 解码和多正例训练 | `src/arc/algorithm/selector.py` |
| 写入/查询阶段边界 | `src/arc/algorithm/adapter.py` |
| 会话历史到离线候选输入的适配 | `src/arc/agent/data/requirements.py` |
| 固定 Retrieve 后的回答调用 | `src/arc/agent/runtime.py` |

该实现说明不改变 `docs/papers/3_method.md` 的文本，也不依赖已经删除的实验章节。
