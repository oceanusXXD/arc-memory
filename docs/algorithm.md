# ARC 算法实现说明

本文档说明当前代码仓库如何实现 ARC。理论定义、符号约定和论文层面的表述以 [`docs/papers/3_method.md`](papers/3_method.md) 为准；本文档只解释这些定义在代码中如何被执行，不引入额外方法设定。

ARC 的实现遵循四条不可违反的边界：

1. **写入策略只使用写入时可见的信息。**
   策略输入是候选来源集合和预算，不接收当前查询、未来查询或参考答案。

2. **决策顺序是内容优先。**
   策略先选择来源集合 \(S\)，再在给定 \(S\) 的条件下选择组织架构 \(g\)。

3. **一次写入最多执行一个最终方案。**
   写入阶段最多进行一次 `Build` 和一次 `Update`；不会尝试多个候选方案后再选择。

4. **查询阶段不重新构建记忆。**
   查询时只执行固定的 `Retrieve` 和回答模型，不调用策略、构建器或更新器。

---

## 1. 状态与候选来源

第 \(i\) 个写入单元由写入前记忆状态 \(\mathcal M_i^{-}\)、新到达历史批次 \(X_i\) 和构建输入预算 \(b\) 组成。候选生成器由 `arc.agent.persistent.candidate_sources` 实现：

```text
E_i = C(X_i, M_i^-)
```

候选生成遵循以下规则：

- 新到达的历史按输入顺序加入候选集合；
- 已有记忆中的来源只能通过可观察的文本重叠和确定性的近期邻域加入；
- 候选生成不读取查询、参考答案、需求标签或查询向量。

每个候选来源保留以下信息：

- 本次策略解码使用的本地整数 `id`；
- 用于持久化和后续更新的稳定 `source_id`；
- 原始文本、会话、时间、说话人、位置、冻结向量及其他可观察元数据。

离线监督编译器也遵循相同的信息边界。正式输入由会话历史和冻结来源向量构造；已经物化的候选记录仍可通过 `candidate_sources` 或 `sources` 字段重放。

---

## 2. 架构与构建器

ARC 的架构集合固定为五类：

| 架构 | 结构约束 |
|---|---|
| `Flat` | 由独立、具有来源支持的事实或事件组成，不建立显式边。 |
| `Chain` | 按规范来源顺序形成相邻路径。 |
| `Tree` | 单根、无环；每个非根节点恰有一个父节点。 |
| `Graph` | 仅建立同一会话内相邻来源之间的可观察关系边。 |
| `Cluster` | 按可观察会话分组；簇之间不重叠，并覆盖全部节点。 |

`arc.algorithm.architecture.instantiate_structure` 根据来源元数据生成固定拓扑；`validate_structure` 负责检查节点覆盖、重复节点、重复边、自环、树根约束和簇覆盖等结构条件。

对于选定来源集合 \(S\) 和架构 \(g\)，`serialize_input` 构造完整的构建输入：

```text
Serialize_g(S) = instruction + schema + architecture_contract + structure + sources
```

因此，来源文本、来源标识和架构字段都会进入序列化结果；查询不会进入序列化输入。`_builder_content` 也会明确要求构建模型不得使用 query 或 future query。

对于非空 \(S\)，`build_once` 会调用一次冻结构建器。构建输出必须是最多六个对象组成的 JSON 数组，每个对象严格采用以下格式：

```json
{"m": "source-supported memory", "src": [1, 2]}
```

其中，`src` 只能引用当前 \(S\) 中的本地来源 ID。

构建器调用后，系统执行 JSON/schema 解析，并进行三层认证：

1. **结构认证**：检查根类型、字段格式、来源 ID 和架构拓扑是否合法；
2. **来源忠实性认证**：检查生成记忆是否能够由输入来源支持；
3. **需求完整性认证**：检查生成记忆是否覆盖离线标注的信息需求。

认证状态只有三种：

- `PASS`：认证通过；
- `FAIL`：存在明确违规，例如解析失败或 schema 违反；
- `UNKNOWN`：API 错误、服务中断或认证信息不足。

`UNKNOWN` 不会被改写为训练负例。

---

## 3. 预算与成本

对于非空方案，精确的构建输入长度定义为：

```text
T_g(S) = frozen_builder_tokenizer(Serialize_g(S))
```

空集合使用规范终态 `(Flat, ∅)`，其构建输入成本记为零。

Tokenizer 由配置中冻结的 builder tokenizer 和 chat template 决定，因此指令、schema、架构字段和模板都会被计入预算。

在来源选择阶段，架构尚未确定，因此策略使用安全上界：

```text
T_hat(S) = max_g T_g(S)
```

最终选择架构时，系统仍会使用对应架构下的精确成本 \(T_g(S)\)。`Reach_b` 会检查规范递增来源序列的每一个前缀是否满足预算，从而保证策略解码和部署写入不会走到越界终态。

预算 \(T_g(S)\) 与生命周期成本 \(C_i(g,S)\) 是两个不同概念：

- \(T_g(S)\) 仅用于判断构建输入是否可执行；
- \(C_i(g,S)\) 记录真实写入生成 token 按未来查询数摊销后的成本，再加上每个未来查询的回答生成 token 成本。

训练中的成本项优先读取成功终态记录中的 `lifecycle_cost`。只有旧试运行记录缺少该字段时，训练代码才会回退到精确构建输入成本。

---

## 4. 离线监督编译

离线更新单元包含候选来源、未来查询及其参考答案。教师模型只在离线阶段使用，用于将查询需求映射为来源支持包：

```text
D_i = {d_1, ..., d_R}
W_rk ⊆ E_i
```

每项需求最多保留三个替代支持包。支持包仅用于缩小离线候选域，不会作为部署策略的输入。

对每个候选来源集合 \(S\) 及五种架构 \(g\)，编译器执行：

```text
m        = Build_g(S; G)
M_g,S    = Update(M_i^-, m)

for q in Q_i:
    R_q    = Retrieve(q, M_g,S)
    y_hat  = Answer(q, R_q)
```

编译器会记录：

- 单查询效用 `query_scores`；
- 平均效用 `utility`；
- 查询覆盖率 `coverage`；
- 构建与回答调用的 usage；
- 生命周期成本 `lifecycle_cost`；
- 三层认证状态及其原因。

设单查询效用阈值为 \(\tau_q\)、平均效用阈值为 \(\tau_U\)、覆盖率阈值为 \(\tau_{\mathrm{cov}}\)。一个方案只有同时满足以下条件时，才会进入成功档案：

- 三层认证均为 `PASS`；
- 平均效用不低于 \(\tau_U\)；
- 查询覆盖率不低于 \(\tau_{\mathrm{cov}}\)；
- 满足预算可达性。

形式上：

```text
W_b(i) = {(S, g): status(S, g) = PASS and Reach_b(S, g) = 1}
```

如果同一来源集合在多个架构下都成功，则所有成功的 \((S,g)\) 配对都会保留。`UNKNOWN` 和未评估方案均不会进入成功档案，也不会被当作失败标签。

正式编译默认评估完整的有限候选域。`limit` 仅用于显式试跑；若候选域被截断，输出中的 `domain_complete` 将设为 `false`。

---

## 5. 内容优先选择器

选择器由 `arc.algorithm.selector.Selector` 实现，联合概率分解为：

```text
P(S, g | E, b) = p_phi(S | E, b) * p_psi(g | S, E, b)
```

即：先选择来源集合 \(S\)，再在给定 \(S\) 的条件下选择架构 \(g\)。

### 5.1 写入阶段特征

`FeatureSchema` 使用冻结来源向量和可观察元数据构造特征。写入阶段不会使用缓存 query vector；候选集合向量中心只作为集合上下文表示。

特征包括：

- 来源向量和候选集合表示；
- 来源与集合上下文之间的交互与差异；
- 位置、长度、时间、检索分数、会话和说话人信息；
- 当前预算特征。

因此，对于同一个 \((E,b)\)，即使传入不同 query 字符串，策略特征也不会改变。

### 5.2 来源解码：选择 \(S\)

来源头维护规范递增前缀 \(S_t\)。合法动作包括未选来源的 `Pick(j)` 和 `STOP`：

```text
A_b(S_t) = {j > last(S_t): T_hat(S_t ∪ {j}) ≤ b - margin} ∪ {STOP}
```

束搜索保留宽度为 `beam_width` 的活动前缀，并保留最多 `top_l` 个完成集合。

所有在线合法动作都会进入 softmax 分母。训练时，动作空间不会被缩减为成功档案中的后继动作。立即执行 `STOP` 对应规范空终态 `(Flat, ∅)`。

### 5.3 架构解码：选择 \(g\)

对于每个完成的 \(S\)，架构头仅在精确预算可行的架构集合中归一化：

```text
G_b(S) = {g: T_g(S) ≤ b}
```

架构分数以完整 \(S\) 的编码状态、候选集合表示和预算特征为条件。最终选择完整联合概率最高的方案：

```text
(S_hat, g_hat) = argmax_S,g [log p_phi(S | E,b) + log p_psi(g | S,E,b)]
```

解码阶段不会执行候选构建或审核；只有最终选中的方案会在部署写入时被真正构造。

---

## 6. 多正例训练

对于成功档案非空的单元—预算对，定义模型分配给全部成功终态的概率质量：

```text
M_theta(i,b) = Σ_(S,g)∈W_b(i) P_theta(S,g | E_i,b)
```

训练损失由两部分组成：

1. `-log M_theta`：提高模型对全部成功终态的总概率质量；
2. 成功终态内部的生命周期成本偏好：相对于预算内最低成本成功方案进行归一化。

代码使用 log-sum-exp 计算第一项，并沿完整的 \((S,g)\) 配对回传梯度。因此：

- 来源选择标签与架构选择标签不会被重新组合；
- `UNKNOWN` 和未评估方案不会被构造成负例；
- 训练阶段不会重新调用构建器、回答模型或审核器，只读取离线编译记录。

优化器为 Adam。训练时，每个更新单元和每个非空预算档案等权处理。

---

## 7. 部署流程

### 7.1 写入阶段

```text
write(X_i, M_i^-):
    E_i ← candidate_sources(X_i, M_i^-)
    (S_hat, g_hat) ← content_first_selector(E_i, b)

    if S_hat = ∅:
        return clone(M_i^-)

    block ← Build_g_hat(S_hat; G)       # exactly one builder call
    return Update(M_i^-, block)         # exactly one state transition
```

`Update` 会将 builder 输出中的本地来源 ID 映射回稳定 `source_id`。

对于相同 provenance 的记忆，新版本会替换旧版本；不同 provenance 的记忆会并存，从而保留可追溯性。

### 7.2 查询阶段

```text
query(q, M_i^+):
    R_q    ← Retrieve(q, M_i^+)
    answer ← Answer(q, R_q)
    return answer
```

`Memory.__call__` 的 query 分支只执行一次 `Retrieve`，并将检索结果交给固定回答模型。

查询阶段不会调用 selector、builder 或 `Update`。

---

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
