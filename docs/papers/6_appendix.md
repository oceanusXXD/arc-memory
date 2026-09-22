# 附录（完整展开版，与第 3、8 章对应；数值以 [] 占位）

## 附录 A　数据集、划分与更新单元

### A.1 LongMemEval-S

LongMemEval-S 由多会话用户–助手历史与面向该历史的问答组成，问题类型覆盖单跳事实、多跳聚合、时序推理、知识更新与跨会话综合等；历史长度达数万至十余万 tokens，来源（会话段）之间主题分散。主指标为官方 GPT-4o correctness judge 的 QA Accuracy（开放式答案按语义等价判定）。该数据集对 ARC 构成**低重叠**压力测试：不同查询的证据往往落在不同会话，核心重叠假设在此最接近不成立，因此用它标定方法的适用边界（§4.5.1、F.3）。规模与划分见 A.4。

### A.2 LoCoMo

LoCoMo 为超长期双人对话，附带事件摘要与 QA，类别 1–4 为单跳、多跳、时序与开放域问答，类别 5 为拒答（adversarial）。主指标为官方实现的 token-level QA F1（类别 1–4 汇总，类别 5 单独报告）。与 LongMemEval-S 相反，LoCoMo 的问答反复指向少量核心生活事件与人物关系，**核心证据重叠高**，是内容优先设计的正面验证域。

### A.3 数据集与方法的匹配性

两域构成假设的两端：LoCoMo 验证“单一持久方案服务多数查询”的正面情形，LongMemEval-S 验证重叠紧张时的退化行为与主题块备选方案（F.4）。该分层是预先指定的，不做事后挑选。

### A.4 划分与分组

按完整历史分组：来源重叠、同一事件及其改写归入同组，防止泄漏。划分在监督编译前冻结。

| 数据集 | 历史数（train/dev/test） | 会话数中位[] | QA 数（train/dev/test） | 历史 tokens 中位[] |
|---|---|---:|---|---:|
| LongMemEval-S | [] | [] | [] | [] |
| LoCoMo | [] | [] | [] | [] |

### A.5 更新单元与未来查询集合

**主配置（单单元）**：每历史一个更新单元，\(\mathcal M_i^{-}=\varnothing\)，\(X_i=\) 完整历史；\(\mathcal Q_i=\) 该历史训练侧 QA（与 dev/test 严格不相交）。**序贯多单元**（F.2）：按时间切批次；离线监督构造时 \(\mathcal M_i^{-}\) 由固定参考写入协议（Flat＋参考相关性前 \(k_{\mathrm{ref}}\)=[] 来源）产生，部署时为 ARC 自身前序写入；二者错配在 F.2 报告。**主配置下未触发的机制**：附录 C.3 的同源替换、冲突双记与 late-wins，以及跨单元预算累计，仅在序贯多单元设置（F.2）中生效；主表结果均为单单元写入，“持续演化”的机制主张以 F.2 的序贯设置为准。

**未来查询–支持证据标注流水线**：(1) 标注 LLM（[]）从 \(\mathcal Q_i\) 与参考答案抽取信息需求 \(\mathcal D_i\)；(2) 对每个需求枚举最小支持包 \(W_{rk}\)（删任一来源即不足）；(3) 查询必要证据集 \(V_q\)＝其需求所选支持包之并（替代包歧义由人工裁定）；(4) 需求与 \(V_q\) 仅用于离线编译与诊断，不进入写入路径。提示骨架（完整模板见 J.5）：

```text
[需求抽取] 给定历史 H 与问答 (q, y*)：列出回答 q 所需的最小信息需求 d_1..d_R；
对每个需求，给出 H 中能支持它的最小来源组合（可多组替代）。
[约束] 不得引用 H 之外信息；若 H 不足，输出 MISSING 并说明缺什么。
```

人工抽查 [] 个单元：需求级一致率 []，支持包级一致率 []，MISSING 裁定一致率 []。

### A.6 清洗与类别分布

删除：无支持来源的 QA、纯世界知识可答 QA、与历史冲突 QA；保留检索遗漏记录（不修补 \(E_i\)）。类别分布：[]。删除判定由标注 LLM（B.1）执行，人工复核比例 []，类别一致率 []。**未过滤集合对照**：ARC 与 Full Context 在未删除集合上的两行结果 []（绝对值与公开报告值的锚定以此为准，措辞见 §4.1.1）；按类别分层的删除强度敏感性 []。

## 附录 B　实现、候选生成与预算机制

### B.1 模型、硬件与版本号

| 组件 | 模型 / 版本 | 角色 |
|---|---|---|
| 构建器 G | [] | 冻结，五架构 schema 构建 |
| 回答模型 F | [] | 冻结，query-time 回答与离线效用 |
| 冻结 embedding | Qwen3-Embedding-0.6B, 版本 [] | 来源表示 |
| 审核/标注 LLM | [] | 三态审核与需求标注 |
| 硬件 | [] | 编译与训练 |

### B.2 候选生成器 C（完整协议）

**来源单元化。** LoCoMo：会话内连续 turn 组，≤[] tokens/单元；LongMemEval-S：会话内语义段（按话题切分，[]）。**显著性预筛**（\(|X_i|>n_{\max}=64\) 时）：

```text
Algorithm C-1  query-free 显著性预筛
1: 单元去重（来源标识 + 近重复文本 sim>[]=0.9 合并，保留最早时间戳）
2: 对每个单元 u 计算强化分 s(u) = Σ_{会话 s'≠s(u)} 1[u 的实体/时间戳在 s' 出现]
3: 按 s(u) 降序排列；同分按时间分层（每会话至少保留 1 个单元）取足 n_max
4: 输出保序候选单元集
```

**邻域与上限。** 从 \(\mathcal M_i^{-}\) 以 \(X_i\) 单元文本为键（BM25＋embedding，RRF）取前 \(k_{\mathrm{nb}}\)=[] 记忆块；按来源标识去重，\(|E_i|\le 64\)。敏感性（显著性 vs 均匀 vs \(n_{\max}=128\)）见 F.6。

**候选集质量诊断。** \(E_i\) 对 \(\bigcup_q V_q\) 的覆盖率 []；支持包引导与显著性预筛相对均匀抽样的增益 []；matched-\(E_i\) 基线（基线在同一 \(E_i\) 上训练与评价、不接触完整历史）质量 []；Algorithm C-1 的 \(|X_i|>n_{\max}\) 触发比例 []（LongMemEval-S 与 LoCoMo 分列）。

### B.3 保守掩码：完整证明与敏感性

**引理 B.1（BPE 次可加性）。** 对所用 tokenizer 与任意字符串 \(x,y\)：\(\operatorname{Tok}(x\oplus y)\le\operatorname{Tok}(x)+\operatorname{Tok}(y)\)。理由（经验）：\(x\) 与 \(y\) 的各自分词拼接是 \(x\oplus y\) 的一个合法分词；但**贪心 BPE 的结果不保证优于任意合法分词**（反例：merges 优先级 \(bc\succ ab\succ ab\mid c\to abc\) 下，整串贪心先合并 \(bc\) 后无法归并得 2 块，而合法分词 \(abc\) 仅 1 块），故“块数 ≤ 任一合法分词”不能作为一般论据，次可加性改按经验性质核验：[] 个随机拼接样本上违反数＝[]。硬预算保证不依赖本引理，由式（5）精确终检（引理 D.2）承担。

**命题 1（完整证明）。** 由 \(\operatorname{Serialize}_g(S)=A_g\oplus\operatorname{src}(S)\oplus B_g\) 与引理 B.1 两次应用：\(T_g(S)\le\operatorname{Tok}(A_g)+T^{\mathrm{src}}(S)+\operatorname{Tok}(B_g)=T^{\mathrm{src}}(S)+o_g\le T^{\mathrm{src}}(S)+\max_{g'}o_{g'}=\hat T(S)\)。故掩码 \(\hat T(S_t\cup\{e_j\})\le b-m\)（\(m\ge0\)）下，任何完成 \(S\) 对全部 \(g\) 满足 \(T_g(S)\le b-m\le b\)；最终 \((S,g)\) 另由式（5）精确确认。裕量 \(m\) 仅抵御 tokenizer 版本漂移，主配置 \(m\)=[]。∎

**实测与敏感性。** \(o_g\) 表：Flat []、Chain []、Tree []、Graph []、Cluster [] tokens。\(m'\in\{0,2,4\}\)：可行性违反 []/[]/[]，PASS@1 []/[]/[]。**保守性代价**：相对逐架构 oracle 掩码，可行集合平均缩小 [] 个来源，PASS@1 差 []。

### B.4 网络与超参数

| 类别 | 项 | 配置 |
|---|---|---|
| 来源编码器 | 结构 | 2 层 Transformer，隐 []，头 []；输入＝冻结 embedding⊕元数据 MLP（会话/时间/位置/长度） |
| 来源编码器 | 递归状态 \(h_t\) | GRU，[] 维 |
| 来源编码器 | 评分头 / 架构头 | 按式（17）/ 式（21）；\(u_g\) 为 [] 维参数向量 |
| 来源编码器 | 总参数量 | []（不含冻结组件） |
| 优化 | 优化器 / lr / warmup / 步数 | Adam / [] / [] / [] |
| 优化 | 种子 | []（汇总规则见 H.3） |
| 目标 | 成本权重 \(\lambda_c\) | [] |
| 目标 | 阈值 \(\tau_U,\tau_q,\tau_{\mathrm{cov}}\) | 开发集冻结，数值见 D.1 |
| 解码 | 束宽 \(w\) / Top-\(L\) | [] / [] |
| 解码 | 部署方式 | argmax（式（27）），无采样 |
| 筛查 | 代理查询数 \(m_0\) / 裕量 \(\Delta\) | [] / []（开发集冻结；\(\Delta\) 取经验 \(\gamma\) 的 0.9 分位，见 E.10） |
| 构建器 | \(G\) 的解码协议 | 五架构一致，参数 []（冻结，B.1） |

### B.5 预算配置

\(\mathcal B=\{1024,2048,4096\}\)；\(L_G\)=[]，\(I_G\)=[]，满足 \(b+L_G\le I_G\)（逐预算核验 []）。\(T^{\mathrm{src}}(S_t)\) 以增量 tokenizer 调用精确维护（每候选 Pick 一次，均时 [] ms）。

### B.6 缓存与索引

来源 embedding 预计算命中率 []；记忆块索引在 Update 后增量重建，均时 [] ms。

### B.7 成本测量协议（预算与成本分离的工程实现）

**计入项。** 写入路径：构建器全部生成调用（输入含序列化来源＋提示＋schema＋聊天模板；输出含后端实际推理 tokens）。读取路径：回答模型全部生成调用（in/out，含系统触发的重试）。**不计入生成 tokens 的项**：embedding、索引检索等非生成式调用（其延迟在 G.4 单独记录）；离线审核/标注调用（单独账本，不进部署成本）。**来源与核对**：计数取自后端 usage 字段，与完整调用日志逐调用核对（误差容忍 0；不一致记 UNKNOWN）。**与预算的分离**：\(T_g\) 仅由 tokenizer 事前计算（B.5），与成本分列记录、互不推导；式（7）的 \(C_i\) 只使用真实使用 tokens。**一次构造、三档过滤的归因**：构造 tokens 只计一次（按 \(\max_b I_G\) 的输入上限执行，不按预算重复计入）；逐预算过滤为确定性集合操作、不产生生成调用，不进入 \(I_G\) 与部署成本账本（见 C.5）。

## 附录 C　五类架构 schema 与 Update（完整）

### C.1 公共序列化 src(S)

```json
{"sources":[{"id":"e3","text":"...","session":2,"time":"2023-05-01T10:00","len":87}]}
```

### C.2 各架构模板、schema 与合法性检查

**Flat。** 提示要点：抽取原子事实条目，保留时间与来源，不建边。输出 `{"entries":[{"id","fact","source_ids","time"}]}`。Post：① `source_ids⊆S` 且非空；② 时间字段与来源一致；③ 无跨条目合并事实。

**Chain。** 提示要点：事件/状态节点组成单一路径，相邻边表达有来源支持的顺序或状态演进。输出 `{"nodes":[...],"edges":[{"from","to","type":"next"}]}`。Post：① edges 构成单一路径；② 节点时间非递减；③ 每边共享来源或时间连续。

**Tree。** 提示要点：唯一根；非根恰一父；上层概括可追溯到下层证据。输出 `{"root":id,"nodes":[{"id","content","parent","source_ids"}]}`。Post：① 唯一根、无环、单父；② 叶 `source_ids⊆S`；③ 非叶节点含可追溯字段指向子来源之并。

**Graph。** 提示要点：实体/事件节点＋有类型稀疏关系边；每条边必须有证据来源。输出 `{"nodes":[...],"edges":[{"from","to","relation","evidence_id"}]}`。Post：① `evidence_id∈S`；② 度数 ≤ \(d_{\max}\)=[]；③ 关系类型在固定集合内。

**Cluster。** 提示要点：互不重叠主题组；组描述由成员支持。输出 `{"clusters":[{"id","topic_desc","member_ids"}]}`。Post：① members 互斥且并 ⊆ S；② `topic_desc` 支持来源 ⊆ 其 members。

所有架构：允许退化实例（如空边集）；生成内容不可回溯到 `source_ids` 即判忠实性失败。五架构的完整提示模板见 J.1；任务头注入 C.1 的序列化来源与 schema 常数（如 \(d_{\max}\)、关系类型集）。

### C.3 Update 协议（伪代码）

```text
Algorithm U  Update(M^-, m; g, S)
1: blk ← hash(g, S.ids, config)
2: 若 blk 已存在：幂等返回 M^-
3: 同源重复：m 中与旧块同来源 id 的条目替换旧条目
4: 冲突：同实体/同时间键冲突 ⇒ 两者保留，conflict=true，附时间戳（读取阶段 late-wins）
5: 更新 embedding 索引与元数据过滤字段；返回 M^+
```

### C.4 回退协议

Build 输出违反 schema ⇒ 记 FAIL，统一空记忆回退（不触发语义修复/重采样），已发生调用计入成本。

### C.5 级联筛查编译（算法 C-2）

```text
Algorithm C-2  CascadeScreen(E_i, Q_i, Q_i^0, b): certified cascade screening
Input: sources E_i, full query set Q_i, proxy subset Q_i^0 (|Q_i^0| = m_0,
       uniform subsample, seed frozen on dev), budget b, margin Delta
Output: verified archive W_b(i) with the SAME admission rule as full compilation

1. Enumerate candidate content subsets S_1..S_K (support-package guided,
   incl. empty and full-set references)                       // same as B.2
2. Gate 0 (zero-cost necessary-condition pruning), per subset S:
   2a. if exists need r with NO package W_rk ⊆ S (i.e. Φ_i(S)=0):
       prune S for all g // P7 MISSING under closure assumption; counts in E.10
   2b. for each g: compute T_g(S) by deterministic serialization
       plus tokenizer count (no generation call);
       if T_g(S) > b: mark (S,g) budget-infeasible
3. Gate 1 (Flat proxy probe), per surviving S:
   3a. build M0_S = Build_Flat(S) with P1-Flat; this artifact IS the
       official Flat construction of S (counts once, reused in Gate 2)
   3b. evaluate u_{i,q}(Flat,S) on q in Q_i^0 with the P8/P9 pipeline
   3c. U0(S) = mean over Q_i^0
   3d. if U0(S) < tau_U - Delta: prune S for all remaining g
4. Gate 2 (full verification), per surviving S and budget-feasible g:
   4a. if g = Flat: reuse M0_S, else build M^{g,S} with P1-g
   4b. evaluate the remaining queries Q_i \ Q_i^0 (reuse Q_i^0 scores)
   4c. run the 3-state audit (C.2 deterministic checks + P6/P7)
   4d. admit (g,S) iff status=PASS and Reach_b(g,S)=1
       // 式(16) unchanged rule（Reach_b=前缀可达预算检查，式(15)）
5. Assemble W_b(i) per budget b; record the pruning gate of every
   skipped candidate (used by the replay analysis in E.10).
```

说明：Gate 0(a) 的无假阴性依赖支持包标注对可替代来源的封闭性，被错剪方案数与空档案单元比例由 E.10 报告。Gate 0 两类检查均为确定性集合与计数操作，不消耗生成调用；Gate 1 的 Flat 构造与其在 Gate 2 的正式构造为同一产物，代理子集评估全额复用；唯一净增开销是 Gate 1 被剪候选的 Flat 构造（其替代了对应的全架构构造），存活者的 Flat 构造不构成额外成本。多预算处理：Gate 0(a) 与 Gate 1 和预算无关，只执行一次；Gate 0(b) 与 Gate 2(d) 逐预算执行。随机性来源：\(\mathcal Q^0_i\) 的抽取（种子冻结，H.3）与构建器解码；同一 \(S\) 的 Flat 构造只采样一次并全程复用，避免级联各级之间的产物漂移。

## 附录 D　监督编译细节与完整证明

### D.1 阈值协议与数值

在开发集上按“PASS 率∈[[],[]] 且 Cov 分布稳定”选 \(\tau_U\)=[]，\(\tau_q\)=[]，\(\tau_{\mathrm{cov}}\)=[]；冻结后测试不改。因 \(\tau_U\) 由目标 PASS 率反推，它是**相对分位**而非绝对质量线，其绝对参照（如相对 Full Context 效用的分位）[]。\(\tau_q\) 的语义：单查询“有效服务”的判定线；\(\tau_{\mathrm{cov}}\) 的语义：允许被差服务的查询比例上限（命题 2）。

### D.2 加权推广与类别分层

\(U_i^{w}=\sum_j w_{ij}u_{i,q}\)（\(\sum w=1\)）；主配置均匀。LoCoMo 分层配置：四类分别设阈值，PASS 取最小值；结果 []。

### D.3 审核协议

三清单：(1) 结构合法性＝C.2 Post 检查（确定性）；(2) 忠实性＝逐生成事实核对支持来源（含时间限定、冲突处理），审核 LLM=[]，提示冻结（全文见 J.2）；(3) 完整性＝逐项核对 \(W_{rk}\) 信息能否由记忆读出。重复构建稳定性：同 \((g,S)\) 重构 [] 次，schema 一致率 []，忠实判定一致率 []。人工抽查 [] 方案，人–机一致率 []，κ=[]；记录中 UNKNOWN 占比 []，回退处置执行率 []。

### D.4 完整证明

**引理 D.1（集合–轨迹双射与终止）。** 掩码中 Pick 满足 \(j>j_t\)，故轨迹 id 严格递增；STOP 恒可用，过程至多 \(n\) 步终止。映射 \(S\leftrightarrow(a_1,\ldots,a_k,\mathrm{STOP})\) 双射：递增序列唯一确定集合，集合的规范序唯一确定序列。因此 \(p_\phi(S\mid E,b)\) 与顺序选择无关，且 \(\sum_{S}p_\phi(S)+\)（未终止质量＝0）\(=1\)（每步 π 在含 STOP 的合法集上归一）。∎

**命题 2（证明）。** 设 \(B=\{q:u_{i,q}<\tau_q\}\)。若 \(|B|>(1-\tau_{\mathrm{cov}})|\mathcal Q_i|\)，则 \(\operatorname{Cov}_i=1-|B|/|\mathcal Q_i|<\tau_{\mathrm{cov}}\)，矛盾。∎

**注记 D.2（覆盖率不可微，作用于编译层）。** \(\operatorname{Cov}\) 含指示函数，不参与 \(\nabla_\theta\)；它以档案准入条件（14）的形式起作用。因此训练目标对 θ 光滑，覆盖率的“阈值效应”体现在档案构成而非梯度。

**命题 3（偶然通过界，完整证明）。** 固定一个方案 \((g,S)\)，其记忆 \(\mathcal M_i^{g,S}\) 固定；单查询效用 \(u_{i,q}\in[0,1]\) 的随机性来自未来查询的抽取，视 \(\mathcal Q_i\) 为未来查询分布的样本。若真实期望 \(\bar u_{\mathrm{true}}\le\tau_U-\delta\)，Hoeffding 给出 \(\Pr[U_i\ge\tau_U]\le\exp(-2|\mathcal Q_i|\delta^2)\)。覆盖率侧：指示变量 \(\mathbf1[u_{i,q}\ge\tau_q]\) 为 \([0,1]\) 有界变量，真实均值 \(\pi_{\mathrm{true}}<\tau_{\mathrm{cov}}\) 时同理由 Hoeffding 得 \(\Pr[\operatorname{Cov}_i\ge\tau_{\mathrm{cov}}]\le\exp(-2|\mathcal Q_i|(\tau_{\mathrm{cov}}-\pi_{\mathrm{true}})^2)\)。对 \(|\Omega_i|\) 个方案、每方案两个事件取 union bound 即得正文不等式。保守性讨论：(a) 同一记忆下不同查询的效用可对相关，但界只要求**同一方案内**样本有界独立（或对固定集合评价时改用无放回抽样的 Hoeffding 形式，指数不变）；(b) 跨方案的相关性由 union bound 吸收，不作独立假设；(c) 若查询抽取轻微违假设，\(|\mathcal Q_i|\) 敏感性（E.4）提供经验复核。该界说明 \(\tau_{\mathrm{cov}}\) 与 \(|\mathcal Q_i|\) 共同构成对候选域大小 \(\log|\Omega_i|\) 的对数级控制。**适用域。** 界严格适用于 \(\mathcal Q_i\) 为未来查询池随机子样本的情形（无放回时取 Hoeffding 无放回形式，指数不变）；当 \(\mathcal Q_i\) 为全部池内查询（主配置）时，\(U_i\)、\(\operatorname{Cov}_i\) 是池统计量而非样本估计，该界退化为对“池内多重比较”的控制，池外外推依赖未来查询与池的分布一致性（该假设与 E.4 敏感性一并报告）。∎

**命题 4（后验/EM 视角，完整证明）。** 定义观测为“成功事件” \(e\)，潜变量 \(z=(g,S)\in\mathcal W_b(i)\)，联合模型 \(q_\theta(e,z)=P_\theta(S,g)\)（成功由 \(z\) 落入档案而确定）。边缘似然 \(q_\theta(e)=\sum_{\mathcal W}P_\theta=M_\theta\)；后验 \(q_\theta(z\mid e)=P_\theta/M_\theta=\rho_\theta\)。于是 \(\nabla_\theta[-\log q_\theta(e)]=-\frac{1}{M_\theta}\sum_{\mathcal W}\nabla_\theta P_\theta=-\sum_{\mathcal W}\rho_\theta\nabla_\theta\log P_\theta=\mathbb E_{z\sim\rho_\theta}[-\nabla_\theta\log P_\theta(z)]\)，即边缘似然梯度的 E 步为对潜变量的**精确**后验期望（档案显式枚举 \(\mathcal W\)，无需采样），M 步为对已验证轨迹的 \(\rho\)-加权似然上升。与 RL 的区别：无在线交互、无奖励方差；与单标签 CE 的区别：不对 \(z\) 做硬指派；与对比学习的区别：不构造负例。∎

**命题 5（完整证明）。** 记 \(P_\theta(S,g)=p_\phi(S)p_\psi(g\mid S)\)，\(M_\theta=\sum_{\mathcal W}P_\theta\)。由 \(\nabla\log M_\theta=\nabla M_\theta/M_\theta\) 与 \(\nabla P_\theta=P_\theta\nabla\log P_\theta\)：
\(\nabla_\theta[-\log M_\theta]=-\sum_{\mathcal W}\frac{P_\theta}{M_\theta}\nabla_\theta\log P_\theta=-\sum_{\mathcal W}\rho_\theta\nabla_\theta[\log p_\phi+\log p_\psi]\)。
**推论 D.1（配对保持）。** 求和仅遍历 \(\mathcal W_b(i)\)：未验证组合 \((S_1,\mathrm{Chain})\notin\mathcal W\) 既无正向项也无“失败标签”项；其参数只经由共享参数被已验证轨迹间接影响。**注记 D.1（归一化竞争）。** 因 π 与 softmax 的分母含全部合法动作，提高成功轨迹概率会相对压低失败轨迹概率，这是预期的概率质量再分配，不等于给失败方案赋梯度标签。∎

**命题 6（监督–部署桥梁，完整证明）。** 使用一致性（§3.4.3 注记）保证部署管线与认证管线相同，故采样解码 \((S,g)\sim P_\theta\) 的部署成功事件等价于 \((S,g)\in\mathcal W_b(i)\)，其概率为 \(\sum_{\mathcal W}P_\theta=M_\theta\)，即 \(-\log M_\theta\) 为采样解码下部署成功率的 NLL。对束/argmax 解码器 \(D\)：补集（失败方案）总质量为 \(1-M_\theta\le\varepsilon\)；采样解码失败概率 \(\le\varepsilon\) 精确成立；argmax 解码在 \(M_\theta\to1\) 时输出落在 \(\mathcal W\) 的概率 \(\to1\)（否则其输出质量将超过补集上界）。有限样本下二者差距由 RQ3 的 SS-NLL 与 PASS@1 互证度量。∎

**命题 7（证明）。** 式（25）第二项求和域为 \(\mathcal W_b(i)\)，失败方案成本不出现；\(\rho_\theta\) 在 \(\mathcal W\) 上归一。故对失败方案参数，成本项梯度恒为 0；第一项对失败方案概率的梯度方向为“移开质量”（因总质量归一至成功集）。不存在使失败方案因便宜而概率升高的项。∎

**推论 1（认证效用下界，证明）。** 采样解码下按“档案内/外”分解期望：\(\mathbb E[U_i]=M_\theta\,\mathbb E[U_i\mid\mathcal W]+(1-M_\theta)\,\mathbb E[U_i\mid\mathcal W^{c}]\)。由准入式（14），\(\mathcal W\) 上 \(U_i\ge\tau_U\)；效用取值于 \([0,1]\)，故 \(\mathbb E[U_i]\ge M_\theta\tau_U\)。等号成立当且仅当 \(\mathbb E[U_i\mid\mathcal W]=\tau_U\) 且档案外效用恒为 0，即下界在“档案方案恰达阈值、失败方案完全无效”时最紧；实际紧度由 E.5 实测（实现效用与下界之差）。单调性：下界随 \(M_\theta\) 单调增，故 \(-\log M_\theta\) 同时最大化该下界。成本侧：\(\mathbb E[C_i\mid\mathcal W]=\mathbb E_{\rho_\theta}[C_i]\) 正是式（25）第二项所塑造的量。∎

**命题 8（写入摊销，完整证明）。** \(\mathcal C(N)=C^{\mathrm{wr}}/N+\bar C^{\mathrm{rd}}\)，\(d\mathcal C/dN=-C^{\mathrm{wr}}/N^2<0\) 单调递减。对照按查询构造基线 \(C_{\mathrm{PQ}}=C^{\mathrm{wr}}+\bar C^{\mathrm{rd}\,'}\)：\(\mathcal C(N)\le C_{\mathrm{PQ}}\iff C^{\mathrm{wr}}(1-1/N)\ge\bar C^{\mathrm{rd}}-\bar C^{\mathrm{rd}\,'}\)。若 \(\bar C^{\mathrm{rd}}\le\bar C^{\mathrm{rd}\,'}\)，右端非正，对一切 \(N\ge1\) 成立（\(N>1\) 严格）。若 \(\bar C^{\mathrm{rd}}>\bar C^{\mathrm{rd}\,'}\)，需 \(1-1/N\ge(\bar C^{\mathrm{rd}}-\bar C^{\mathrm{rd}\,'})/C^{\mathrm{wr}}\)，即 \(N\ge N^*=\left\lceil C^{\mathrm{wr}}/(C^{\mathrm{wr}}-(\bar C^{\mathrm{rd}}-\bar C^{\mathrm{rd}\,'}))\right\rceil\)（分母为正）。ARC-PQ 与 ARC 使用同一冻结读取机制；其单查询读取成本实测偏差为零（\(\bar C^{\mathrm{rd}}=\bar C^{\mathrm{rd}\,'}\)，G.1 实测）时 \(N=1\) 即不劣、\(N>1\) 严格占优；偏差非零时以上述 \(N^*\) 交叉点为准。∎ **推论 D.2.** 优势随 \(N\) 单调增大（G.1 敏感性与交叉点实测）。

**命题 9（级联筛查的漏检界，证明）。** 固定内容子集 \(S\)，其 Flat 构造固定；单查询代理效用 \(u_{i,q}(\mathrm{Flat},S)\in[0,1]\)，随机性来自 \(\mathcal Q^0_i\) 的抽取，与命题 3 采用同一 i.i.d. 假定。记 \(\mu_F(S)=\mathbb E[u_{i,q}(\mathrm{Flat},S)]\)，错剪事件为 \(\bar U^0(S)<\tau_U-\Delta\)。若 \(\mu_F(S)\ge\tau_U-\Delta\)，令 \(\varepsilon(S)=\mu_F(S)-(\tau_U-\Delta)\ge0\)，则 Hoeffding 给出 \(\Pr[\bar U^0-\mu_F\le-\varepsilon(S)]\le\exp(-2m_0\varepsilon(S)^2)\)，即正文不等式；对全体满足 \(\mu_F\ge\tau_U-\Delta\) 的候选取 union bound 得期望错剪数上界。与命题 3 的关系：命题 3 约束入档侧（低价值方案偶然通过），其条件统计量只在 Gate 2 全额评估上计算，与级联剪除了谁无关；命题 9 约束召回侧（高价值方案被错剪），两界正交叠加。适用域与保守性：\(\mathcal Q^0_i\) 为 \(\mathcal Q_i\) 的均匀子样本（种子冻结）时抽样形式与命题 3 相同；查询间相关性按命题 3 的保守性讨论处理（无放回 Hoeffding，指数不变）。\(\mu_F<\tau_U-\Delta\) 区域无统计保证：该区域方案的入档价值只能来自架构收益 \(\gamma(S)>\Delta\)，界外行为由开发集冻结的 \(\Delta\) 与 E.10 的经验 \(\gamma\) 分布、档案一致性共同保障。覆盖率侧完全平行：以指示变量均值 \(\pi_F(S)\) 代替 \(\mu_F(S)\)、\(\tau_{\mathrm{cov}}\) 代替 \(\tau_U\)，同一 Hoeffding 论证成立；正文为简洁只陈述效用侧。∎


**引理 D.2（解码安全性）。** 沿掩码展开的任意前缀满足 \(\hat T\le b-m\)，由命题 1 对全部 \(g\) 有 \(T_g\le b\)；最终 \((\hat S,\hat g)\) 经式（5）精确确认。故部署路径无预算违反。∎

## 附录 E　基线适配、方法比较与补充结果

### E.0 方法比较总表（逐维度论证）

| 方法 | 决策位置 | 结构 | 内容选择 | 监督 | 预算 | 与 ARC 的本质差异 |
|---|---|---|---|---|---|---|
| Mem0/A-MEM/LightMem | 写入 | 单一/启发式 | 启发式 | ✗ | ✗ | 无学习联合决策、无预算 |
| MemBuilder/Mem-α/Memory-R1 | 写入 | 单一/多组件（Mem-α：core/episodic/semantic） | RL 策略 | RL 奖励（Memory-R1 为 EM/QA；Mem-α 为四项加权和，非单奖励） | ✗ | 无多正例、无硬预算 |
| MemGAS/MAGMA | 查询 | 多粒度/多图 | 查询阶段 | ✗ |  | 写入阶段不决策 |
| LeanMem | 写入＋查询 | 类型 | 写入阶段启发式 | ✗ |  | 无结果监督、无联合训练 |
| FluxMem | 查询 | 单结构学习 | ✗ | 单标签离线 | ✗ | 不筛内容、单正例 |
| BudgetMem | 查询 | ✗ |  | ✓ | 档位 | 无结构、查询阶段 |
| MESA | 查询 | 预构建视图 | 视图内子集 | 标量验证目标 | ✗ | 视图预构建无预算、标量不分层、查询阶段 |
| **ARC** | **写入** | **五架构联合** | **学习** | **执行认证多正例** | **硬写入** | — |

### E.1 逐基线适配细则

Full Context/RAG-RRF：\(\mathcal T_{\mathrm{wr}}=0\)。Mem0：保留 add/update；写入调用全计入 \(\mathcal T_{\mathrm{wr}}\)。LightMem：保留聚类摘要管道。MemGAS/MAGMA：保留查询阶段粒度/图操作。LeanMem：保留写入类型与查询阶段类型选择。FluxMem：保留其结构选择器，统一 F。BudgetMem：档位映射到 \(\mathcal B\)。MESA：视图预构建按其协议，选择保留查询阶段。扩展基线（A-MEM、CoM、Memory-R1、Mem-α、MemBuilder、MemChain、Mnemis、StructMem、LiCoMemory、SimpleMem）适配同规则，结果 []。

**重实现与原文报告值的一致性对照。** 各基线在原文设定下的报告值与本地重实现值逐行核对（协议见 §4.1.3）：

| 基线 | 原文报告值 | 本地重实现 | 差异 |
|---|---|---|---|
| LightMem | LME-S Acc 68.64、LoCoMo Acc 72.99（ACC；judge=GPT-4o-mini，backbone=GPT-4o-mini，各取原文最高组） | [] | [] |
| MemGAS | LME-S 4o-J 60.20 / F1 20.38；LoCoMo 4o-J 41.07 / F1 17.66（生成 GPT-4o-mini，judge=GPT-4o，Contriever） | [] | [] |
| MAGMA | LoCoMo J 70.0（judge=gpt-4o-mini，含 adversarial 第 5 类）；LongMemEval Acc 均值 61.2（gpt-4o-mini，>100K 上下文） | [] | [] |
| LeanMem | LME-S Acc 91.80、LoCoMo Acc 84.87（backbone=GPT-4.1-mini；LoCoMo 为 SimpleMem 协议 Acc、类 1–4，弃原生 F1） | [] | [] |
| FluxMem | LongMemEval 未报告（原文仅测 PersonaMem 与 LoCoMo）；LoCoMo 均值 F1 51.16 / B-1 41.73 / R-L 49.51（GPT-4.1，类 1–4，无 judge 指标） | [] | [] |
| BudgetMem | LoCoMo F1 43.05 / J 54.62；LongMemEval F1 40.24 / J 60.50（BudgetMem-Cap，LLaMA-3.3-70B，性能优先 λ=0） | [] | [] |
| MESA | LoCoMo token-F1 49.0（Qwen3-32B）；LongMemEval 未报告（主评测为 AMA-Bench） | [] | [] |

**原文值口径注（解读差异列时连同本注）：** 各家 judge、backbone 与类别口径互不一致，也与本文协议（官方 GPT-4o judge、LoCoMo 类 1–4 汇总 F1 与 J）不同，原文值仅作量级锚定，不可直接并置。LightMem 为 GPT-4o-mini judge 的 ACC、多组压缩设定取最高（其 LongMemEval 离线并行更新变体为 67.07，LoCoMo 按原表注为离线更新后）；MemGAS 为 GPT4o-as-Judge；MAGMA 的 LoCoMo judge 含 adversarial 第 5 类、LongMemEval 为 >100K 上下文的分题型均值；LeanMem 弃原生 F1、改报同族 GPT-4.1-mini judge 下 SimpleMem 协议 Acc（类 1–4）；FluxMem 不报 judge 且未测 LongMemEval；BudgetMem 为查询时按需抽取、记录 λ=0 性能优先点、原文未标 LongMemEval 子集；MESA 未测 LongMemEval。FluxMem 指 arXiv:2602.14038、BudgetMem 指 arXiv:2602.06025，均有同名不同文工作，核对时认准编号。

### E.2 其他预算主结果

\(b=1024/4096\) 下表 2 全表 []。

### E.3 ARC-PQ 与重叠分层

结果：Acc [] vs []（LME-S），F1 [] vs []（LoCoMo），Tokens/q [] vs []。按 \(J_i\) 高/低分层的质量差 []（预期低重叠组 ARC-PQ 优势更大）。

### E.4 \(|\mathcal Q_i|\) 敏感性

\(\{5,10,20,\text{全部}\}\)：PASS@1 []，主表质量 []，档案规模 []。\(\tau_U\)、\(\tau_{\mathrm{cov}}\) 判定对 \(|\mathcal Q_i|\) 的依赖由命题 3 给出（主配置取全池时按池内多重比较/经验校准解读，不构成对池外查询的概率控制；i.i.d. 抽样解读的适用域见 D.4；式为 \(2|\Omega_i|\exp(-2|\mathcal Q_i|\delta^2)\)），实测偏差记录于此。

### E.5 覆盖对照与配对差/前沿

诊断档案非空比例 []；匹配覆盖重算表 3 []。每预算配对差与前沿顶点 []。**推论 1 下界核验**：argmax 解码下实现效用 \(\bar U_i\ge M_\theta\tau_U\) 的满足率 []、平均裕度 []；采样解码版满足率 []（理论上恒成立，作为实现正确性检查）。argmax 解码的有限样本条件 \(M_\theta\ge\lvert\mathcal W\rvert/(\lvert\mathcal W\rvert+1)\) 满足率 []（与上合并报告）。回答模型迁移（档案冻结、换 \(F'\) 只重跑读取）：质量差 []，\(\operatorname{Score}\) 与测试指标的一致性 []。

### E.6 刷新实验

设置一 []；设置二 []（SS-NLL/PASS@1/开销前后对比）。

### E.7 补充指标

Recall@[]/NDCG@[] []；BLEU-1/judge []。

**LoCoMo 类别级拆分（本地统一协议；单元格＝F1 / B-1 / J，均 ↑）。**

| 方法 | Cat 1 多跳 | Cat 2 时序 | Cat 3 开放域 | Cat 4 单跳 | Average |
|---|---:|---:|---:|---:|---:|
| Full Context | [] / [] / [] | [] / [] / [] | [] / [] / [] | [] / [] / [] | [] / [] / [] |
| RAG-RRF | [] / [] / [] | [] / [] / [] | [] / [] / [] | [] / [] / [] | [] / [] / [] |
| Mem0 | [] / [] / [] | [] / [] / [] | [] / [] / [] | [] / [] / [] | [] / [] / [] |
| LightMem | [] / [] / [] | [] / [] / [] | [] / [] / [] | [] / [] / [] | [] / [] / [] |
| MemGAS | [] / [] / [] | [] / [] / [] | [] / [] / [] | [] / [] / [] | [] / [] / [] |
| MAGMA | [] / [] / [] | [] / [] / [] | [] / [] / [] | [] / [] / [] | [] / [] / [] |
| LeanMem | [] / [] / [] | [] / [] / [] | [] / [] / [] | [] / [] / [] | [] / [] / [] |
| FluxMem | [] / [] / [] | [] / [] / [] | [] / [] / [] | [] / [] / [] | [] / [] / [] |
| BudgetMem | [] / [] / [] | [] / [] / [] | [] / [] / [] | [] / [] / [] | [] / [] / [] |
| MESA | [] / [] / [] | [] / [] / [] | [] / [] / [] | [] / [] / [] | [] / [] / [] |
| **ARC (\(b=2048\))** | **[] / [] / []** | **[] / [] / []** | **[] / [] / []** | **[] / [] / []** | **[] / [] / []** |

**注。** 单元格三值＝token-level F1 / BLEU-1 / LLM-judge J，与正文双主指标一致（B-1 为词面精度视角，J 为语义判分；R-L 不报告）。类别 1–4 汇总，类别 5 拒答按 §8.1.1 协议单独报告、不计入 Average。全部数值由本地统一重实现协议产出；各家原文类别口径（4/5 类、judge 与骨干差异）与 E.1 一致，未对齐前不可跨表并置。

**图 A2　机制消融可视化（对应正文表 3）。** 数据：变体 × {SS-NLL, PASS@1, 成功条件成本差}，每数据集 []。图型：分面点区间图：每个数据集一个面板，横轴为变体，左轴 PASS@1（柱＋CI）、右轴 SS-NLL（折点），ARC 列加阴影。

### E.8 成本拆分

Write/Query-time Tokens、Calls 全表 []。

### E.9 定性案例

每个案例按“历史摘要 → 候选方案与三态审核结果 → 部署读取示例 → 成本拆分”四段呈现。

案例 1（多正例）：同一单元 \((S_1,\mathrm{Graph})\) 与 \((S_2,\mathrm{Chain})\) 均 PASS，内容骨架 []。案例 2（覆盖率排除）：均值 []≥τ_U 但 Cov=[]<τ_cov 被排除，差服务查询示例 []。案例 3（架构切换收益）：固定 \(S\) 下 Graph vs Best-Fixed Chain，ΔU=[]，ΔC=[]。

### E.10 级联筛查：离线复判协议与结果

**离线复判协议。** 全量编译日志已记录每个 \((g,S)\) 的审核三态、逐查询效用与真实构造/评估 token 开销，离线复判因此不新增任何调用：按算法 C-2 依序判定每个候选应进入的关卡，Gate 1 的代理均值由日志中 Flat 候选在 \(\mathcal Q^0_i\) 上的已记录效用计算（\(\mathcal Q^0_i\) 以冻结种子从 \(\mathcal Q_i\) 均匀抽取，种子见 H.3）。报告四项：(i) 调用与编译 tokens 节省（表 5，口径同 G.3）；(ii) 档案一致性 \(|\mathcal W^{\mathrm{casc}}_b\cap\mathcal W^{\mathrm{full}}_b|/|\mathcal W^{\mathrm{casc}}_b\cup\mathcal W^{\mathrm{full}}_b|\)，逐预算；(iii) 下游不变性：以级联档案重训后的 SS-NLL（式（31））与 PASS@1（式（32））相对全量档案训练的差值；(iv) 经验架构收益 \(\gamma(S)=\max_g U_i(g,S)-U_i(\mathrm{Flat},S)\) 的分布（均值、0.9 分位、最大值），核验命题 9 注记的 \(\Delta\) 选择。

**结果。** 构造调用节省 []%，评估调用节省 []%，编译 tokens 节省 []%；档案一致性：\(b=1024\) 为 []，\(b=2048\) 为 []，\(b=4096\) 为 []；\(\gamma\) 分布：均值 []，0.9 分位 []，最大值 []；下游差值：SS-NLL []，PASS@1 []；被错剪方案数 []（以全量档案核计）；其中因支持包标注非封闭被 Gate 0(a) 剪除、但完整性审核可通过的方案数 []，空档案单元比例 []。

## 附录 F　泛化、诊断与补充消融

### F.1 划分稳定性

三划分关键量 SD []。

### F.2 更长历史与序贯多单元

LongMemEval-M/拼接长历史 []；序贯配置质量差 [] 与参考–自写错配 []。

### F.3 重叠诊断（完整协议）

```text
Algorithm O-1  重叠诊断
1: 每单元取 V_q（A.5）
2: J_i ← 式(34) 的 pairwise Jaccard 均值
3: 共享核心覆盖率 ← max_{S: T̂(S)≤b} |{q: V_q⊆S}|/|Q_i|，贪心覆盖近似，近似比 []
4: 按问题类型分层重复 2-3
```

数值：LoCoMo \(J\)=[]、覆盖率=[]；LME-S \(J\)=[]、覆盖率=[]；分层 []。

**表 A1　保留核心与预算利用，主预算 \(b=2048\)。** 空写入分两种成因（无可行 \((g,S)\)、空写为最优）分列报告，均以全部更新单元为分母。

| 数据集 | 平均 \(\|\hat S\|\) | 预算利用率 \(\hat T(\hat S)/b\) | 空写–无可行 (%) | 空写–空写最优 (%) | CovCore (%) |
|---|---:|---:|---:|---:|---:|
| LongMemEval-S | [] | [] | [] | [] | [] |
| LoCoMo | [] | [] | [] | [] | [] |

**图 A3　核心重叠与证据保留。** 三联面板：(a) 两数据集并排的 \(J_i\) 箱线图；(b) 共享核心覆盖率–预算曲线（两数据集两条线）；(c) CovCore 柱状图＋CI。数据：\(J_i\) 分位数 []、覆盖率随预算 []、CovCore []、按问题类型分层 []。

### F.4 主题块备选方案

\(J_i<\tau_J\)=[] 时切块（会话块或 k-means，\(k\)=[]），逐块 ARC，读取跨块检索。备选方案 vs 单块：质量差 []、成本差 []。

### F.5 架构使用分层

按批次长度/跨度/冗余/实体密度/冲突/旧记忆重叠分层 []；跨预算迁移矩阵 []。

**表 A2　架构选择比例与空写入比例（%）。** 前五列在每行非空写入中合计为 100%；最后一列以全部更新单元为分母。

| 数据集 / 预算 | Flat | Chain | Tree | Graph | Cluster | 空写入 |
|---|---:|---:|---:|---:|---:|---:|
| LongMemEval-S / 1024 | [] | [] | [] | [] | [] | [] |
| LongMemEval-S / 2048 | [] | [] | [] | [] | [] | [] |
| LongMemEval-S / 4096 | [] | [] | [] | [] | [] | [] |
| LoCoMo / 1024 | [] | [] | [] | [] | [] | [] |
| LoCoMo / 2048 | [] | [] | [] | [] | [] | [] |
| LoCoMo / 4096 | [] | [] | [] | [] | [] | [] |

### F.6 稳健性与预筛敏感性

噪声 []；冲突 []；标签扰动 []；逐类切换 []；失败案例 []。预筛：显著性 vs 均匀 vs \(n_{\max}=128\) 的 CovCore/质量 []。

### F.7 补充消融

束宽 \(w\in\{\}\)：[]；Top-\(L\in\{\}\)：[]；编码器深度 1/2/4：[]；元数据消融：[]；强制非空（去 STOP）：[]；逐架构 oracle 掩码 vs 保守：[]（与 B.3 互证）。

## 附录 G　成本明细与摊销

### G.1 命题 8 的 N 敏感性与图 A1

\(N\in\{1,5,20,80\}\)：ARC []，ARC-PQ []（实测使用 tokens，B.7）；交叉点 []，与命题 8 的 \(N^*=\lceil C^{\mathrm{wr}}/(C^{\mathrm{wr}}-(\bar C^{\mathrm{rd}}-\bar C^{\mathrm{rd}\,'}))\rceil\)（实测代入）对照。**图 A1**：log-N 双线图，标注交叉点与占优区间。

### G.2 成本总表

| 方法 | Write | Query-time | Calls | P50/P95 | 存储 | GPU | 费用 |
|---|---:|---:|---:|---|---:|---|---|
| ARC | [] | [] | [] | [] | [] | [] | [] |
| 基线… | [] | [] | [] | [] | [] | [] | [] |

### G.3 离线编译开销（对照 RL 训练范式）

Build [] 次、审核 [] 次、wall-time []、训练 GPU []。与 RL 类基线（Memory-R1、Mem-α、MemBuilder）的对照报告四项：rollout/环境交互次数（ARC 为 0）、训练 GPU 时（ARC []，RL 类 []）、奖励模型/合成 QA 构造开销（ARC 无，MemBuilder 类需合成会话级 QA [] 条）、训练稳定性（跨种子方差：ARC []，RL 类 []）。论证：ARC 编译为**一次性、可复用、可并行**的 Build/Audit 记录，训练仅为轻量监督学习（无环境交互、精确 EM 梯度、无采样方差），总离线开销 []（倍率相对 []）。若启用级联筛查（§3.3.5），构造调用降为 []，编译 tokens 降为 []，相对全量编译节省 []%（一致性核验见 E.10）。

### G.4 延迟/存储/费用

E2E P50/P95 延迟 []，非生成式检索延迟 []，存储规模 []，实际费用 []；逐方法拆分见 G.2 总表。

### G.5 缓存

来源 embedding 与构建记录缓存命中率 []，节省的重复构建调用 []；缓存失效与更新协议见 B.6。

## 附录 H　统计与可复现

### H.1 paired bootstrap

按历史聚类，B=[]，95% CI []。

### H.2 多重比较

主表家族 Holm；RQ2/RQ4 分别校正；校正后标记 []。

### H.3 种子

[] 个；汇总规则 []。

### H.4 零容忍

\(\epsilon=0\) 时 Beneficial Switch []；与主配置差 []。

### H.5 可复现清单

代码/提示/schema 发布计划 []（提示与评价模板全文见附录 J）；冻结组件版本表见 B.1；随机种子列表 []；编译档案哈希 []。

**LLM 使用披露（占位，按 ICLR 政策须在正文附录单列 “LLM Usage” 节）**：本文以 LLM 执行三态审核、需求/支持包标注与提示模板生成，所用模型与版本见 B.1，完整提示见附录 J；正式披露文本与 OpenReview 表单按投稿模板填写（待补）。

## 附录 I　完整伪代码

```text
Algorithm 2  离线监督编译（详细）
for 训练单元 i:
  E_i ← C(X_i, M_i^-)                      # B.2
  S_cand ← 支持包引导候选（Φ_i 覆盖、替代组合、增补、E_i、∅；≤K）
  for S in S_cand, g in G with T_g(S)+L_G≤I_G:
      m ← Build_g(S; G)                     # 冻结
      M ← Update(M_i^-, m)
      for q in Q_i: u[q] ← Score(F(q, R(q,M)), y_q*)
      U ← mean(u); Cov ← mean(u≥τ_q); status ← Audit(m,S,g; D_i, {W_rk})   # 三态
      记录 (g,S,m,T_g,C^wr,{C^rd},U,Cov,status)
  for b in B: W_b(i) ← { (g,S): status=PASS ∧ Reach_b=1 }   // 式(16)：PASS 已含双阈值
      // 含规范空写终态 (Flat,∅)：通过全部检查即入档（正文 §3.3 注记）
训练：最小化式(26)（Algorithm 3）
```

```text
Algorithm 3  训练（多正例边缘化）
for batch of (i,b) with W_b(i)≠∅:
  for (g,S) in W_b(i): logP[g,S] ← log p_φ(S|E_i,b) + log p_ψ(g|S,E_i,b)
  LSE ← logsumexp_{W_b(i)} logP                     # 数值稳定
  ρ ← softmax_{W_b(i)} logP
  loss ← -LSE + λ_c · Σ_{W_b(i)} ρ·(C_i-c_b*)/c_b*
  反向传播（logsumexp 梯度即 ρ 加权和，命题 5）
```

```text
Algorithm 4  写入时解码（详细）
active ← {∅}，分数 0；done ← ∅
for t = 0..n:
  cand ← ∅
  for (S_t, score) in active:
    for j in {j>j_t: T̂(S_t{e_j})≤b-m}: cand += (S_t∪{e_j}, score+log π_φ(j))
    done += (S_t, score+log π_φ(STOP))
  active ← top-w(cand)
for (S, sS) in top-L(done):
  for g in {g: T_g(S)≤b}: score(S,g) ← sS + log p_ψ(g|S,E_i,b)   # S=∅ 时规范取 {Flat}（§3.4.2）
(Ŝ,ĝ) ← argmax score
若 Ŝ≠∅: M⁺ ← Update(M^-, Build_ĝ(Ŝ)) 否则 M⁺ ← M^-（规范空写 (Flat,∅)，无生成调用）
```

```text
Algorithm 5  查询时读取
R_q ← Retrieve(q, M⁺)      # 非生成式
return F(q, R_q)           # 唯一生成式调用
```

---

## 附录 J　完整提示与评价模板

全部模板在监督编译开始前冻结，训练、评价与部署全程不变；离线编译与部署共用同一套模板（使用一致性，§3.4.3）。模板中 {} 为运行时填充的槽位；提示语言为英文（数据集语言）。结构合法性审核为确定性 Post 检查（C.2），不使用提示。级联筛查（§3.3.5）不引入新模板：Gate 1 复用 P1（Flat）与 P8/P9 评估管线，Gate 0 为确定性集合与计数检查。

### J.0 提示清单

| 编号 | 名称 | 调用组件 | 用途 | 位置 |
|---|---|---|---|---|
| P1–P5 | Build-{Flat, Chain, Tree, Graph, Cluster} | 冻结构建器 \(G\) | 候选记忆构造（式（4）） | J.1 |
| P6 | Audit-Faithfulness | 审核 LLM | 事实级忠实性三态审核 | J.2 |
| P7 | Audit-Completeness | 审核 LLM | 支持包信息完整性审核 | J.2 |
| P8 | Score-Utility | 回答模型 \(F\)＋judge | 单查询效用 \(u_{i,q}\)（式（11）） | J.3 |
| P9 | Answer | 回答模型 \(F\) | 部署问答（唯一生成式读取调用） | J.4 |
| P10 | Extract-Needs | 标注 LLM | 需求抽取与支持包标注 | J.5 |
| P11 | Judge-QA | LLM judge | LoCoMo J 指标 | J.6 |

### J.1 构建器提示（P1–P5）

公共系统前缀（五架构共用；任务头另注入 C.1 的序列化来源与 schema 常数）：

```text
You are a memory constructor for a long-term memory system. You receive a list
of source units, each with {id, text, session, time, length}, and a target schema.
Rules:
1. Use ONLY information from the given sources. Never add facts, events,
   relations, or timestamps absent from the sources.
2. Every generated item must cite the source ids it derives from.
3. Keep original time expressions; do not normalize or infer new times.
4. Output strictly valid JSON following the schema. No extra text.
```

**P1　Build-Flat。**

```text
Task: extract atomic fact entries from the sources.
Entry: {"id","fact","source_ids","time"}
- One fact per entry; never merge facts across sources.
- Preserve time and provenance for every entry.
Output: {"entries":[...]}
```

**P2　Build-Chain。**

```text
Task: organize the sources into a single chronological/state chain.
- Edges {"from","to","type":"next"} must form ONE simple path.
- Node times non-decreasing along the path.
- Each edge supported by shared sources or direct temporal continuity.
Output: {"nodes":[{"id","content","time","source_ids"}],"edges":[...]}
```

**P3　Build-Tree。**

```text
Task: organize the sources into a single-root hierarchy.
- Exactly one root; every non-root node has exactly one parent; no cycles.
- Leaf source_ids are subsets of the given sources.
- Every summary node must be traceable to the union of its children's sources.
Output: {"root":"id","nodes":[{"id","content","parent","source_ids"}]}
```

**P4　Build-Graph。**

```text
Task: build a sparse typed relation graph over entities/events.
- Every edge must cite an evidence_id from the given sources.
- Node degree <= d_max (given in the task header).
- Relation types must come from the fixed set in the task header.
Output: {"nodes":[{"id","name","type","source_ids"}],
        "edges":[{"from","to","relation","evidence_id"}]}
```

**P5　Build-Cluster。**

```text
Task: partition the entries into disjoint topic clusters.
- Each source belongs to exactly one cluster; clusters do not overlap.
- Each cluster description must be supported by its member sources.
Output: {"clusters":[{"id","topic_desc","member_ids"}]}
```

构建器解码协议随 B.1 冻结、五架构一致（参数 []）；各提示的输出约束与 C.2 的 Post 检查逐条对应。

### J.2 审核提示（P6–P7）

**P6　Audit-Faithfulness。**

```text
You are a strict auditor. For the constructed memory and the source list,
check EVERY atomic claim:
- Is the claim directly supported by the cited sources?
- Are time qualifiers consistent with the sources?
- Are entity relations exactly as stated in the sources?
Per claim: SUPPORTED / VIOLATION (quote the conflicting source).
Memory verdict: PASS if no VIOLATION; FAIL otherwise;
UNKNOWN if the sources are insufficient to decide.
```

**P7　Audit-Completeness。**

```text
Given required information units W = {w_1..w_m} (support packages) and the
constructed memory, check for each w_k whether its information can be read
out from the memory (combining multiple entries is allowed).
Per unit: COVERED / MISSING (state what is absent).
Memory verdict: PASS if all COVERED; FAIL if any MISSING;
UNKNOWN if undecidable.
```

三态裁决与 PASS/FAIL/UNKNOWN 的合成规则见式（14）与 C.4；审核 LLM 及其与人工判定的一致率报告于 D.3。P7 按“信息可读出”判定（允许跨条目组合），与 Gate 0 的 id 级 \(\Phi_i\) 检查在**支持包标注封闭**假定下等价；标注非封闭时的差异计数见 E.10。

### J.3 效用评分提示（P8，对应式（11））

```text
Question: {q}
Reference answer: {y*}
Candidate answer (produced by the frozen answer model using ONLY this
candidate memory): {y_hat}
Rate utility on [0,1]:
- 1.0: semantically equivalent to the reference on all required points
- partial credit: list the missing points
- 0.0: wrong, empty, or unjustified refusal
Output JSON: {"score": s, "missing": [...]}
```

\(u_{i,q}\) 取该分值；\(\tau_q\) 为“有效服务”判定线（语义见 D.1）。评分 judge 冻结（模型 []）。

### J.4 回答提示（P9，部署读取）

```text
System: Answer the question using ONLY the provided memory context.
If the context is insufficient, state that you cannot answer.
Do not use outside knowledge.
User: Context: {R_q}
Question: {q}
```

该提示对应查询时的唯一生成式调用（§3.6 注记）；离线编译的效用评价使用同一模板，保证训练与部署路径相同。

### J.5 需求抽取与支持包提示（P10，A.5 骨架的完整版）

```text
[Needs] Given history H and QA pair (q, y*), list the minimal information
needs d_1..d_R required to answer q.
[Support] For each need d_r, give ALL minimal source groups from H that
support it (each group alone suffices; groups are alternatives).
[Constraints] Use only information inside H. If H is insufficient, output
MISSING and state what is absent.
Output JSON: {"needs":[{"d":"...","supports":[[e_ids],[e_ids]]}],
              "missing":"..."}
```

替代包歧义的人工裁定、MISSING 判定与抽查一致率报告于 A.5。该提示仅用于离线标注与诊断，不进入写入路径。

### J.6 LLM-Judge 提示（P11，LoCoMo J 指标）

```text
Question: {q}
Ground truth: {y*}
Prediction: {y_hat}
Judge whether the prediction matches the ground truth semantically.
Ignore phrasing differences; numbers, dates, and names must match exactly.
Output JSON: {"judge":"correct"|"incorrect","reason":"..."}
```

与 LoCoMo 官方评价实现及 Mem0 的 judge 报告惯例一致（§4.1.1）；LongMemEval-S 主指标使用官方 GPT-4o correctness judge，不作修改。
