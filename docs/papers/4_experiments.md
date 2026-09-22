# 4 实验

实验围绕四个递进问题展开：端到端质量–生命周期成本表现（RQ1）、写入阶段自适应架构相对固定架构的必要性（RQ2）、多正例结构化监督与内容优先因子化向未见记忆更新单元的泛化（RQ3），以及 ARC 写入行为的诊断与收益（RQ4），包括内容优先（S→g）设计所依据的核心证据重叠诊断、保留核心的预算利用与架构反事实。

四个 RQ 分别对应四个主张；每个 RQ 在正文保留一个主图或主表，其余图表移入附录：**C1（质量–成本）** 写入阶段、查询分布感知的联合构造改善质量–摊销生命周期成本前沿，且优势随查询复用次数增大（RQ1；主表 2，附录 G 图 A1）；**C2（自适应结构）** 按写入内容选择架构优于按数据集固定的单一架构（RQ2；主图 3）；**C3（监督机制）** 多正例已验证集、覆盖率约束、内容条件架构与成本偏好各自贡献泛化（RQ3；主表 3，附录 E.7 图 A2）；**C4（行为与依据）** 核心证据重叠假设成立、保留核心覆盖未来证据、架构切换带来 Pareto 收益（RQ4；主表 4，附录 F.3 表 A1 与图 A3、F.5 表 A2）。

## 4.1 实验设置

### 4.1.1 数据、更新单元与信息隔离

主实验采用 LongMemEval-S 与 LoCoMo，二者是 Mem0[^mem0]、Zep[^zep]、A-MEM[^amem]、MESA[^mesa] 等基线论文最常采用的数据集，沿用相同基准与指标定义；各方法数值均在**统一重实现协议**下本地报告，与原文报告值的一致性对照见附录 E.1。指标口径：LongMemEval-S 以官方 GPT-4o judge 的 QA Accuracy 为主指标（与 Zep、Mem0 一致）；LoCoMo 以官方实现的 token-level QA F1 与 LLM-judge 准确率（记 J）为双主指标（F1 与 Mem0、A-MEM 一致，J 与 Mem0 的 judge 报告一致），并按统一协议汇总类别 1–4，类别 5 的拒答结果单独报告。[^longmemeval][^locomo][^locomo-code] BLEU-1 与两个数据集的 retrieval Recall/NDCG 作为补充指标列于附录 A/E；tokens/query 与延迟作为成本与效率指标报告（与 Mem0 的 token 消耗报告、Zep 的延迟报告一致）。两个数据集合起来覆盖核心证据重叠程度的两端：LoCoMo 的会话证据高度集中、重叠较高，LongMemEval-S 的多主题历史重叠较低，分别检验观察一的适用域；重叠诊断协议与分层结论见附录 A.3 与 §4.5.1。

训练、开发和测试按完整历史分组，来源重叠、同一事件及其改写版本归入同组。每个训练历史进一步形成若干**记忆更新单元**：单元 \(i\) 包含写入前记忆 \(\mathcal M_i^{-}\)、当前待写入历史批次 \(X_i\)，以及仅用于离线监督评价的未来查询集合 \(\mathcal Q_i\)。测试历史和测试问题不参与训练监督构造、候选终态评价、超参数选择、Best Fixed 选择或 checkpoint 选择。

未来查询在 ARC 中只承担离线监督与评价的角色：训练候选 \((S,g)\) 构造完成后，\(\mathcal Q_i\) 用于估计该持久记忆对多数未来查询的长期价值（平均效用与覆盖率）；部署写入时 ARC 看不到任何未来查询。更新单元的划分、未来查询与支持证据的对应规则、类别分布和数据清洗细节见附录 A。跨历史划分稳定性、LongMemEval-M 与更长历史实验见附录 F。

### 4.1.2 受控写入与读取条件

ARC 与 RQ2–RQ4 的受控对照共享同一个 query-independent 候选生成器 \(\mathcal C\)。对每个写入单元，候选集合由当前历史批次 \(X_i\) 与写入前记忆 \(\mathcal M_i^{-}\) 中的有限关联邻域组成，最多保留 64 个来源（超长历史按附录 B.2 的 query-free 显著性预筛取足上限）。若需要从已有记忆取得更新邻域，主配置沿用 BM25 与冻结 Qwen3-Embedding-0.6B 的 RRF 融合，但检索键由当前写入批次产生，而不是未来测试查询。候选去重、邻域大小、缓存与索引更新协议见附录 B。

构建输入预算设为

\[
\mathcal B=\{1024,2048,4096\}\ \text{tokens}.
\]

所有受控比较使用相同的冻结构建模型 \(G\)、持久化更新协议、最终回答模型 \(F\)、五类 schema 与评价器。离线主配置每个更新单元至多保留 \(K=16\) 个来源集合，再进行架构展开；各训练变体共享同一批实际 Build/Audit 记录。效用阈值 \(\tau_U\)、单查询阈值 \(\tau_q\) 与覆盖率阈值 \(\tau_{\mathrm{cov}}\) 在开发集冻结，裕量 \(m\) 按附录 B.3 的实测边界常数设定（具体数值见附录 D.1/B.3）。

当持久记忆构造完成后，query-time 只执行固定读取器 \(\mathcal R(q,\mathcal M)\) 与回答模型，不再运行 ARC policy，也不重新选择 \(g\)、\(S\) 或重新 Build。为隔离写入策略的影响，RQ2–RQ4 使用相同的 query-time retriever、top-\(k\)/token 上限和回答后端。模型版本、硬件、训练规模、完整超参数与解码配置见附录 B；构建 schema 见附录 C；离线未来查询评价与审核一致性见附录 D。

### 4.1.3 质量、生命周期成本与统计协议

持久记忆系统包含一次或多次写入，以及随后对同一记忆的多次查询。正文主表因此使用**摊销生命周期 tokens/query**作为统一成本指标。设测试集上的实际写入调用 token 总量为 \(\mathcal T_{\mathrm{wr}}\)，全部测试查询从读取到最终回答的生成 token 总量为 \(\mathcal T_{\mathrm{rd}}\)，测试查询数为 \(N_Q\)，定义

\[
\mathcal T_{\mathrm{life}}
=
\frac{
\mathcal T_{\mathrm{wr}}+\mathcal T_{\mathrm{rd}}
}{N_Q}.
\tag{30}
\]

该定义把一次性持久化构造成本按实际服务的测试查询数摊销，避免把 write-time Build 错计为每个查询都重复发生。对于 Full Context 或无需生成式写入的基线，其写入生成成本按实际执行路径计为零；对于具有复杂持久化维护过程的方法，所有实际写入调用均计入 \(\mathcal T_{\mathrm{wr}}\)。

附录 G 分别报告 Write Tokens、Query-time Tokens/q、Online LLM Calls、E2E P50/P95、存储规模、GPU 时间、实际费用、离线监督搜索与训练成本，并给出不同查询复用次数下的生命周期敏感性分析。检索 Recall/NDCG 属于机制诊断，不作为端到端主表指标。

所有成本由本地重新运行日志计算；失败查询仍保留在质量与平均成本分母中。学习方法使用多随机种子，主要比较按完整历史聚类执行 paired bootstrap，报告 95% 置信区间；多重比较校正与跨种子汇总规则见附录 H。

## 4.2 RQ1：ARC 能否改善持久长期记忆的质量–生命周期成本权衡？

RQ1 比较完整系统在测试历史构造完成后的最终问答质量与摊销生命周期成本。Full Context 和 RAG-RRF 提供无持久化压缩与统一检索参照；Mem0 提供通用长期记忆基线；LightMem 提供效率导向基线。[^mem0][^lightmem] 与 ARC 机制直接相邻的主表方法包括 MemGAS、MAGMA、LeanMem、FluxMem、BudgetMem 与 MESA。[^memgas][^magma][^leanmem][^fluxmem][^budgetmem][^mesa]

这些方法保留各自原生的记忆生命周期：具有预构建持久记忆的方法先完成其写入/维护，再处理测试查询；具有查询时路由或多结构选择的方法保留其原有查询阶段操作。最终回答模型、数据划分和评价协议尽可能统一；方法必需的专用组件保持原有功能。扩展比较列于附录 E，其中重点包含以在线强化学习训练写入策略的 Memory-R1、Mem-α 与 MemBuilder，用于检验监督编译与在线 RL 两种训练范式在质量、训练开销与稳定性上的取舍（训练开销对照见附录 G.3）；A-MEM、Chain-of-Memory、MemChain、Mnemis、StructMem、LiCoMemory 与 SimpleMem 一并列入。[^amem][^com][^memoryr1][^memalpha][^membuilder][^memchain][^mnemis][^structmem][^licomemory][^simplemem]

正文只比较 ARC 的预先指定主写入预算 \(b=2048\)；\(b=1024\) 与 \(4096\) 的完整结果、写入/读取成本拆分以及生命周期复用敏感性见附录 G。RQ2 在三个预算上进一步比较 Adaptive 与固定架构。为量化“写入一次、多次读取”相对“按查询构造”的代价，附录 E 另报告 query-time 构造变体 ARC-PQ（每个测试查询按其自身条件联合解码并构造一次）作为上界消融。

**表 2　长期记忆系统的端到端质量与摊销生命周期成本。**

| 方法 | LongMemEval-S Acc. (%) ↑ | Tokens/q ↓ | LoCoMo F1 (%) ↑ | LoCoMo J (%) ↑ | Tokens/q ↓ |
|---|---:|---:|---:|---:|---:|
| Full Context | [] | [] | [] | [] | [] |
| RAG-RRF | [] | [] | [] | [] | [] |
| Mem0 | [] | [] | [] | [] | [] |
| LightMem | [] | [] | [] | [] | [] |
| MemGAS | [] | [] | [] | [] | [] |
| MAGMA | [] | [] | [] | [] | [] |
| LeanMem | [] | [] | [] | [] | [] |
| FluxMem | [] | [] | [] | [] | [] |
| BudgetMem | [] | [] | [] | [] | [] |
| MESA | [] | [] | [] | [] | [] |
| **ARC (\(b=2048\))** | **[]** | **[]** | **[]** | **[]** | **[]** |

主表回答“达到给定端到端质量时，一份长期记忆从构造到多次读取需要多少摊销生成开销”。LongMemEval-S 的 retrieval Recall/NDCG，LoCoMo 的 BLEU-1 与类别级 F1/J 拆分，以及写入成本、纯查询时成本、调用次数、延迟与费用均列于附录 E/G。两数据集每查询摊销分母的 \(|\mathcal Q_i|\) 规模不同（约 [] 与约 []，见 A.4），Tokens/q 列跨数据集不直接互比；\(N\in\{1,5,20,80\}\) 的复用敏感性与 ARC-PQ 对照见 G.1/E.3。

## 4.3 RQ2：写入阶段的自适应构造是否优于固定架构？

RQ2 将 ARC Adaptive 与固定 Flat、Chain、Tree、Graph、Cluster 比较。每个固定版本在所有非空**记忆更新单元**上锁定同一种架构，并在该架构对应的成功终态上训练相同容量的内容优先来源策略；Adaptive 允许策略根据当前写入候选 \(E_i\)、已选内容 \(S\) 和预算选择架构。各版本共享候选生成器、训练更新单元、离线 Build/Audit 记录、成本目标、构建器、读取器、回答模型和预算。

对每个数据集和预算，开发集端到端质量最高的固定架构定义为 \(g_{\mathrm{fixed}}^*(d,b)\)；质量并列时选择摊销生命周期成本更低者，并在测试前冻结。该固定架构对该数据集和预算下的全部测试写入单元保持不变。固定架构缺少成功监督的更新单元–预算对按与 ARC 相同的覆盖规则处理，监督覆盖率与匹配覆盖对照见附录 E。

**图 3　ARC Adaptive 与五种固定架构的质量–生命周期成本轨迹。** 两个面板分别对应 LongMemEval-S 和 LoCoMo；横轴为 Lifecycle Tokens/q，纵轴为任务质量。每种方法在 \(b\in\{1024,2048,4096\}\) 下形成三个操作点，并给出置信区间。图 3 数据先以表格呈现（方法 × 预算：质量 []、摊销成本 []，见附录 E.5），再绘制为轨迹图并标出质量–成本前沿；合适图型为带预算标注的点–线轨迹图（每方法一条三操作点折线），前沿方法加粗。

该比较检验的是：面对不同历史内容、冗余程度、时间结构与更新状态时，允许写入策略改变组织方式，是否比对所有更新单元固定一种架构更有效。Adaptive 相对 Best Fixed 的质量差、成本差和质量–成本前沿结果均为 []；每预算配对差、写入批次属性分层与显著性检验见附录 E/H。

## 4.4 RQ3：多正例与内容优先监督是否泛化为有效的写入决策？

RQ3 检验从离线成功档案到未见写入单元的三个环节：成功终态的概率质量、单次写入决策的真实成功率，以及成功条件下的成本偏好。先冻结模型、超参数和 checkpoint，再在未参与训练或模型选择的诊断更新集合 \(\mathcal I_{\mathrm{diag}}\) 上重新执行 Build/Audit；这里的候选生成协议独立于待评策略，由此建立统一诊断档案 \(\mathcal W_b^{\mathrm{diag}}(i)\)。诊断未来查询只用于评价，不输入待评策略。

### 4.4.1 三项互补指标

**Success-set NLL。** 在具有非空诊断成功档案的更新单元集合 \(\mathcal I_b^+\) 上计算

\[
\operatorname{SS\!-\!NLL}_b
=
-\frac1{|\mathcal I_b^+|}
\sum_{i\in\mathcal I_b^+}
\log
\left[
\sum_{(g,S)\in\mathcal W_b^{\mathrm{diag}}(i)}
p_\phi(S\mid E_i,b)\,
p_\psi(g\mid S,E_i,b)
\right].
\tag{31}
\]

概率使用完整写入动作空间归一化，包含来源 Pick/STOP 动作与架构动作。诊断档案规模有限，其覆盖率与各项指标一并报告。

**PASS@1。** 对每个诊断更新单元按正式 write-time 协议解码一个内容束并对每个完成的 \(S\) 评分全部可行架构，取联合概率最高的 \((\hat S_i,\hat g_i)\)，真实构造一次持久记忆，再使用该单元的未来查询集合和 §3.3 的平均效用＋覆盖率＋三态审核判断是否成功：

\[
\operatorname{PASS@1}_b
=
\frac1{|\mathcal I_{\mathrm{diag}}|}
\sum_{i\in\mathcal I_{\mathrm{diag}}}
\mathbf 1
[
\operatorname{status}_i(\hat S_i,\hat g_i)=\mathrm{PASS}
].
\tag{32}
\]

@1 表示一次完整写入策略输出；束内其他候选不逐一构造，也不从多个已构造方案中择优。UNKNOWN 不计为成功并保留在分母中。

**认证下界核验（推论 1）。** 对每个诊断更新单元，记录 argmax 部署方案的实现平均效用 \(\bar U_i\)，核验 \(\bar U_i\ge M_\theta(i,b)\cdot\tau_U\) 的满足率与平均裕度；采样解码版按同一协议核验（理论上恒成立，兼作实现正确性检查）。结果列于附录 E.5。

**成功条件成本差。** 记 \(c_{b,\mathrm{diag}}^*(i)\) 为诊断档案内最低已知成功生命周期成本。不同变体的成功样本难度不同，直接比较成本差会有偏差；因此只在“有诊断正例、且所有主表变体都 PASS”的共同更新单元集合 \(\mathcal I_b^\cap\) 上比较

\[
\Delta C_{\mathrm{succ}}^{(v)}
=
\frac1{|\mathcal I_b^\cap|}
\sum_{i\in\mathcal I_b^\cap}
\frac{
C_i(\hat S_v,\hat g_v)-c_{b,\mathrm{diag}}^*(i)
}{
c_{b,\mathrm{diag}}^*(i)
}.
\tag{33}
\]

该指标与训练中的成本定义一致；写入成本和读取成本拆分结果见附录 E/G。

### 4.4.2 对照与机制判定

主表保留五个直接消融。**Single-positive** 为每个更新单元、每个预算选择最低成本的一个已验证成功终态，并以单终态似然训练；**g→S factorization** 将联合概率反向因子化为 \(p(g\mid E_i,b)p(S\mid E_i,b,g)\)，网络容量与 pair-level 监督不变，检验内容优先顺序的作用；**g⊥S** 令架构头不条件于已选内容，即 \(p_\psi(g\mid E_i,b)\)，检验“\(g\) 是对 \(S\) 的组织决策”这一条件化的作用；**w/o Coverage** 的成功档案只使用平均效用阈值 \(U_i\ge\tau_U\) 与审核，去掉覆盖率约束，检验覆盖率对“对多数查询有效”的贡献；**w/o Cost-aware** 设置 \(\lambda_c=0\)，保留多正例边缘目标。其余网络、监督档案、训练步数与解码配置一致。

**表 3　结构化监督向写入策略的泛化，主预算 \(b=2048\)。**

| 数据集 | 变体 | Success-set NLL ↓ | PASS@1 (%) ↑ | 成功条件成本差 (%) ↓ |
|---|---|---:|---:|---:|
| LongMemEval-S | Single-positive | [] | [] | [] |
| LongMemEval-S | g→S factorization | [] | [] | [] |
| LongMemEval-S | g⊥S | [] | [] | [] |
| LongMemEval-S | w/o Coverage | [] | [] | [] |
| LongMemEval-S | w/o Cost-aware | [] | [] | [] |
| LongMemEval-S | **ARC** | **[]** | **[]** | **[]** |
| LoCoMo | Single-positive | [] | [] | [] |
| LoCoMo | g→S factorization | [] | [] | [] |
| LoCoMo | g⊥S | [] | [] | [] |
| LoCoMo | w/o Coverage | [] | [] | [] |
| LoCoMo | w/o Cost-aware | [] | [] | [] |
| LoCoMo | **ARC** | **[]** | **[]** | **[]** |

预期模式：Single-positive 的 SS-NLL 与 PASS@1 应差于 ARC（丢弃多正例信息）；g→S 在重叠高的 LoCoMo 上损失较小、在重叠较低的 LongMemEval-S 上损失较大；g⊥S 的 PASS@1 与成本差变差说明内容条件化的架构选择有效；w/o Coverage 的 PASS@1（按覆盖率度量计）下降；w/o Cost-aware 的成功条件成本差升高而质量基本不变。诊断成功档案覆盖率、共同成功子集大小和跨架构多正例比例均为 []。

机制消融结果的分面点区间可视化（变体 × {SS-NLL, PASS@1, 成功条件成本差}）以图 A2 列于附录 E.7。其他预算、未见预算、正例数量、候选规模、束宽、效用与覆盖率阈值、已验证集覆盖以及代表性查询集规模 \(|\mathcal Q_i|\) 的敏感性见附录 E。

§3.7 的策略刷新不单列新的 RQ。附录 E 以两种设置检验增量能力：其一分阶段加入新的记忆更新单元，其二在固定评价查询和执行协议下为既有更新单元加入新发现的成功终态；比较刷新前后的 Success-set NLL、PASS@1 与刷新训练开销，结果均为 []。若新增未来评价查询，则对受影响档案重新审核，而非直接做正例并集。

## 4.5 RQ4：ARC 学到了怎样的写入行为，这些选择是否有用？

### 4.5.1 核心证据重叠诊断：内容优先的设计依据

内容优先（S→g）因子化基于一个可检验的假设：同一更新单元内不同未来查询的高价值证据高度重叠，因此存在一个对多数查询都有价值的核心 \(S\)。该诊断只使用离线标注，不参与训练。对每个更新单元，取每个查询的最小支持包作为其必要证据集 \(V_q\)（§3.3.1），报告两项统计：其一为查询间重叠

\[
J_i
=
\frac{2}{|\mathcal Q_i|(|\mathcal Q_i|-1)}
\sum_{q\ne q'}
\frac{|V_q\cap V_{q'}|}{|V_q\cup V_{q'}|},
\tag{34}
\]

其二为共享核心覆盖率：在预算可行的单一来源集合 \(S\) 下，满足 \(V_q\subseteq S\) 的查询比例的最大值。\(J_i\) 高且共享核心覆盖率高，说明单一持久方案即可服务多数查询，支持 S-first；若重叠低，则应改为按主题块划分更新单元后再应用 ARC（附录 F 给出该备选方案的对照设置与结果）。

诊断结果：LoCoMo 的 \(J_i\) 与共享核心覆盖率高于 LongMemEval-S（LoCoMo：\(J\)=[]，共享核心覆盖率=[]；LongMemEval-S：\(J\)=[]，共享核心覆盖率=[]；组间差异的配对检验与 95% CI 按 §4.1.3 协议在附录 H.1 报告，[]），即 LoCoMo 的多数查询反复依赖同一批核心会话证据，直接支持内容优先设计；LongMemEval-S 的重叠程度按问题类型分层见附录 F。

### 4.5.2 保留核心、预算利用与证据保留率

报告写入策略实际保留的内容及其对测试查询的证据保留情况。记测试更新单元的写入结果为 \(\hat S_i\)，定义预算利用率为 \(\hat T(\hat S_i)/b\)；并定义**留出证据保留率**：对仅用于诊断的测试查询支持集 \(V_q\)，

\[
\operatorname{CovCore}
=
\frac1{|\mathcal Q_{\mathrm{test}}|}
\sum_{q\in\mathcal Q_{\mathrm{test}}}
\mathbf 1\!\left[V_q\subseteq \hat S_i\right].
\tag{35}
\]

CovCore 度量“写入时保留的核心是否覆盖未来查询真正需要的证据”，是连接重叠假设与端到端质量的桥梁。注意 \(V_q\) 在测试期仅用于诊断，不进入写入路径。

保留核心与预算利用的逐数据集统计（平均 \(|\hat S|\)、预算利用率 \(\hat T(\hat S)/b\)、空写入比例与 CovCore）以表 A1 列于附录 F.3；核心重叠与证据保留的三联可视化（\(J_i\) 分布、共享核心覆盖率–预算曲线、CovCore）以图 A3 列于附录 F.3。预期 LoCoMo 因重叠高而 CovCore 高、平均 \(|\hat S|\) 更集中；LongMemEval-S 若 CovCore 较低，则其质量上限受证据保留约束，相关分层与主题块备选方案见附录 F。

### 4.5.3 不同写入状态与预算下的架构使用

统计测试历史中各记忆更新单元在不同预算下的架构选择。空来源集合单独报告，五种架构的使用比例在非空写入中计算；各数据集、各预算下的架构选择比例与空写入比例以表 A2 列于附录 F.5。

架构分布用于分析策略是否随**写入内容**发生系统性变化，而不是随未来查询临时切换。分析重点包括更新批次长度、时间跨度、信息冗余、实体密度、新旧事实冲突以及与已有记忆的重叠程度；同时考察相同写入单元在不同预算下的结构迁移。完整分层结果见附录 F。

### 4.5.4 固定来源的写入架构反事实

为隔离“结构选择”本身的作用，对 ARC 选择非空 \(\hat S_i\) 且 \(\hat g_i\ne g_{\mathrm{fixed}}^*(d,b)\) 的诊断更新单元固定来源集合 \(\hat S_i\)，分别使用 ARC 所选架构与开发集确定的 Best Fixed 构造持久记忆：

\[
\mathcal M_i^{\mathrm{ARC}}
=
\operatorname{Update}
\left(
\mathcal M_i^{-},
\operatorname{Build}_{\hat g_i}(\hat S_i)
\right),
\qquad
\mathcal M_i^{\mathrm{CF}}
=
\operatorname{Update}
\left(
\mathcal M_i^{-},
\operatorname{Build}_{g_{\mathrm{fixed}}^*(d,b)}(\hat S_i)
\right).
\tag{36}
\]

两分支共享来源、构建模型、解码配置、写入前记忆、query-time retriever、回答模型和诊断未来查询集合。只有两种架构的序列化均满足同一写入预算时才进入预算内配对统计；反事实超预算的更新单元单独报告不可行比例，不通过删减来源重新适配。

定义该更新单元上的总体未来效用差与生命周期成本差为 \(\Delta U_i=U_i^{\mathrm{ARC}}-U_i^{\mathrm{CF}}\)，\(\Delta C_i=C_i^{\mathrm{ARC}}-C_i^{\mathrm{CF}}\)。一次结构切换满足容忍阈值下的 Pareto 改善，当且仅当

\[
\operatorname{Benefit}(i)
=
\mathbf 1
\left[
(\Delta U_i>\epsilon_U\land\Delta C_i\le\epsilon_C)
\ \lor\
(\Delta U_i\ge-\epsilon_U\land\Delta C_i<-\epsilon_C)
\right].
\tag{37}
\]

\(\epsilon_U,\epsilon_C\) 在测试前冻结，零容忍敏感性见附录 H。

**表 4（RQ4 主表）　固定来源条件下的写入架构切换收益，主预算 \(b=2048\)。** Switch Rate 以非空写入单元为分母；Paired Feasible 以发生架构切换的更新单元为分母；Beneficial Switch 及平均差值在同一预算内可行的配对样本集上计算。

| 数据集 | Best Fixed | Switch Rate (%) | Paired Feasible (%) | Beneficial Switch (%) ↑ | \(\Delta\) Quality（百分点）↑ | \(\Delta\) Lifecycle Tokens ↓ |
|---|---|---:|---:|---:|---:|---:|
| LongMemEval-S | [] | [] | [] | [] | [] | [] |
| LoCoMo | [] | [] | [] | [] | [] | [] |

该反事实固定 \(S\)，因此差异主要来自架构专属构造及其对后续检索和回答的影响。联合报告切换频率、预算可行比例、收益比例与平均质量/成本差，可以区分“经常换结构但收益有限”和“结构改变确实改善持久记忆价值”两种情况。其余预算、写入状态分层、检索噪声、冲突事实、标签扰动和典型失败案例见附录 E/F/G；结果均为 []。

---

## 4.6 编译效率：级联筛查（对应附录 E.10、G.3）

级联筛查（§3.3.5）不改变入档规则，其评估只需回答两个问题：省多少，漏多少。

**协议。** 主结果（表 2–4）一律使用全量编译，档案定义与全部命题不受级联影响；级联作为成本等价性评估做事后复判：利用全量编译日志中已记录的逐方案审核结论、逐查询效用与真实 token 开销，按算法 C-2 离线复判三级关卡，统计其本可跳过的构造与评估调用。\(m_0\) 与 \(\Delta\) 在开发集冻结（B.4），测试历史零参与。级联的唯一随机来源是 \(\mathcal Q^0_i\) 的抽样，其种子冻结，故离线复判结果唯一确定；级联的任何关卡均不使用测试集信息。

**指标。** (i) 调用节省：真实构造调用与效用评估调用的减少量，并折算编译 tokens（口径同 G.3）；(ii) 档案一致性：级联档案与全量档案的 Jaccard 重合度，逐预算报告；(iii) 下游不变性：以级联档案重训策略后的 SS-NLL（式（31））与 PASS@1（式（32））相对全量档案训练的差值；(iv) 经验架构收益 \(\gamma(S)=\max_g U_i(g,S)-U_i(\mathrm{Flat},S)\) 的分布（命题 9 注记的诊断量）。

**表 5　级联筛查相对全量编译：成本节省与档案一致性。** 调用节省以相对全量编译的百分比报告；档案一致性为逐预算 Jaccard 的均值；下游差值为级联训练减去全量训练。

| 数据集 | 构造调用节省 (%) ↓ | 评估调用节省 (%) ↓ | 编译 tokens 节省 (%) ↓ | 档案一致性 ↑ | SS-NLL 差 | PASS@1 差 |
|---|---:|---:|---:|---:|---:|---:|
| LongMemEval-S | [] | [] | [] | [] | [] | [] |
| LoCoMo | [] | [] | [] | [] | [] | [] |

**判读规则（预注册）。** 若经验 \(\gamma\) 的 0.9 分位低于 \(\Delta\)（\(\Delta\) 即按该规则在开发集冻结），档案一致性应接近 1；一致性每下降 1 个百分点，须同时报告对应的下游指标漂移，以排除“省了开销但漏掉关键方案”的情形。

---

## 本节参考文献

[^longmemeval]: Wu et al. *LongMemEval: Benchmarking Chat Assistants with Long-Term Interactive Memory*. arXiv:2410.10813；官方数据与评价实现：xiaowu0162/LongMemEval。

[^locomo]: Maharana et al. *Evaluating Very Long-Term Conversational Memory of LLM Agents*. ACL 2024. DOI: 10.18653/v1/2024.acl-long.747.

[^locomo-code]: LoCoMo 官方实现：snap-research/locomo，`task_eval/evaluation.py`。

[^mem0]: Chhikara et al. *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory*. arXiv:2504.19413, 2025.

[^zep]: Rasmussen et al. *Zep: A Temporal Knowledge Graph Architecture for Agent Memory*. arXiv:2501.13956, 2025.

[^amem]: Xu et al. *A-MEM: Agentic Memory for LLM Agents*. arXiv:2502.12110, 2025.

[^lightmem]: Fang et al. *LightMem: Lightweight and Efficient Memory-Augmented Generation*. arXiv:2510.18866v4, 2026.

[^com]: *Chain-of-Memory: Lightweight Memory Construction with Dynamic Evolution for LLM Agents*. arXiv:2601.14287v2, 2026.

[^memgas]: Xu et al. *From Single to Multi-Granularity: Toward Long-Term Memory Association and Selection of Conversational Agents*. ICLR 2026.

[^magma]: Jiang et al. *MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents*. ACL 2026. DOI: 10.18653/v1/2026.acl-long.1709.

[^leanmem]: Liao et al. *LeanMem: Simple and Efficient Long-Term Memory for LLM Agents*. arXiv:2608.03463, 2026.

[^fluxmem]: Lu et al. *Choosing How to Remember: Adaptive Memory Structures for LLM Agents*. arXiv:2602.14038, 2026.

[^budgetmem]: Zhang et al. *Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory*. arXiv:2602.06025, 2026.

[^mesa]: Zhao et al. *MESA: Task-Adaptive Multi-Structure Evidence Selection for Long-Horizon Agent Memory*. arXiv:2608.10108, 2026.

[^memoryr1]: Yan et al. *Memory-R1: Enhancing Large Language Agents to Manage and Utilize Memories via Reinforcement Learning*. ACL 2026. DOI: 10.18653/v1/2026.acl-long.583.

[^memalpha]: Wang et al. *Mem-α: Learning Memory Construction via Reinforcement Learning*. arXiv:2509.25911, 2025.

[^membuilder]: Shen et al. *MemBuilder: Reinforcing LLMs for Long-Term Memory Construction via Attributed Dense Rewards*. ACL 2026. DOI: 10.18653/v1/2026.acl-long.1284.

[^memchain]: Ma et al. *MemChain: Learning Interpretable Memory Traces for Memory-Augmented LLM Agents*. arXiv:2607.24097, 2026.

[^mnemis]: *Mnemis: Dual-Route Retrieval on Hierarchical Graphs for Long-Term LLM Memory*. arXiv:2602.15313v2, 2026.

[^structmem]: *StructMem: Structured Memory for Long-Horizon Behavior in Long-Horizon LLMs*. arXiv:2604.21748, 2026.

[^licomemory]: *LiCoMemory: Lightweight and Cognitive Agentic Memory for Efficient Long-Term Reasoning*. arXiv:2511.01448, 2025.

[^simplemem]: Liu et al. *SimpleMem: Efficient Lifelong Memory for LLM Agents*. arXiv:2601.02553, 2026.
