# 2 相关工作

## 2.1 长期智能体记忆

RAG 奠定了外部存储、检索与生成的基本范式 \citep{lewis2020retrieval}。长期智能体记忆进一步将外部状态扩展为持续演化的交互经验：Generative Agents 使用记忆流、反思和规划，MemoryBank 采用遗忘/强化机制，MemGPT 通过虚拟上下文管理缓解窗口限制 \citep{park2023generative,zhong2024memorybank,packer2023memgpt}。LoCoMo 和 LongMemEval 分别从超长多会话对话以及抽取、跨会话/时间推理、更新和拒答等能力评估此类系统 \citep{maharana2024locomo,wu2025longmemeval}。

近年的写入机制覆盖显式增删改操作和可演化笔记 \citep{chhikara2025mem0,xu2025amem}、时态图与层级存储 \citep{rasmussen2025zep,kang2025memoryos}，以及轻量过滤、压缩和记忆准入 \citep{fang2026lightmem,liu2026simplemem,nan2025nemori,zhang2026amac}。其他工作从异质 profile/event/record 表示、类型化单元、主题文档、事实双轨、片段整合或生命周期关系丰富持久化内容 \citep{liao2026leanmem,qiu2026dimmem,ji2026infinimemory,luo2026memsif,li2026lycheememory,yacoubi2026memorylace,latimer2025hindsight}；EdgeMem、REALM 和 RPMem 则分别强调证据保留、检索驱动重整和参数化跨会话状态 \citep{cui2026edgemem,song2026realm,zhao2026rpmem}。

## 2.2 结构选择、决策时点与预算

异质表示不等同于写入时结构选择。MemGAS、MAGMA、CompassMem、RippleMem 和 HERO 虽构建多粒度或多图记忆，但主要在**查询时**完成粒度选择、图遍历或关联扩展 \citep{xu2026memgas,jiang2026magma,hu2026compassmem,ji2026ripplemem,lin2026hero}。FluxMem 在写入时基于交互特征选择单一结构，LeanMem 在写入时进行异质分类、在查询时再组合；相对地，MESA、BudgetMem、MemChain、LazyMem 和 MEMO 均在已知查询后选择视图、计算档位或工作记忆 \citep{lu2026fluxmem,liao2026leanmem,zhao2026mesa,zhang2026budgetmem,ma2026memchain,yu2026lazymem,gao2026memo}。

成本约束也有不同对象：LightMem、SimpleMem 与 MemRefine 主要控制构造、维护或存储成本 \citep{fang2026lightmem,liu2026simplemem,kim2026memrefine}；EMBER 直接在预查询阶段、固定来源证据预算内学习保留 evidence capsules \citep{li2026ember}；BudgetMem 和 MEMO 则约束查询时计算或回答上下文 \citep{zhang2026budgetmem,gao2026memo}。因此，ARC 不宣称首个预算化写入方法；其特定问题是对来源子集与架构 \((S,g)\) 作联合决策，并以构建模型**真实序列化输入**的 token 数作为硬可行性条件。

## 2.3 学习信号与 ARC 定位

学习型方法已从启发式控制转向结果驱动的记忆管理：Memory-R1、Mem-\(\alpha\)、MemBuilder、AgeMem 和 MEM1 分别以结果奖励、会话级稠密反馈或联合记忆—推理学习优化记忆操作 \citep{yan2026memoryr1,wang2025memalpha,shen2026membuilder,yu2026agemem,zhou2026mem1}。Memory-R2、CMI-Mem、CHIME、MemCalib 与 Grounding Agent Memory 又分别从局部重 rollout、内在信息奖励、操作归因、反事实校准和环境探测改善信用分配或记忆可靠性 \citep{yan2026memoryr2,wang2026cmimem,ye2026chime,cao2026memcalib,suresh2026grounding}。多正例边缘似然与弱监督语义解析中的 maximum marginal likelihood 具有形式关联 \citep{berant2013semantic}；验证器筛选和 best-of-\(N\) 则提供“生成—认证—保留”式训练的通用背景 \citep{cobbe2021verifiers,shao2024deepseekmath}。

ARC 的区别不在于否定这些路线，而在于监督单位：离线枚举/搜索完整 \((S,g)\) 后，经部署同款 `Build/Audit/QA` 链路认证，保留**全部** PASS 终态，而不任选单一标签或将 UNKNOWN/未搜索候选视为负例。由此，结构合法性、来源忠实性、查询覆盖与下游效用被分别核验；部署阶段只执行一次最终构造，适用于“写一次、读多次”的持久记忆场景。

## 参考文献（ICLR/natbib citation keys）

> 以下条目与正文中的 citation key 一一对应。使用 ICLR LaTeX 模板时，可将这些条目转写或导入 `.bib` 文件，并以正文中的 `\citet{}` / `\citep{}` 命令生成作者—年份引用。标注“arXiv 预印本”的条目没有以预印本替代已存在的正式发表版本。

- **lewis2020retrieval** — Lewis, P., Perez, E., Piktus, A., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* In *Advances in Neural Information Processing Systems*, 33, 9459–9474. https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html
- **park2023generative** — Park, J. S., O’Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). *Generative Agents: Interactive Simulacra of Human Behavior.* In *Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology (UIST ’23)*. https://doi.org/10.1145/3586183.3606763
- **zhong2024memorybank** — Zhong, W., Guo, L., Gao, Q., Ye, H., & Wang, Y. (2024). *MemoryBank: Enhancing Large Language Models with Long-Term Memory.* *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(17), 19724–19731. https://doi.org/10.1609/aaai.v38i17.29946
- **packer2023memgpt** — Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., & Gonzalez, J. E. (2023). *MemGPT: Towards LLMs as Operating Systems.* arXiv preprint arXiv:2310.08560. https://arxiv.org/abs/2310.08560
- **maharana2024locomo** — Maharana, A., Lee, D.-H., Tulyakov, S., Bansal, M., Barbieri, F., & Fang, Y. (2024). *Evaluating Very Long-Term Conversational Memory of LLM Agents.* In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 13851–13870. https://doi.org/10.18653/v1/2024.acl-long.747
- **wu2025longmemeval** — Wu, D., Wang, H., Yu, W., Zhang, Y., Chang, K.-W., & Yu, D. (2025). *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory.* In *International Conference on Learning Representations (ICLR)*. https://proceedings.iclr.cc/paper_files/paper/2025/hash/d813d324dbf0598bbdc9c8e79740ed01-Abstract-Conference.html
- **chhikara2025mem0** — Chhikara, P., Khant, D., Aryan, S., Singh, T., & Yadav, D. (2025). *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory.* In *ECAI 2025*, *Frontiers in Artificial Intelligence and Applications*, 413, 2993–3000. https://doi.org/10.3233/FAIA251160
- **xu2025amem** — Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., & Zhang, Y. (2025). *A-Mem: Agentic Memory for LLM Agents.* In *Advances in Neural Information Processing Systems*, 38, 17577–17604. https://doi.org/10.52202/085713-0593
- **rasmussen2025zep** — Rasmussen, P., et al. (2025). *Zep: A Temporal Knowledge Graph Architecture for Agent Memory.* arXiv preprint arXiv:2501.13956. https://arxiv.org/abs/2501.13956
- **kang2025memoryos** — Kang, J., Ji, M., Zhao, Z., & Bai, T. (2025). *Memory OS of AI Agent.* In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing*, 25961–25970. https://doi.org/10.18653/v1/2025.emnlp-main.1318
- **fang2026lightmem** — Fang, J., Deng, X., Xu, H., et al. (2026). *LightMem: Lightweight and Efficient Memory-Augmented Generation.* In *International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=dyJ0GWpjJB
- **liu2026simplemem** — Liu, J., Su, Y., Xia, P., Zhou, Y., Han, S., Zheng, Z., Xie, C., Ding, M., & Yao, H. (2026). *SimpleMem: Efficient Lifelong Memory for LLM Agents.* arXiv preprint arXiv:2601.02553. https://arxiv.org/abs/2601.02553
- **nan2025nemori** — Nan, J., Ma, W., Wu, W., & Chen, Y. (2025). *NEMORI: Self-Organizing Agent Memory Inspired by Cognitive Science.* arXiv preprint arXiv:2508.03341. https://arxiv.org/abs/2508.03341
- **zhang2026amac** — Zhang, G., Jiang, W., Wang, X., Behr, A., Zhao, K., Friedman, J., Chu, X., & Anoun, A. (2026). *Adaptive Memory Admission Control for LLM Agents.* In *ICLR 2026 Workshop on MemAgent*. https://arxiv.org/abs/2603.04549
- **liao2026leanmem** — Liao, Y., Wu, L., Hou, M., Liu, H., Wu, H., & Wang, Z. (2026). *LeanMem: Simple and Efficient Long-Term Memory for LLM Agents.* arXiv preprint arXiv:2608.03463. https://arxiv.org/abs/2608.03463
- **qiu2026dimmem** — Qiu, W., et al. (2026). *DimMem: Dimensional Structuring for Efficient Long-Term Agent Memory.* arXiv preprint arXiv:2605.15759. https://arxiv.org/abs/2605.15759
- **ji2026infinimemory** — Ji, S., et al. (2026). *Infini Memory: Maintainable Topic Documents for Long-Term LLM Agent Memory.* arXiv preprint arXiv:2606.10677. https://arxiv.org/abs/2606.10677
- **luo2026memsif** — Luo, Y., Xu, X., & Yang, Z. (2026). *MemSIF: From Structured Interactions to Dual-Track Fact Memory for LLM Agents.* arXiv preprint arXiv:2608.01742. https://arxiv.org/abs/2608.01742
- **li2026lycheememory** — Li, D., et al. (2026). *LycheeMemory V2: Efficient Long-Term Memory for LLM Agents via Semantic Segment-Level Consolidation.* arXiv preprint arXiv:2608.12990. https://arxiv.org/abs/2608.12990
- **yacoubi2026memorylace** — Yacoubi, M., et al. (2026). *MemoryLACE: Memory Lifecycle-Aware Consolidation and Evidence Retrieval.* arXiv preprint arXiv:2609.03201. https://arxiv.org/abs/2609.03201
- **latimer2025hindsight** — Latimer, C., et al. (2025). *Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects.* arXiv preprint arXiv:2512.12818. https://arxiv.org/abs/2512.12818
- **cui2026edgemem** — Cui, Z., et al. (2026). *EdgeMem: LLM-Free Agent Memory Construction and Retrieval via Evidence-Preserving Multi-Anchor Hypergraph.* arXiv preprint arXiv:2609.05553. https://arxiv.org/abs/2609.05553
- **song2026realm** — Song, Y., et al. (2026). *Retrieval-Driven Memory Reconsolidation for Long-Term LLM Agents.* arXiv preprint arXiv:2609.16053. https://arxiv.org/abs/2609.16053
- **zhao2026rpmem** — Zhao, F., et al. (2026). *RPMem: Learning Long-Term Recurrent Parametric Memory Across Sessions for LLM Agents.* arXiv preprint arXiv:2609.23466. https://arxiv.org/abs/2609.23466
- **xu2026memgas** — Xu, D., Wen, Y., Jia, P., et al. (2026). *From Single to Multi-Granularity: Toward Long-Term Memory Association and Selection of Conversational Agents.* In *International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=i2yIvZARnG
- **jiang2026magma** — Jiang, D., Li, Y., Li, G., & Li, B. (2026). *MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents.* In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 36848–36865. https://doi.org/10.18653/v1/2026.acl-long.1709
- **hu2026compassmem** — Hu, Y., Liu, J., Tan, J., Zhu, Y., & Dou, Z. (2026). *Memory Matters More: Event-Centric Memory as a Logic Map for Agent Searching and Reasoning.* In *Findings of the Association for Computational Linguistics: ACL 2026*, 22389–22407. https://doi.org/10.18653/v1/2026.findings-acl.1123
- **ji2026ripplemem** — Ji, J., et al. (2026). *RippleMem: From Isolated Retrieval to Associative Recollection for Long-Term Agent Memory.* arXiv preprint arXiv:2608.13334. https://arxiv.org/abs/2608.13334
- **lin2026hero** — Lin, Y., et al. (2026). *HERO: Human-profile Enhanced Retrieval Optimization Framework for Long-term Agent Memory.* arXiv preprint arXiv:2608.22310. https://arxiv.org/abs/2608.22310
- **lu2026fluxmem** — Lu, M., Wu, M., Liu, F., et al. (2026). *Choosing How to Remember: Adaptive Memory Structures for LLM Agents.* arXiv preprint arXiv:2602.14038. https://arxiv.org/abs/2602.14038
- **zhao2026mesa** — Zhao, B., Chen, Y., Feng, Y., et al. (2026). *MESA: Task-Adaptive Multi-Structure Evidence Selection for Long-Horizon Agent Memory.* arXiv preprint arXiv:2608.10108. https://arxiv.org/abs/2608.10108
- **zhang2026budgetmem** — Zhang, H., Yue, H., Feng, T., et al. (2026). *Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory.* In *Proceedings of the 43rd International Conference on Machine Learning (ICML)*. https://arxiv.org/abs/2602.06025
- **ma2026memchain** — Ma, Y., et al. (2026). *MemChain: Learning Interpretable Memory Traces for Memory-Augmented LLM Agents.* arXiv preprint arXiv:2607.24097. https://arxiv.org/abs/2607.24097
- **yu2026lazymem** — Yu, J., et al. (2026). *LazyMem: Retrieve Broadly, Construct Selectively for Efficient Long-Term Agent Memory.* arXiv preprint arXiv:2607.22690. https://arxiv.org/abs/2607.22690
- **gao2026memo** — Gao, X., et al. (2026). *MEMO: Multimodal Evidence Memory Organization for Long-Horizon LLM Agents.* arXiv preprint arXiv:2609.07471. https://arxiv.org/abs/2609.07471
- **li2026ember** — Li, Y., Banerjee, S., & Che, T. (2026). *EMBER: Efficient Memory via Budgeted Evidence Retention for Long-Horizon Agents.* arXiv preprint arXiv:2606.05894. https://arxiv.org/abs/2606.05894
- **kim2026memrefine** — Kim, M., et al. (2026). *MemRefine: LLM-Guided Compression for Long-Term Agent Memory.* arXiv preprint arXiv:2606.13177. https://arxiv.org/abs/2606.13177
- **yan2026memoryr1** — Yan, S., Yang, X., Huang, Z., et al. (2026). *Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning.* In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 12805–12825. https://doi.org/10.18653/v1/2026.acl-long.583
- **wang2025memalpha** — Wang, Y., et al. (2025). *Mem-\(\alpha\): Learning Memory Construction via Reinforcement Learning.* arXiv preprint arXiv:2509.25911. https://arxiv.org/abs/2509.25911
- **shen2026membuilder** — Shen, Z., Wu, Z., Lai, F., Lian, S., & Rao, Y. (2026). *MemBuilder: Reinforcing LLMs for Long-Term Memory Construction via Attributed Dense Rewards.* In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 27868–27887. https://doi.org/10.18653/v1/2026.acl-long.1284
- **yu2026agemem** — Yu, Y., Yao, L., Xie, Y., Tan, Q., Feng, J., Li, Y., & Wu, L. (2026). *Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents.* In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 21457–21483. https://doi.org/10.18653/v1/2026.acl-long.981
- **zhou2026mem1** — Zhou, Z., Qu, A., Wu, Z., et al. (2026). *MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents.* In *International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=XY8AaxDSLb
- **yan2026memoryr2** — Yan, S., Bahloul, A., Nie, E., et al. (2026). *Memory-R2: Fair Credit Assignment for Long-Horizon Memory-Augmented LLM Agents.* arXiv preprint arXiv:2605.21768. https://arxiv.org/abs/2605.21768
- **wang2026cmimem** — Wang, Y., et al. (2026). *CMI-Mem: Toward Generalizable Long-Term Memory Management via CMI-Augmented Reinforcement Learning.* arXiv preprint arXiv:2607.20553. https://arxiv.org/abs/2607.20553
- **ye2026chime** — Ye, Y., et al. (2026). *CHIME: Credit-Aware Hierarchical Memory Evolution for Long-Horizon Agentic Planning.* arXiv preprint arXiv:2609.02074. https://arxiv.org/abs/2609.02074
- **cao2026memcalib** — Cao, R., et al. (2026). *MemCalib: Benchmarking and Optimizing Memory Use in LLM Agents.* arXiv preprint arXiv:2609.24259. https://arxiv.org/abs/2609.24259
- **suresh2026grounding** — Suresh, S., et al. (2026). *Grounding Agent Memory: Environment-Probing Curation for Enterprise Agents.* arXiv preprint arXiv:2609.11060. https://arxiv.org/abs/2609.11060
- **berant2013semantic** — Berant, J., Chou, A., Frostig, R., & Liang, P. (2013). *Semantic Parsing on Freebase from Question-Answer Pairs.* In *Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing*, 1533–1544. https://aclanthology.org/D13-1160/
- **cobbe2021verifiers** — Cobbe, K., Kosaraju, V., Bavarian, M., et al. (2021). *Training Verifiers to Solve Math Word Problems.* arXiv preprint arXiv:2110.14168. https://arxiv.org/abs/2110.14168
- **shao2024deepseekmath** — Shao, Z., Wang, P., Zhu, Q., et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv preprint arXiv:2402.03300. https://arxiv.org/abs/2402.03300
