# R2W-GBM

R2W-GBM 是 LoCoMo 的写时记忆策略。每个对话 turn 选择十个存储臂之一，或选择 `none`：

```text
raw | raw+kv | raw+event | raw+graph | raw+hq |
sum | sum+kv | sum+event | sum+graph | sum+hq | none
```

主链如下：

1. 从真实 `conversation.session_N`、`session_N_date_time`、`dia_id` 与 `qa.evidence` 读取 LoCoMo。
2. L1 仅生成检索/QA 诊断。L2 在完整 raw 背景下逐 turn、逐臂测量 `p vs none` 的端到端 token-F1 差分，得到 `ΔE`、`ΔF`、带符号 `Ψ`、命中率、标准误和成本分量。
3. 五个按 conversation 分组的 LightGBM 分别学习第一关收益包络和第二关 pooled arm 效应/命中率。
4. 在线第一关先用 raw 的确定性成本判 `none`；放行后才构建 raw 与九个 non-raw 草稿，经固定事实证书门后按风险调整净值选臂。

旧 PyTorch 多头网络、CRC、迭代 L3 标签和 v1 工件均不支持加载。

## 运行前提

- Python 3.10 或更高版本。
- LoCoMo 文件，例如 `../data/LoCoMo/data/locomo10.json`。也支持 LongMemEval 的原始
  `longmemeval_s_cleaned.json` 或其 record 子集：每个 haystack session 会成为一个
  R2W memory，`answer_session_ids` 严格映射到 evidence。
- 真实训练与基线 QA 需要 `LLM_API_KEY`；远程 embedding 需要 `EMBEDDING_API_KEY`，未设置时使用 `LLM_API_KEY`。
- QG、NER、本地构建器和 NLI 模型由 Hugging Face 下载；首次运行需要网络和磁盘空间。

数据不随代码仓库提交。可从 [LoCoMo 官方仓库](https://github.com/snap-research/locomo) 获取，并放到 `../data/LoCoMo/data/locomo10.json`，或在命令中传入实际文件路径。

PowerShell：

```powershell
cd code
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

$env:LLM_API_KEY = "<your-llm-key>"
$env:EMBEDDING_API_KEY = "<your-embedding-key>" # 若相同可省略
```

若已从 FutureMem 导入本地 `code/.env`，运行时会自动读取它（该文件被 git 忽略）。
使用其中的阿里 embedding profile 时，在启动 Python 前设置：

```powershell
$env:R2W_ENV_FILE = '.env.rag_futuremem10'
```

`scripts/setup.sh` 与 `scripts/run_train.sh` 供 Bash 使用；Windows 请使用以上命令。

## 配置

生成一份完整且严格的配置后，按自己的模型、端点和维度修改：

```powershell
python -m r2w.pipeline_train --write-default-config configs/r2w.json
```

必须核对：

- `embedding_model`、`embedding_base_url`、`embedding_dim`：远程返回维度必须等于 `embedding_dim`。`embedding_base_url=""` 时使用本地 SentenceTransformer。
- `llm_model`、`llm_base_url`：结构化构建、冻结 reader 与事实拆分使用同一个真实 OpenAI-compatible API。
- `constructor_backend`：`api` 为 JSON API 构建；`local` 为 `constructor_local_model` 指定的本地 Seq2Seq JSON 模型。`local` 仍需要 LLM API 用于 reader 和事实证书。
- `retrieval_k`、`teacher_rank_limit`、`lambda_w/lambda_r/lambda_s`、`fidelity_ratio` 都会进入工件。

配置字段严格校验；缺失或额外字段都会拒绝加载。

## 真实训练

```powershell
python -m r2w.pipeline_train `
  --config configs/r2w.json `
  --data ../data/LoCoMo/data/locomo10.json `
  --out artifacts/locomo-r2w-gbm
```

这是实际训练入口，不会使用 fake reader 或 fake constructor。它按对话划分 60% train、20% validation、20% held-out（LoCoMo 10 段即 6/2/2）；在每个 split 构建全部十个表示并执行 L2 `p vs none` 重放。它会产生大量 API 调用，请先在小数据子集检查端点、配额和 JSON 契约。

输出目录：

```text
config.json             # 严格 v3 配置快照
model.joblib            # 五折 GBM 集成
features.joblib         # 训练区特征统计
case_memory.json        # 局部审计案例库，不混合修正线上效应
policy.json             # validation 选定的 λ/κ/θ
measurement_data.npz    # validation 测量分量，可重选 λ/κ
provider_usage.json     # 真实 API 返回的 token 用量；不会估算缺失字段
manifest.json           # 版本、哈希、split 与测量协议
```

只重选运营汇率，不重做 L2 或重训 GBM：

```powershell
python -m r2w.pipeline_calibrate `
  --artifacts artifacts/locomo-r2w-gbm `
  --lambda-w 0.0 --lambda-r 0.01 --lambda-s 0.0
```

## 无 API 的人工 value 诊断

`pipeline_query_oracle_gbm` 用真实 LoCoMo query 作为条件，为该 query 的 gold
evidence memory 选择存储 action。它通过确定性的 query/memory 统计、交互特征、草稿
维度和生产十臂特征训练同一个 pooled GBM，再由零成本 `DecisionPolicy` 选择预测 value
最高的 action。整个过程不调用 API、embedding、LLM、QG 或 reader：

```powershell
python -m r2w.pipeline_query_oracle_gbm `
  --data ../data/LoCoMo/data/locomo10.json `
  --manifest oracle_manifests/locomo_query_value_oracle.json `
  --out reports/locomo-query-value-oracle-gbm.json
```

清单包含 7 个 train conversation 的 42 个 query-memory 样本和 3 个完全隔离的 test
conversation 的 18 个样本。每条样本人工选择一个 value profile；同一 profile 复用一组
完整的十臂 value。answer、LoCoMo category、profile 名和 oracle value 均不进入特征。
当前 profile 的 oracle-best 覆盖 6/10 个 action，因此它验证十臂 value 排序链路和六类
最佳 action 的泛化，尚未覆盖每个 action 都成为最佳臂的情况。

当前固定清单的参考结果是严格 action 一致 9/18（50.0%）、oracle 位于预测 Top-3
15/18（83.3%）、compression 轴一致 14/18（77.8%）、key 轴一致 11/18
（61.1%）。训练集多数 action 在 test 上为 6/18（33.3%），十臂均匀随机期望为
10%。严格不一致也可能来自 profile 对不同长度或冗余 memory 复用同一 value；报告会保留
逐样本十臂 oracle/predicted value、rank 和 regret 供人工审计。

这个结果是人工 value 与 gold evidence 条件下的离线诊断。它不代表 L2 causal effect、
真实检索、写入端到端效果或 LoCoMo benchmark；真实训练仍使用上一节的
`pipeline_train`。

## 公平基线

`raw` 是主基线 `always_raw_naive_rag`：每个 turn 原样入索引，随后使用与 R2W 相同的 embedding、RRF、`retrieval_k` 和 reader。`none` 只是无热记忆诊断线；它不是 Naive RAG。`full_history_context` 是不经检索而将完整历史交给 reader 的长上下文参考线。

默认运行 `none`、`full_history_context`、`always_raw_naive_rag` 与固定摘要：

```powershell
python -m r2w.pipeline_baselines `
  --config configs/r2w.json `
  --data ../data/LoCoMo/data/locomo10.json `
  --out reports/baselines.json
```

全部固定表示：

```powershell
python -m r2w.pipeline_baselines `
  --config configs/r2w.json `
  --data ../data/LoCoMo/data/locomo10.json `
  --actions raw,raw+kv,raw+event,raw+graph,raw+hq,sum,sum+kv,sum+event,sum+graph,sum+hq `
  --out reports/all-fixed-baselines.json
```

报告包含端到端 `token_f1`、evidence recall@k、平均 reader 上下文大小、固定写入/索引量及实际 API token 账本。所有固定臂共用同一检索和 reader。
`mean_normalized_exact_match` 是和 token-F1 使用同一分词规则的严格准确率；它不替代
LongMemEval 需要外部 judge 的官方 accuracy。

LongMemEval 真实 API smoke test（这里的 `sum` 是 R2W 的固定压缩候选，完整 R2W
策略评测需要先以训练 split 训练 v3 artifact）：

```powershell
$env:R2W_ENV_FILE = '.env.rag_futuremem10'
$env:R2W_EMBEDDING_CACHE = 'reports/longmemeval-r2w-embeddings.npz'
python -m r2w.pipeline_baselines `
  --data ../futuremem/external/LongMemEval/data/longmemeval_s_cleaned.json `
  --actions raw,sum --without-none --without-full-context `
  --limit-conversations 4 `
  --out reports/longmemeval-api-baselines.json
```

`R2W_EMBEDDING_CACHE` 以表示文本的 SHA-256 键持久化向量，只有文本完全相同才复用。
FutureMem 以 `episode_id::memory_id` 作为键的缓存不能直接用于 R2W 的带元数据表示。

## 在线写入

在线推理必须提供固定的事实证书 NLI 模型。事实拆分继续使用配置里的真实 LLM；`--certificate-weights` 是 `cov_f`、`1-contra`、`margin` 的三个固定非负权重，和为 1。

```powershell
python -m r2w.pipeline_infer `
  --artifacts artifacts/locomo-r2w-gbm `
  --turn '{"dia_id":"D1:42","speaker":"Alice","text":"I moved yesterday.","timestamp":"1:00 pm on 8 May, 2023","session":1}' `
  --context '[]' `
  --speaker-a Alice --speaker-b Bob `
  --certificate-nli-model cross-encoder/nli-deberta-v3-base
```

第一关判 `none` 时不会加载 QG、结构化构建器或 NLI，也不会构建九个 non-raw 草稿。放行后才构建全部候选并执行 `fidelity_ratio` 证书门。`ShadowAwareMemoryWriter` 提供热索引、冷归档、`none` 影子和非 raw 的 raw shadow。

## 成本与 token

算法决策冻结为 `word_count` 三分量：

```text
[write, index, reader_payload]
raw:     [0, payload, payload]
non-raw: [payload + keys, payload + keys, payload]
```

它不是模型 tokenizer 或供应商账单 token。真实 API 用量位于 `provider_usage.json` 或基线报告的 `provider_usage`，仅累计 API 响应显式提供的 `usage.prompt_tokens`、`usage.completion_tokens` 和 `usage.total_tokens`；缺失 usage 会标记为未报告，绝不以词数估算。

报告 R2W 对 Naive RAG 的成本时请拆开 ingestion 与 query：

\[
C_{lifetime}(N)=C_{ingest}+N\,C_{query}.
\]

为了产生逐臂标签，训练会构建九个候选草稿；这是离线测量开销，应与部署时每 turn 仅执行最终选中动作的成本分开报告。

## 验证

无 API 测试只验证算法、工件和数据流，不代表真实模型效果：

```powershell
python -m unittest discover -s tests -t . -v
python -m compileall -q r2w tests
```

测试覆盖 LoCoMo 映射、L2 公式、五折 GBM、第一关 `none`、十臂 argmax、证书拒绝、工件拒绝、writer 与固定基线。真实 API/本地构建器联网与权重集成测试不在离线测试范围内；应先做小数据真实 smoke test。
