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
- LoCoMo 文件，例如 `../data/LoCoMo/data/locomo10.json`。
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
