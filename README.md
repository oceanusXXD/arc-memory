# arc

arc 按联合证据—上下文选择流程实现：写入阶段固定生成与查询无关的候选 (E)，先由 H 选择来源子集 S，再对完整 S 条件化选择结构 g，使用完整规范化序列化输入和配置的聊天模板计算 (T(E,S,g))，离线枚举 architecture-content 候选并执行结构/来源/需求/效用/覆盖率的 PASS/FAIL/UNKNOWN 三态认证，在线 H 只生成递增来源 ID 和 STOP。获胜方案只执行一次 Build+Update；查询阶段只走固定 Retrieve+Answer，不再调用 H。在线 `our` 要求已拟合的 selector checkpoint、冻结 embedding 和冻结 tokenizer；缺少任一组件会直接报错。`rank_pack` 是单独的显式 baseline，不是在线路径的隐式兜底。

源码分为 arc/algorithm、arc/agent 和 arc/baseline。算法目录只保留候选域、编译、选择器、构建适配器和标准库核心；Baseline 保留 Full、No-memory、Naive-RAG、Rank-pack 与 arc 评估，审计记录保留选择、结构、构建状态和费用日志。

模型统一通过完整 Claude Code CLI（`claude -p --output-format json`）调用。支持两条路径：云端是 Claude Code → 本地 Anthropic Messages relay → 硅基流动；本地是 Claude Code → 本仓库自带的 Qwen vLLM bridge。仓库没有 Python/OpenAI 风格的模型直连客户端。所有调用保存原始响应和 usage。配置见 configs/locomo.yaml，环境变量见 .env.example。

### Claude Code 凭据与本地 bridge

所有生成角色当前使用 `Qwen/Qwen3.5-4B`，温度 `0.7`，并开启 thinking（预算默认 1024）。保留原生 Claude Code CLI；
云端硅基流动调用经过本地兼容中转，整理 system 消息并将 thinking、温度与输出上限写入 API 请求。

Python 入口自动加载当前仓库的 `.env`，支持现有的 `SILICONFLOW_API_KEY`。
也可以设置 `ANTHROPIC_API_KEY` 或 `ANTHROPIC_AUTH_TOKEN`。密钥属于指定的 API 地址，
`LIGHTNING_API_KEY` 不会自动映射。直接使用 CLI 前先导出环境：

```bash
set -a; source .env; set +a
python scripts/check_credentials.py
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://api.siliconflow.cn}"
# 云端路径只允许硅基流动 endpoint；当前硅基流动密钥供 Claude Code 使用。
if [ -z "${ANTHROPIC_API_KEY:-}${ANTHROPIC_AUTH_TOKEN:-}" ]; then
  export ANTHROPIC_API_KEY="$SILICONFLOW_API_KEY"
fi
```

本地路径由本仓库自己的 `src/arc/local_qwen.py` 和 `src/arc/vllm_proxy.py` 启动服务并完成协议转换，和其他仓库没有运行时依赖。
需要本机 vLLM/GPU。

```bash
# 终端一：启动 4B 模型和 bridge，ready 后生成 claude.env。
python scripts/anthropic_bridge.py --port 8080 --env-file runs/local/claude.env --thinking on
# 终端二：
set -a; source runs/local/claude.env; set +a
python scripts/check_credentials.py
```

每次 CLI 调用保存到 `runs/<run_id>/calls/*.json`，含原始 stdout/stderr、响应、usage、
请求参数和错误状态，凭据经过脱敏。非零退出码、`is_error`、API 错误和超时都按失败处理。

## 安装与真实验证

    python -m pip install -r requirements.txt
    export PYTHONPATH=src
    python scripts/verify_real.py --config configs/verify_small.yaml --stage all

验证脚本只使用真实 LoCoMo 问题，并通过 Claude Code 调用真实 `Qwen/Qwen3.5-4B` API。
`configs/verify_small.yaml` 只保留一个真实 conversation，用于把数据量压到最小，但不使用假数据。

## 离线编译

先从会话历史和冻结来源向量生成与问题无关的 compiler 候选输入。该命令不调用模型；需求标注会在离线编译阶段由 compiler/teacher 生成：

    python -m arc.agent.data.requirements --config configs/locomo.yaml

输入 JSONL 每行包含候选 sources、future_queries、requirements，以及每项需求的 packages。正式编译默认评估完整有限候选域，UNKNOWN 不会被当作 FAIL。非空输入只调用一次 G，空输入成本和调用数均为零。

    python -m arc.algorithm.compiler --config configs/locomo.yaml --input data/processed/requirements.jsonl --output runs/locomo_arc/compilation.jsonl

使用包含冻结查询/来源向量和完整候选成本的编译档案训练 H：

    python -m arc.algorithm.selector --input runs/locomo_arc/compilation.jsonl --output models/arc_selector.pt --epochs 30

## Agent 与基线

    from arc.agent import Agent, MemoryRequest
    from arc.agent.config import load_config
    from arc.algorithm.adapter import build_memory
    config = load_config("configs/locomo.yaml")
    agent = Agent(config, build_memory)
    request = MemoryRequest("Alice 住在哪里？", [{"source_id": "s1", "text": "Alice lives in Oslo."}])
    packet = agent.prepare_memory(request)
    answer = agent.run(request.question, packet)

评估会写入 scores、costs、audit、summary 和 manifest；审计包含 selection、来源 ID、构建状态、原始记忆、结构错误、构建/回答 usage、延迟和兜底原因。

    python -m arc.baseline.evaluate run --config configs/locomo.yaml --arms no_memory full_memory naive_rag rank_pack our --split final --limit 1

规范文档：方法章节见 [docs/papers/3_method.md](docs/papers/3_method.md)，算法实现说明见 [docs/algorithm.md](docs/algorithm.md)。
