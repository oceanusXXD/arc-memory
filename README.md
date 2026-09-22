# ARC

ARC 是 `docs/papers/3_method.md` 的代码实现。它把长期记忆的决策放在写入阶段：先从写入前状态和新历史生成与查询无关的候选集合 \(E_i=C(X_i,M_i^-)\)，再按内容优先顺序选择来源集合 \(S\)，最后在给定 \(S\) 的条件下选择架构 \(g\)。构建输入使用完整序列化结果和冻结 tokenizer 计数，部署时只执行一次 `Build+Update`；查询阶段固定执行 `Retrieve+Answer`，不重新调用 selector、builder 或 updater。

离线编译器对有限的 \((S,g)\) 候选域执行真实构建、持久化更新和未来查询评估，并记录结构、来源、需求、效用、覆盖率三态认证（`PASS/FAIL/UNKNOWN`）以及生命周期成本。训练使用所有通过认证的方案作为多正例监督；空集合使用规范终态 `(Flat, ∅)`。完整实现说明见 [docs/algorithm.md](docs/algorithm.md)，理论定义以 [docs/papers/3_method.md](docs/papers/3_method.md) 为准。

## 代码结构

- `src/arc/algorithm/`：候选域、五种架构、序列化与构建、三层认证、离线编译和内容优先 selector。
- `src/arc/agent/`：持久记忆状态、写入/查询边界、Claude Code 调用和数据处理。
- `src/arc/baseline/`：`no_memory`、`full_memory`、`naive_rag`、`rank_pack` 和 `our` 评估路径。
- `scripts/`：数据准备、真实服务检查、桥接服务和运行入口。
- `tests/`：算法边界和语义验证测试。

## 安装与本地检查

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=src
python -m compileall -q src scripts tests
pytest -q
```

这些检查不调用远程模型。真实 LoCoMo 运行需要数据、冻结检索向量、selector checkpoint 和模型凭据；本仓库不再包含实验用 `configs/*.yaml` 文件，因此真实运行时需要自行提供 YAML 配置并通过 `--config` 传入。命令行入口的默认配置路径仍是 `configs/locomo.yaml`，没有该文件时不要省略 `--config`。

## 配置和模型调用

Python 入口会自动读取仓库根目录的 `.env`。配置可用 `arc.agent.config.load_config()` 的内置默认值，也可用 YAML 覆盖默认值。生成模型统一由完整 Claude Code CLI（`claude -p --output-format json`）调用；SiliconFlow 通过本地 Anthropic Messages relay 接入，本地 Qwen 则使用仓库内的 bridge。仓库没有 Python/OpenAI 风格的直连模型客户端。

云端调用需要配置 `ANTHROPIC_BASE_URL` 和对应的 `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN`，或在 YAML 中指定 `claude_code.api_key_env`。本地 bridge 的启动方式如下：

```bash
python scripts/anthropic_bridge.py --port 8080 --env-file runs/local/claude.env --thinking on
set -a; source runs/local/claude.env; set +a
python scripts/check_credentials.py --config /path/to/config.yaml
```

每次模型调用都会在配置的 `paths.runs_dir/<run_id>/calls/` 下保存脱敏后的请求、原始响应、usage 和错误状态。

## 离线编译流程

编译输入必须已经包含与查询无关的 `candidate_sources`（或兼容字段 `sources`）、`future_queries` 和 Grok-4.6 生成并验证的 `requirements`/`packages`。需求标注、来源候选和完整候选域的含义见 [docs/algorithm.md](docs/algorithm.md)。使用自己的配置和数据时：

```bash
CFG=/path/to/config.yaml
export PYTHONPATH=src

# 从会话历史与冻结来源向量生成 compiler 输入
python -m arc.agent.data.requirements --config "$CFG" \
  --output /path/to/requirements.jsonl

# 默认评估完整有限候选域；只有显式指定 --limit 才会截断
python -m arc.algorithm.compiler --config "$CFG" \
  --input /path/to/requirements.jsonl \
  --output /path/to/compilation.jsonl

# 用编译档案训练 selector
python -m arc.algorithm.selector \
  --input /path/to/compilation.jsonl \
  --output /path/to/arc_selector.pt --epochs 30
```

`requirements` 阶段不读取当前问题来构造写入候选；未来查询只用于离线价值评估。编译器不会把 `UNKNOWN` 改写成训练负例，`domain_complete=false` 表示候选域被显式截断或尚未全部评估。

## 在线写入与查询

在线 `our` 路径必须提供已拟合 selector checkpoint、冻结来源 embedding、匹配的 embedding metadata 和冻结 tokenizer。写入请求使用 `phase="write"`，查询请求使用同一持久状态的 `phase="query"`：

```python
from arc.agent import Agent, MemoryRequest
from arc.agent.config import load_config
from arc.algorithm.adapter import build_memory

config = load_config("/path/to/config.yaml")
agent = Agent(config, build_memory)

write = MemoryRequest(
    question="", write_batch=history, phase="write",
    sample_id="sample", qa_id="write-1",
)
written = agent.prepare_memory(write)

query = MemoryRequest(
    question="Alice 住在哪里？", evidence=[], phase="query",
    memory_state=written.persistent_state,
)
answer = agent.run(query.question, agent.prepare_memory(query))
```

写入阶段 selector 的特征不使用 query；查询阶段只读取 `persistent_state`。若只需要显式 baseline，可在评估入口选择 `no_memory`、`full_memory`、`naive_rag` 或 `rank_pack`。

## 公开仓库

代码已发布到 [oceanusXXD/arc-memory](https://github.com/oceanusXXD/arc-memory)。
