# 本地 Qwen3.5-9B + Claude Code CLI 部署与效果对齐

调查日期：2026-09-13。目标是和 **Claude Code CLI + 同一个 Qwen3.5-9B API** 对比。本文的显存配置是容量估算和实验起点；当前开发环境没有可用 CUDA GPU，尚未测出 9B 的吞吐、缓存收益或 API 任务成功率差异。

直接回答“是否满血、至少多少显存”：当前是未做权重量化的 BF16 文本部署，
尚未达到经过验证的完整多模态/API 等价状态。单会话 128K 文本部署保守建议
从 48GB 显存开始，256K 可优先考虑 80/96GB；更小显存也可能运行某些配置，
但没有一个显存数字能单独保证和 API 一致。简明配置表已放入根目录 README。

## 1. 调查结论

优先顺序是：**对齐权重/模板和 thinking -> 保证工具调用协议正确 -> 给足上下文和输出预算 -> 再优化缓存与吞吐**。仅把模型放进显存、开放一个 HTTP 端口，不能保证 Claude Code 工作正常。缓存通常改善 prefill/首 token 延迟，不会让 9B 获得更强的推理能力。[Qwen 模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)、[vLLM APC 文档](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)

同名 API 也可能使用不同 revision、量化、chat template、采样、thinking 预算或隐藏提示词。API 厂商不公开这些信息时，只能比较可观测的任务质量和延迟，不能承诺逐 token 相同。本仓库 `.env.example` 的 API 示例是 SiliconFlow；这不是对其内部推理配置的确认。

本次继续使用已有 bridge，链路如下：

```text
Claude Code CLI (Bash / Read / Edit / MCP / permissions)
  -> http://127.0.0.1:8080/v1/messages   本仓库 Anthropic bridge
  -> http://127.0.0.1:8000/v1/chat/completions   vLLM
  -> Qwen/Qwen3.5-9B
```

启动后的服务可接普通 `claude` 交互终端，也可接本仓库 `native` 电路评估入口；没有用自写 agent loop 替换 CLI。bridge 不执行工具。`ANTHROPIC_BASE_URL` 必须设为 `http://127.0.0.1:8080`，**不能附加 `/v1`**，因为 CLI 自己追加 `/v1/messages`。

当前 vLLM 源码也提供原生 `/v1/messages` 和 `/v1/messages/count_tokens`，所以不应宣称所有 vLLM 版本都必须加 bridge。本次保留已有适配层以统一采样、模型别名、token 统计和协议测试；原生端点可另作对照，但尚未完成真实 Qwen 工具会话验证。[vLLM 原生 Anthropic 路由](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/anthropic/api_router.py)

## 2. 模型与生成配置

Qwen3.5-9B 是带视觉编码器的模型，文本部分采用 24 层 Gated DeltaNet + 8 层 full attention 的混合布局；9B 的 FFN 是 dense，不能把大号 Qwen3.5 MoE 的八卡配置照搬过来。官方给出的原生窗口是 262,144，并建议复杂 thinking 任务至少保留 128K；缩到 16K/32K 是资源取舍，不能当成与长上下文 API 等效。[模型结构与部署建议](https://huggingface.co/Qwen/Qwen3.5-9B#model-overview)

| 项目 | 本次默认值 | 原因 |
| --- | --- | --- |
| 权重 | `Qwen/Qwen3.5-9B`，可选 `--revision COMMIT_SHA` | 使用 post-trained 模型；Base、LoRA、量化版本需单独对比 |
| 精度 | BF16 权重，`--kv-cache-dtype auto` | 先建立未量化的质量基线 |
| context / output | 131,072 / 32,768 tokens | 同时容纳 system、工具 schema、历史和 reasoning/final/tool JSON |
| thinking | `--thinking on` | 显式传入 `chat_template_kwargs.enable_thinking=true` |
| 工具解析 | `qwen3` reasoning + `qwen3_coder` tools | 官方 9B 部署示例的组合 |
| 并发 | `--max-num-seqs 1` | 单个交互式会话的实验起点 |
| prefill | chunked prefill，batch token 上限 8192 | 后续按 TTFT 和解码延迟调节 |
| 前缀缓存 | 开启；`--no-prefix-caching` 可真正关闭 | 对比连续工具会话的 prefill 复用 |

官方采样建议与本次 profile 对应关系如下。[Qwen Best Practices](https://huggingface.co/Qwen/Qwen3.5-9B#best-practices)

| profile | thinking | temperature | top_p | top_k | presence_penalty |
| --- | --- | --- | --- | --- | --- |
| `coding`（默认） | on | 0.6 | 0.95 | 20 | 0.0 |
| `general` | on | 1.0 | 0.95 | 20 | 1.5 |
| `coding` 或 `general` | off | 0.7 | 0.8 | 20 | 1.5 |
| `passthrough` | 任意 | 保留 CLI / backend 的取值 | 同左 | 同左 | 同左 |

前三行还固定 `min_p=0`、`repetition_penalty=1`。profile 会覆盖 CLI 发来的这些采样字段；需要严格复现 API 原始参数时用 `--sampling-profile passthrough`，并在 backend generation config 中固定 CLI 未提供的字段。`--thinking auto` 根据 Messages 的 `thinking.type` 开关模板，字段缺失时采用 backend 默认；它不把 Anthropic `budget_tokens` 或 `effort` 自动等价成 Qwen reasoning 预算。

Qwen 的历史 thinking 应交给官方 Jinja 模板处理：bridge 保留结构化 `reasoning_content`；模板区分旧用户轮次和当前工具轮次，避免手工把所有历史 `<think>` 拼进文本。固定 revision 时，本次命令同时固定 tokenizer revision。[官方 chat template](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/chat_template.jinja)

## 3. 机器配置与显存估算

以 BF16 文本推理估算，9B 参数约占 18 GB 十进制空间。只计算 full-attention KV：

```text
2 (K,V) * 8 layers * 4 KV heads * 256 head_dim * 2 bytes
= 32 KiB/token
64K / 128K / 256K tokens -> 约 2 / 4 / 8 GiB attention KV
```

这不包含 DeltaNet recurrent state、缓存块对齐、CUDA graph、临时 activation、allocator 和其他进程；混合架构的实际 cache 分配要看 vLLM 启动日志，不能把上面数字当成总显存。数字来自官方 9B 结构，属于推导。[模型结构](https://huggingface.co/Qwen/Qwen3.5-9B#model-overview)

| GPU 显存 | 建议起点 | 定位 |
| --- | --- | --- |
| 24 GB | BF16，context 16K、output 4K，utilization 0.90 | 小任务试跑；可能 OOM，不保证满足 Claude Code 的系统提示长度 |
| 32 GB | BF16，context 64K、output 16K | 可用性起点；长任务对 API 比较时要标明窗口差异 |
| 48 GB | BF16，context 128K、output 32K，utilization 0.90 | 本文推荐的单卡容量起点，仍需实测 |
| 80/96 GB | BF16，context 256K、output 32K，utilization 0.90 | 更长输入或多会话；还需确认 CLI 的压缩行为 |

单卡能容纳时先用 TP=1；两张卡的 NVLink/PCIe 通信可能抵消小模型的加速收益。TP=2 是容量/吞吐选项，不保证单会话更快。消费卡即使显存够用，显存带宽、功耗限制和散热也会影响 decode 速度；这里没有按卡型号承诺 tokens/s。

CPU 内存建议至少 32 GB、优先 64 GB；8 个以上可用 CPU 核心和本地 NVMe 是实用起点，模型文件/缓存至少预留 50 GB。它们是工程建议，不是官方最低配置。CPU offload、跨网络盘加载和多进程争用 GPU 都应单独测量，不能与纯 GPU BF16 基线混在一起。

## 4. 安装、启动与直接连接 CLI

推理环境与仓库的 SFT/RL 训练环境分开安装，避免升级 vLLM 顺带替换训练用 torch。按官方安装指引选择匹配驱动的 wheel；模型卡发布时的 nightly 要求不等于所有后续部署都要追 nightly。[vLLM 安装](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)、[Qwen recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)

```bash
uv venv /data/venvs/qwen-serve --python 3.12
uv pip install --python /data/venvs/qwen-serve/bin/python vllm --torch-backend=auto
export HF_HOME=/data/huggingface
export CUDA_VISIBLE_DEVICES=0

# 确认 GPU 和安装参数；成功验收后冻结这个环境的版本。
nvidia-smi
/data/venvs/qwen-serve/bin/vllm serve --help=all

# 无 GPU 也可以预览实际命令；不会下载或加载权重。
python agent/trace.py serve-local --dry-run

# 前台常驻；默认写 vLLM 日志，ready 后生成 CLI 的环境文件。
python agent/trace.py serve-local \
  --vllm-bin /data/venvs/qwen-serve/bin/vllm \
  --model Qwen/Qwen3.5-9B --served-model-name qwen-local \
  --port 8000 --bridge-port 8080 \
  --max-model-len 131072 --max-output-tokens 32768 \
  --thinking on --sampling-profile coding \
  --env-file artifacts/work/local-qwen/claude.env
```

首次下载慢时先下载模型，或增加 `--startup-timeout 1800`。正式实验加 `--revision` 指定实际的 Hugging Face commit SHA。后台就绪检查要求 `/v1/models` 包含所配置的模型名；端口被占用会报错，退出或启动失败会清理拥有的子进程。日志位于 `artifacts/work/local-qwen/vllm.log`，可以通过 `--log-file` 更改。

在第二个终端、仓库根目录下执行：

```bash
source artifacts/work/local-qwen/claude.env
claude --model qwen-local
```

这是普通 Claude Code CLI，仍由 CLI 管理工作目录、工具和权限。生成的环境变量只作用于当前 shell；退出本地服务后，环境文件并不代表 endpoint 仍可用。API 对照请使用单独的 shell，以免继承本地模型别名。

若要跑仓库的电路任务：

```bash
source artifacts/work/local-qwen/claude.env
python agent/trace.py native --client claude --task OTA-EASY \
  --model qwen-local --max-evals 8 --timeout 1800 \
  --out artifacts/work/local-qwen-ota
```

已有外部启动的 vLLM、只想让 `native` 临时创建 bridge 时：

```bash
# 有 backend key 时设置 VLLM_API_KEY；不会放进 argv。
python agent/trace.py native --local \
  --vllm-base-url http://127.0.0.1:8000/v1 --model qwen-local \
  --local-context-length 131072 --local-max-output-tokens 32768 \
  --local-thinking on --local-sampling-profile coding \
  --task OTA-EASY --out artifacts/work/local-qwen-existing
```

两种入口区别：`serve-local` 启动/管理模型和常驻 bridge；`native --local` 只启动临时 bridge，backend 必须已经存在。不要把 `--vllm-base-url` 指向 8080 的 Anthropic bridge。

bridge 没有访问鉴权，仅允许绑定 loopback；`local-vllm` 是 CLI 的占位凭据。远程 GPU 使用 SSH 端口转发或自行加鉴权网关。backend 需要鉴权时设置 `VLLM_API_KEY`；本次没有把真实 key 写进命令预览或 descriptor。

## 5. Claude Code 上下文与工具兼容

生成的环境将主模型、Haiku/Sonnet/Opus 等别名统一到 `qwen-local`，防止辅助调用请求不存在的模型。`ENABLE_TOOL_SEARCH=false` 提前加载工具，`CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` 避免发送 bridge 不支持的 deferred-tool 协议；没有关闭本地工具权限检查。[CLI 环境变量](https://code.claude.com/docs/en/env-vars)

同时设置 `CLAUDE_CODE_MAX_OUTPUT_TOKENS`、`CLAUDE_CODE_AUTO_COMPACT_WINDOW` 和 `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`。128K/32K 默认配置将压缩比例设为 68%，预留输出与约 8K 的保守余量。低于 100K 的 backend 窗口通过降低百分比补偿 CLI 压缩窗口的 100K 下限。未知模型的 tokenizer 估算仍可能存在偏差；首次启动在 `/context` 检查行为，长任务前实测接近边界的请求。超过 200K 时本工具仍保守按 200K 配置 CLI，未声称启用了 CLI 的完整 256K 窗口。[CLI context 与 auto-compaction](https://code.claude.com/docs/en/model-config#context-window-and-auto-compaction)

`max-model-len` 是输入与输出的总预算。bridge 使用 vLLM `/tokenize` 计算 `count_tokens`，包含同一份 tools 和 chat template，已移除“JSON 字符数 / 4”的估算；backend 不支持 `/tokenize` 时返回错误，不伪造精确 token 数。backend 的超窗 400 保持 400，避免让 CLI 误以为临时 502 而重试。

当前支持文本、thinking、tool use/result、非流式和流式 Messages。工具 JSON 分片先组装和校验，再以 Anthropic tool block 交给 CLI；普通文本/思考持续流式返回。非法 JSON、丢失 finish_reason 或中途断流显式报错。图片、PDF、tool_reference 等未支持的内容类型会拒绝，不会静默丢弃后假装完整输入。需要视觉能力时应使用经过验证的多模态协议链路。

## 6. 缓存与延迟调优

| 缓存/优化 | 作用 | 本次策略 |
| --- | --- | --- |
| HF 权重缓存 | 避免重复下载 | `HF_HOME` 指向持久化 NVMe |
| 编译/CUDA graph 缓存 | 降低部分冷启动开销 | 保留同一推理环境和缓存目录；首次启动单独计时 |
| GPU prefix KV/state cache | 重复前缀少做 prefill | 默认 APC 开启，服务跨会话常驻 |
| Anthropic `cache_control` | API 厂商的缓存语义和计费提示 | bridge 不转发该字段；vLLM 按 token 前缀自动复用 |
| KV 量化/offload/外部缓存 | 节省显存或扩展 cache 容量 | 不在 BF16 基线启用，需另测精度和传输延迟 |
| MTP/speculative decoding | 改善部分工作负载的 decode 延迟 | 不默认启用，与 APC 分开测试 |

缓存命中取决于**模板渲染后的相同 token 前缀**，不是请求 JSON 字节完全相同。保持 system prompt、工具顺序与 schema 稳定；压缩、编辑早期历史、切模型、重排工具都会降低复用。不要为了命中而删除当前文件内容或 tool result，更不要缓存完整工具回答后跳过实际执行。[APC 原理与限制](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)

Qwen3.5 是混合 attention/recurrent 架构，APC 还涉及 Mamba/DeltaNet state 的块对齐。官方 recipe 将部分 `align` cache 行为标为 experimental；某些 vLLM 版本对 APC、MTP 和分块的组合有限制。本次不强制高级 cache mode，继承安装版本对该模型的实现。若只有开启 APC 后失败，用 `--no-prefix-caching` 做同任务对照，再选择通过验收的 vLLM 版本；不是所有失败都能靠加显存解决。[Qwen recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)

调优顺序：先 BF16 单并发 + 128K；观察冷/热 TTFT、TPOT、prefix hit 和 preemption；再尝试 prefill batch 2048/4096/8192、并发 1/2/4。逐项更改并记录结果。APC 对长输出的逐 token 解码帮助有限，不能把冷启动、prefill 和 decode 混成一个“速度”。MTP 示例在不同版本可能叫 `mtp` 或 `qwen3_next_mtp`，使用该版本支持的参数，不照搬大号 MoE 的性能数字。

启动时已经传入 `--enable-prompt-tokens-details`。vLLM 返回 `prompt_tokens_details.cached_tokens` 时，bridge 将其映射为 `cache_read_input_tokens`，并从 Anthropic 的新输入计数中扣除，避免把缓存 token 重复相加。没有厂商相同的缓存创建/TTL 计费语义，不应拿本地这组数字估算 API 账单。

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8000/metrics | rg 'prefix_cache|time_to_first_token|inter_token_latency|kv_cache_usage|preemptions'
```

`8080/health` 只说明 bridge 在运行；模型是否 ready 看 backend models 和实际生成。启用了 `VLLM_API_KEY` 时，按部署版本要求为 backend 请求加认证头。

## 7. 和 API 做有效对照

固定 CLI 版本、模型 revision（API 不公开则记 unknown）、prompt、工具集、工作目录、任务集、输出/context/时间预算、thinking 与采样。使用未参与 SFT/RL 的任务，先至少 10 次配对试跑，再扩大样本；不能用一次成功判断“接近”。建议在项目内事先定义可接受的成功率差异，例如绝对差不超过 5 个百分点，但这个门槛不是已经达到的测试结果，10 个样本也不足以证明它。

分开记录这些指标：

- 质量：任务通过率、每轮有效评估率、真实测量指标、错误停止与遗漏要求；
- 工具：有效 JSON 比例、未知工具名、工具执行错误、无效/重复调用、重试次数；
- 资源：冷启动、TTFT p50/p95、TPOT、每任务总时长、peak VRAM、OOM/preemption；
- 缓存：冷请求和重复长前缀请求分组，记录 prefix hit token 增量，不能只看累计命中率；
- 预算：输入/输出 token、压缩次数、超窗与 `max_tokens` 截断，单位统一后再比较。

本仓库的 API relay `anthropic_proxy.py` 会缓冲上游完整响应，不能用它的首字节时间直接推导 API 原生 TTFT；端到端 CLI 用时仍可比较。需要纯推理 TTFT 时，用相同流式 HTTP 客户端分别测本地 backend 和 API，单独统计网络与 bridge 开销。不要把本地首字节（`message_start`）当成首个模型 token。

```bash
python agent/trace.py verify --run artifacts/work/local-qwen-ota
python agent/trace.py report --run artifacts/work/local-qwen-ota
```

`trajectory.json`、CLI JSONL 和评估产物用于定位失败；没有工具调用、推理被输出预算截断、模拟器环境缺失，应分别分类。配置不同导致的差异不能归结为本地模型权重能力不足。

## 8. 本次实现与验证边界

已实现 `serve-local`（预览、模型/tokenizer revision、日志排空、就绪检测、清理、环境文件）、`native --local` 配置继承、Qwen sampling/thinking 控制、协议转换与缓存统计修复。真实 Claude Code CLI 已通过本地确定性 backend 的 `Read -> tool_result -> final` 验证；该验证覆盖 CLI 工具往返，不是模型推理评测。

开发环境检测到 Claude Code 2.1.259、vLLM 0.27.1、torch 2.13.0+cu130、transformers 5.16.1，但 `torch.cuda.is_available()` 为 false、GPU 数量为 0。`vllm serve --help=all` 也因没有可识别的 device 而失败。这组版本只是检测结果，不是经过 GPU 验收的推荐组合。尚未完成真实 9B GPU 启动、128K/256K 极限测试、APC 命中/收益测量和同模型 API 对照。

Claude Code 官方目前不保证网关后的非 Claude 模型；升级 CLI 或 vLLM 后应重新执行协议和任务验收。本次没有修改 Claude Code 的全局配置、安装训练依赖或下载模型权重。[Claude Code 网关边界](https://code.claude.com/docs/en/llm-gateway)
