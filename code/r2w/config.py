"""R2W-GBM 的冻结配置和工件协议。"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
from dataclasses import asdict, dataclass, fields
from pathlib import Path


ALIYUN_FAILOVER_MODELS = (
    "qwen3.7-max-2026-06-08",
    "glm-5.1",
    "kimi-k3",
    "deepseek-v4-flash-0731",
    "glm-5.2",
    "kimi-k2.7-code",
    "deepseek-v4-pro-0813",
    "qwen3.7-plus-2026-05-26",
    "qwen3.8-2.4t-a95b",
    "qwen3.8-max",
    "qwen3.7-flash",
)


def _load_env_file(path: Path) -> None:
    """读取 FutureMem 同格式的简单 shell .env，不打印值。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not name or not name.replace("_", "").isalnum():
            continue
        value = value.strip()
        try:
            parsed = shlex.split(value, comments=True, posix=True)
            value = parsed[0] if parsed else ""
        except ValueError as exc:
            raise ValueError(f"环境文件 {path} 中 {name} 的引号不合法。") from exc
        # FutureMem 的 .env.rag_futuremem10 使用 "$LLM_BASE_URL"；在已加载
        # 的基础 .env 上执行 shell 风格变量展开。
        value = os.path.expandvars(value)
        os.environ[name] = value


def _load_runtime_env() -> None:
    root = Path(__file__).resolve().parents[1]
    _load_env_file(root / ".env")
    profile = os.environ.get("R2W_ENV_FILE", "").strip()
    if profile:
        profile_path = Path(profile)
        if not profile_path.is_absolute():
            profile_path = root / profile_path
        _load_env_file(profile_path)


_load_runtime_env()


def _env_int(name: str, fallback: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return fallback
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数。") from exc


def _env_float(name: str, fallback: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return fallback
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是数值。") from exc


@dataclass(slots=True)
class R2WConfig:
    """所有会改变测量、特征或决策语义的值都进入工件快照。"""

    artifact_version: int = 3
    repr_protocol_version: str = "r2w-repr-v3"
    label_schema_version: str = "r2w-label-gbm-v1"

    embedding_model: str = "qwen3.7-text-embedding"
    embedding_dim: int = 1024
    embedding_base_url: str = "https://ws-r47ew96n372cv89s.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    embedding_device: str = "cpu"
    embedding_api_batch_size: int = 8
    embedding_api_timeout: float = 600.0
    qg_model: str = "castorini/doc2query-t5-base-msmarco"
    qg_num_queries: int = 2
    qg_num_beams: int = 2
    ner_model: str = "dslim/bert-base-NER"
    llm_model: str = "kimi-k3"
    llm_base_url: str = "https://ws-r47ew96n372cv89s.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512
    llm_api_timeout: float = 600.0
    llm_reasoning_effort: str = ""
    llm_failover_models: tuple[str, ...] = ALIYUN_FAILOVER_MODELS

    # API 和本地结构化构建器都实现同一协议；本次不在训练/测试中调用真实后端。
    constructor_backend: str = "api"
    constructor_local_model: str = ""
    constructor_local_device: str = "cpu"

    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rare_idf_threshold: float = 3.0
    rrf_k: int = 60
    retrieval_k: int = 10
    teacher_rank_limit: int = 20
    key_budget_words: int = 64
    summary_ratio: float = 0.60
    kv_value_words: int = 16
    event_max_items: int = 3
    event_item_words: int = 20
    graph_max_items: int = 3
    graph_item_words: int = 20
    hq_max_items: int = 2
    hq_total_words: int = 40
    hq_answerability_overlap: float = 0.5

    interference_query_count: int = 30
    empirical_bayes_n0: float = 4.0
    inverse_variance_weights: bool = False

    gbm_folds: int = 5
    gbm_estimators: int = 200
    gbm_learning_rate: float = 0.05
    gbm_num_leaves: int = 15
    gbm_min_child_samples: int = 10

    lambda_w: float = 0.0
    lambda_r: float = 0.01
    lambda_s: float = 0.0
    kappa_e: float = 1.0
    kappa_f: float = 1.0
    fidelity_ratio: float = 0.95
    policy_lambda_scales: tuple[float, ...] = (0.5, 1.0, 2.0)
    policy_kappas: tuple[float, ...] = (0.0, 0.5, 1.0)

    case_k: int = 10
    case_similarity_floor: float = 0.3
    case_max_size: int = 200_000
    none_shadow_probability: float = 0.05
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.artifact_version != 3:
            raise ValueError("R2W-GBM 只接受 artifact_version=3。")
        if self.repr_protocol_version != "r2w-repr-v3":
            raise ValueError("R2W-GBM 只接受 repr_protocol_version=r2w-repr-v3。")
        if self.label_schema_version != "r2w-label-gbm-v1":
            raise ValueError("R2W-GBM 只接受 r2w-label-gbm-v1 标签协议。")
        if self.constructor_backend not in {"api", "local"}:
            raise ValueError("constructor_backend 只能是 api 或 local。")
        if self.gbm_folds != 5:
            raise ValueError("R2W-GBM 固定使用五折 conversation ensemble。")
        if not 0.0 < self.summary_ratio <= 1.0:
            raise ValueError("summary_ratio 必须在 (0, 1]。")
        if self.empirical_bayes_n0 < 0.0:
            raise ValueError("empirical_bayes_n0 不能为负。")
        if self.case_k < 1 or self.case_max_size < 1:
            raise ValueError("case_k 与 case_max_size 必须为正数。")
        if not -1.0 <= self.case_similarity_floor <= 1.0:
            raise ValueError("case_similarity_floor 必须在 [-1,1]。")
        if not 0.0 <= self.fidelity_ratio <= 1.0:
            raise ValueError("fidelity_ratio 必须在 [0, 1]。")

    def hash(self) -> str:
        """模型/特征/测量语义哈希；运行时位置和密钥不参与。"""
        payload = self.to_dict()
        for name in (
            "embedding_device",
            "embedding_api_batch_size",
            "embedding_api_timeout",
            "llm_api_timeout",
            "constructor_local_device",
            "lambda_w",
            "lambda_r",
            "lambda_s",
            "kappa_e",
            "kappa_f",
            "fidelity_ratio",
            "policy_lambda_scales",
            "policy_kappas",
        ):
            payload.pop(name)
        serialised = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "R2WConfig":
        expected = {field.name for field in fields(cls)}
        unknown = set(values) - expected
        missing = expected - set(values)
        if unknown or missing:
            raise ValueError(
                "工件配置不是 R2W-GBM v3："
                f"unknown={sorted(unknown)}, missing={sorted(missing)}"
            )
        return cls(**values)


def default_config(**overrides) -> R2WConfig:
    runtime = {
        "embedding_model": os.environ.get("EMBEDDING_MODEL", "qwen3.7-text-embedding"),
        "embedding_base_url": os.environ.get(
            "EMBEDDING_BASE_URL",
            "https://ws-r47ew96n372cv89s.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        ),
        "embedding_device": os.environ.get("EMBEDDING_DEVICE", "cpu"),
        "embedding_api_batch_size": _env_int("EMBEDDING_BATCH_SIZE", 8),
        "embedding_api_timeout": _env_float("EMBEDDING_API_TIMEOUT", 600.0),
        "llm_model": os.environ.get("LLM_MODEL", "kimi-k3"),
        "llm_base_url": os.environ.get(
            "LLM_BASE_URL",
            "https://ws-r47ew96n372cv89s.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        ),
        "llm_api_timeout": _env_float("LLM_API_TIMEOUT", 600.0),
        "llm_reasoning_effort": os.environ.get("LLM_REASONING_EFFORT", "").strip(),
    }
    failover = os.environ.get("LLM_FAILOVER_MODELS", "").strip()
    if failover:
        runtime["llm_failover_models"] = tuple(
            value.strip() for value in failover.split(",") if value.strip()
        )
    runtime.update(overrides)
    return R2WConfig(**runtime)


def load_config(path: str | Path) -> R2WConfig:
    with Path(path).open(encoding="utf-8") as handle:
        return R2WConfig.from_dict(json.load(handle))


def save_config(cfg: R2WConfig, path: str | Path) -> None:
    """写入可直接传给所有 CLI 的完整、严格配置快照。"""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(cfg.to_dict(), handle, ensure_ascii=False, indent=2)


def llm_runtime_config(cfg: R2WConfig) -> tuple[str, str | None, str]:
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少 LLM_API_KEY，R2W 不会伪造 API 构建结果。")
    if cfg.llm_temperature != 0.0:
        raise ValueError("结构化 API 构建要求 temperature=0。")
    model = cfg.llm_model.strip()
    if not model:
        raise ValueError("llm_model 不能为空。")
    return api_key, cfg.llm_base_url.strip() or None, model
