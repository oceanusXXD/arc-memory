"""R2W-GBM 的冻结配置和工件协议。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path


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
    return R2WConfig(**overrides)


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
