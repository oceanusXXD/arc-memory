from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib, json

CORE_ACTIONS = ("none", "raw", "sum", "raw+kv", "raw+event", "raw+hq")
EXTENDED_ACTIONS = (
    "none", "raw", "sum", "raw+kv", "raw+event", "raw+hq",
    "raw+graph", "sum+kv", "sum+event", "sum+graph", "sum+hq",
)


def _validate_protocol_versions(cfg: "R2WConfig") -> None:
    if cfg.artifact_version != 4 or cfg.repr_protocol_version != "r2w-repr-v4":
        raise ValueError("只接受 R2W v4 当前算法协议。")
    if cfg.label_schema_version != "r2w-label-v4":
        raise ValueError("只接受 r2w-label-v4 标签协议。")


def _validate_retrieval_and_budgets(cfg: "R2WConfig") -> None:
    if cfg.retrieval_k < 1:
        raise ValueError("retrieval_k 必须 >= 1。")
    if not 0 < cfg.s_min <= 1:
        raise ValueError("s_min 必须在 (0,1]。")
    if not 0 < cfg.summary_ratio <= 1 or cfg.summary_max_tokens < 1:
        raise ValueError("摘要长度参数非法。")
    if cfg.key_budget_tokens < 0 or cfg.H_Q <= 0:
        raise ValueError("key_budget_tokens/H_Q 非法。")


def _validate_thresholds_and_prices(cfg: "R2WConfig") -> None:
    if not 0 <= cfg.tau_min <= cfg.tau_mean <= 1:
        raise ValueError("支持阈值必须满足 0<=tau_min<=tau_mean<=1。")
    if min(cfg.lambda_select, cfg.lambda_commit, cfg.lambda_index, cfg.lambda_read) < 0:
        raise ValueError("资源价格不能为负。")


def _validate_scalars(cfg: "R2WConfig") -> None:
    allowed = {
        "select_scalar": {"calls", "input_tokens", "output_tokens", "gpu_ms", "wall_ms", "cost"},
        "commit_scalar": {"calls", "gpu_ms", "wall_ms", "cost"},
        "index_scalar": {"byte_day", "bytes", "vector_count", "vector_bytes"},
        "read_scalar": {"hit_probability", "expected_body_tokens", "retrieval_ms", "reader_ms", "cost"},
    }
    for field, values in allowed.items():
        if getattr(cfg, field) not in values:
            raise ValueError(f"{field} 非法。")


def _validate_risk_and_fallback(cfg: "R2WConfig") -> None:
    if min(cfg.kappa_e, cfg.kappa_f) < 0:
        raise ValueError("风险系数不能为负。")
    if cfg.value_model_fallback not in {"none", "raw"}:
        raise ValueError("value_model_fallback 只能是 none 或 raw。")

@dataclass(frozen=True)
class R2WConfig:
    artifact_version: int = 4
    repr_protocol_version: str = "r2w-repr-v4"
    label_schema_version: str = "r2w-label-v4"
    use_extended_actions: bool = False
    retrieval_k: int = 10
    rrf_k: int = 60
    s_min: float = 0.10
    summary_ratio: float = 0.60
    summary_max_tokens: int = 128
    key_budget_tokens: int = 96
    kv_max_items: int = 4
    event_max_items: int = 3
    hq_max_items: int = 3
    hq_item_max_tokens: int = 32
    graph_max_items: int = 4
    tau_mean: float = 0.90
    tau_min: float = 0.60
    lambda_select: float = 0.0
    lambda_commit: float = 0.0
    lambda_index: float = 0.0
    lambda_read: float = 0.0
    select_scalar: str = "cost"
    commit_scalar: str = "cost"
    index_scalar: str = "byte_day"
    read_scalar: str = "expected_body_tokens"
    H_Q: float = 1000.0
    kappa_e: float = 1.0
    kappa_f: float = 1.0
    value_model_fallback: str = "raw"
    random_seed: int = 0

    def __post_init__(self):
        _validate_protocol_versions(self)
        _validate_retrieval_and_budgets(self)
        _validate_thresholds_and_prices(self)
        _validate_scalars(self)
        _validate_risk_and_fallback(self)

    @property
    def actions(self) -> tuple[str, ...]:
        return EXTENDED_ACTIONS if self.use_extended_actions else CORE_ACTIONS

    @property
    def storage_actions(self) -> tuple[str, ...]:
        return tuple(a for a in self.actions if a != "none")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict):
        return cls(**value)

    def hash(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(raw).hexdigest()[:16]
