"""R2W-GBM 的普通 validation 决策配置；不包含 CRC。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import R2WConfig


@dataclass(frozen=True)
class CostStatistics:
    mean_write: float
    mean_hit_rate: float
    index_per_context_word: float

    @classmethod
    def fit(cls, costs: np.ndarray, hit_rates: np.ndarray) -> "CostStatistics":
        costs = np.asarray(costs, dtype=np.float32)
        hits = np.asarray(hit_rates, dtype=np.float32)
        if costs.ndim != 3 or costs.shape[2] != 3:
            raise ValueError("成本必须是 [turn, arm, write/index/context]。")
        nonraw = costs[:, 1:, :]
        context = np.maximum(nonraw[:, :, 2], 1e-6)
        return cls(
            mean_write=float(nonraw[:, :, 0].mean()),
            mean_hit_rate=float(hits[:, 1:].mean()),
            index_per_context_word=float(np.mean(nonraw[:, :, 1] / context)),
        )


@dataclass(frozen=True)
class DecisionPolicy:
    lambda_w: float
    lambda_r: float
    lambda_s: float
    kappa_e: float
    kappa_f: float
    fidelity_ratio: float
    cost_statistics: CostStatistics
    config_hash: str

    def bar_cost(self, raw_cost: np.ndarray) -> float:
        raw = np.asarray(raw_cost, dtype=np.float32)
        if raw.shape != (3,):
            raise ValueError("raw_cost 必须是 write/index/context 三元组。")
        return float(
            self.lambda_w * self.cost_statistics.mean_write
            + self.lambda_r * raw[2] * self.cost_statistics.mean_hit_rate
            + self.lambda_s * raw[2] * self.cost_statistics.index_per_context_word
        )

    def values(self, effects: np.ndarray, hit_rates: np.ndarray, sigma: np.ndarray, costs: np.ndarray) -> np.ndarray:
        effects = np.asarray(effects, dtype=np.float32)
        hit_rates = np.asarray(hit_rates, dtype=np.float32)
        sigma = np.asarray(sigma, dtype=np.float32)
        costs = np.asarray(costs, dtype=np.float32)
        if costs.shape != (len(effects), 3):
            raise ValueError("逐臂成本必须是 [arm, 3]。")
        return (
            effects
            - self.lambda_w * costs[:, 0]
            - self.lambda_r * costs[:, 2] * hit_rates
            - self.lambda_s * costs[:, 1]
            - self.kappa_f * sigma
        ).astype(np.float32)

    def gate(self, y_b: float, sigma_b: float, raw_cost: np.ndarray) -> float:
        """第一关的确定性准入分数。

        这里只读取 raw 的可直接计数成本；调用方必须在构造任何非 raw
        草稿前执行它，才能让 ``none`` 真正节省构建工作。
        """
        return float(y_b - self.bar_cost(raw_cost) + self.kappa_e * sigma_b)

    def decide(self, y_b: float, sigma_b: float, effects: np.ndarray, hit_rates: np.ndarray, sigma: np.ndarray, costs: np.ndarray, feasible: np.ndarray | None = None) -> "PolicyDecision":
        gate_value = self.gate(y_b, sigma_b, np.asarray(costs)[0])
        if gate_value <= 0.0:
            return PolicyDecision("none", gate_value, np.zeros(len(effects), dtype=np.float32))
        values = self.values(effects, hit_rates, sigma, costs)
        allowed = np.ones(len(values), dtype=bool) if feasible is None else np.asarray(feasible, dtype=bool)
        if allowed.shape != values.shape:
            raise ValueError("feasible 必须为逐臂布尔向量。")
        if not allowed[0]:
            raise ValueError("raw 必须始终在可行集中。")
        values = values.copy()
        values[~allowed] = -np.inf
        return PolicyDecision(int(np.argmax(values)), gate_value, values)


@dataclass(frozen=True)
class PolicyDecision:
    action_index: int | str
    gate_value: float
    values: np.ndarray


def _oracle(effects: np.ndarray, hits: np.ndarray, costs: np.ndarray, policy: DecisionPolicy) -> int | str:
    values = policy.values(effects, hits, np.zeros_like(effects), costs)
    best = int(np.argmax(values))
    return "none" if values[best] <= 0.0 else best


def select_policy(
    cfg: R2WConfig,
    cost_statistics: CostStatistics,
    admission: np.ndarray,
    admission_sigma: np.ndarray,
    predicted_effects: np.ndarray,
    predicted_hit_rates: np.ndarray,
    effect_sigma: np.ndarray,
    costs: np.ndarray,
    observed_effects: np.ndarray,
    observed_hit_rates: np.ndarray,
    measured: np.ndarray,
) -> DecisionPolicy:
    """在 validation split 上按观测净价值遗憾选择 λ/κ。"""
    predicted_effects = np.asarray(predicted_effects, dtype=np.float32)
    predicted_hit_rates = np.asarray(predicted_hit_rates, dtype=np.float32)
    observed_effects = np.asarray(observed_effects, dtype=np.float32)
    observed_hit_rates = np.asarray(observed_hit_rates, dtype=np.float32)
    measured = np.asarray(measured, dtype=bool)
    if (
        predicted_effects.shape != predicted_hit_rates.shape
        or predicted_effects.shape != observed_effects.shape
        or predicted_effects.shape != observed_hit_rates.shape
        or predicted_effects.shape != measured.shape
        or costs.shape[:2] != predicted_effects.shape
    ):
        raise ValueError("验证预测、观测标签、measured mask 与成本必须逐臂对齐。")
    if not measured.all():
        raise ValueError("validation policy 选择需要所有候选臂具备观测标签。")
    best: tuple[float, float, DecisionPolicy] | None = None
    for scale in cfg.policy_lambda_scales:
        for kappa in cfg.policy_kappas:
            policy = DecisionPolicy(
                cfg.lambda_w * scale,
                cfg.lambda_r * scale,
                cfg.lambda_s * scale,
                kappa,
                kappa,
                cfg.fidelity_ratio,
                cost_statistics,
                cfg.hash(),
            )
            regrets, selected_costs = [], []
            for row in range(len(predicted_effects)):
                predicted = policy.decide(
                    admission[row],
                    admission_sigma[row],
                    predicted_effects[row],
                    predicted_hit_rates[row],
                    effect_sigma[row],
                    costs[row],
                )
                oracle = _oracle(
                    observed_effects[row], observed_hit_rates[row], costs[row], policy
                )
                observed_values = policy.values(
                    observed_effects[row],
                    observed_hit_rates[row],
                    np.zeros_like(observed_effects[row]),
                    costs[row],
                )
                oracle_value = 0.0 if oracle == "none" else float(observed_values[oracle])
                chosen_value = (
                    0.0
                    if predicted.action_index == "none"
                    else float(observed_values[predicted.action_index])
                )
                regrets.append(oracle_value - chosen_value)
                if predicted.action_index != "none":
                    selected_costs.append(float(costs[row, predicted.action_index].sum()))
            candidate = (float(np.mean(regrets)), float(np.mean(selected_costs) if selected_costs else 0.0), policy)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
    assert best is not None
    return best[2]


def save_policy(policy: DecisionPolicy) -> dict:
    value = asdict(policy)
    value["cost_statistics"] = asdict(policy.cost_statistics)
    return value


def load_policy(value: dict, cfg: R2WConfig) -> DecisionPolicy:
    if value.get("config_hash") != cfg.hash():
        raise RuntimeError("决策配置与 R2W-GBM 工件配置不一致。")
    return DecisionPolicy(
        lambda_w=float(value["lambda_w"]),
        lambda_r=float(value["lambda_r"]),
        lambda_s=float(value["lambda_s"]),
        kappa_e=float(value["kappa_e"]),
        kappa_f=float(value["kappa_f"]),
        fidelity_ratio=float(value["fidelity_ratio"]),
        cost_statistics=CostStatistics(**value["cost_statistics"]),
        config_hash=str(value["config_hash"]),
    )
