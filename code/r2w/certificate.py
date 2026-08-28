"""抽取类表示的可复测事实证书与人审权重拟合。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import json

import numpy as np
import torch

from .llm import LLMClient, LLMProtocolError
from .llm_prompts import FACT_SPLIT
from .representations import CERTIFICATE_ACTIONS, Repr


class NLIScorer(Protocol):
    def score(self, premise: str, hypothesis: str) -> float: ...


class TransformersNLIScorer:
    """冻结 Hugging Face NLI 分类器；模型名由部署方显式提供。"""

    def __init__(self, model_name: str, device: str = "cpu"):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("启用事实证书时必须提供非空 NLI 模型名。")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device).eval()
        self.device = torch.device(device)
        labels = {
            int(index): str(label).casefold()
            for index, label in self.model.config.id2label.items()
        }
        matches = [index for index, label in labels.items() if "entail" in label]
        if len(matches) != 1:
            raise RuntimeError(
                "NLI 模型 id2label 必须恰好包含一个 entailment 标签，"
                f"实际为 {labels}。"
            )
        self.entailment_index = matches[0]

    def score(self, premise: str, hypothesis: str) -> float:
        encoded = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        with torch.no_grad():
            probabilities = torch.softmax(self.model(**encoded).logits[0], dim=-1)
        return float(probabilities[self.entailment_index].item())


@dataclass(frozen=True)
class CertificateComponents:
    cov_f: float
    contra: float
    margin: float

    def features(self) -> np.ndarray:
        return np.asarray((self.cov_f, 1.0 - self.contra, self.margin), dtype=np.float64)


def _probability(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} 必须是 [0,1] 数值。")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须是 [0,1] 数值。") from exc
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} 必须是 [0,1] 数值。")
    return number


def measure_certificate(
    payload: str,
    raw_window: str,
    nli: NLIScorer,
    llm: LLMClient,
) -> CertificateComponents:
    """将真实生成 payload 拆成原子事实，再逐条对 raw 来源做 NLI。"""
    response = llm.json(FACT_SPLIT.format(payload=payload), purpose="certificate")
    facts = response.get("facts")
    if not isinstance(facts, list) or any(
        not isinstance(fact, str) or not fact.strip() for fact in facts
    ):
        raise LLMProtocolError("事实拆分响应必须包含非空字符串组成的 facts 数组。")
    facts = list(dict.fromkeys(fact.strip() for fact in facts))
    if not facts:
        return CertificateComponents(0.0, 1.0, 0.0)
    scores = np.asarray(
        [_probability(nli.score(raw_window, fact), "NLI entailment score") for fact in facts],
        dtype=np.float64,
    )
    return CertificateComponents(
        cov_f=float(np.mean(scores >= 0.5)),
        contra=float(np.mean(scores < 0.2)),
        margin=float(np.mean(np.abs(scores - 0.5))),
    )


def fit_certificate_weights(
    components: np.ndarray, human_error: np.ndarray
) -> tuple[float, float, float]:
    """在人审子集上拟合方向受限的线性分数，并归一化为凸组合。"""
    values = np.asarray(components, dtype=np.float64)
    errors = np.asarray(human_error, dtype=bool)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) != len(errors):
        raise ValueError("证书特征必须是与人审标签等长的 [n,3] 数组。")
    if (
        not len(values)
        or not np.all(np.isfinite(values))
        or np.any((values < 0) | (values > 1))
        or np.any(values[:, 2] > 0.5)
    ):
        raise ValueError("证书特征必须非空；前两列在 [0,1]，margin 在 [0,0.5]。")
    if np.unique(errors).size < 2:
        raise ValueError("证书权重拟合需要同时包含正确与错误的人审样本。")
    target = (~errors).astype(np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    ridge = centered.T @ centered + np.eye(3) * 1e-6
    weights = np.maximum(np.linalg.solve(ridge, centered.T @ (target - target.mean())), 0.0)
    if float(weights.sum()) <= 1e-12:
        weights = np.ones(3, dtype=np.float64)
    weights /= weights.sum()
    return tuple(float(value) for value in weights)


def score_certificate_components(
    components: np.ndarray, weights: tuple[float, float, float]
) -> np.ndarray:
    values = np.asarray(components, dtype=np.float64)
    fitted = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("证书特征必须是 [n,3] 数组。")
    if fitted.shape != (3,) or np.any(fitted < 0) or not np.isclose(fitted.sum(), 1.0):
        raise ValueError("证书权重必须是和为 1 的非负三维向量。")
    return values @ fitted


def load_certificate_reviews(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """读取人审 JSON；字段严格对应开发文档 certs 表的真实列。"""
    with Path(path).open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError("证书人审文件必须是非空 JSON 数组。")
    components, errors = [], []
    provenance = {"conversation_id": [], "dia_id": [], "arm": []}
    required = {
        "conversation_id",
        "dia_id",
        "arm",
        "cov_f",
        "contra",
        "margin",
        "human_error",
    }
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or required - set(row):
            raise ValueError(f"证书人审第 {index} 行缺少字段 {sorted(required)}。")
        unexpected = set(row) - required
        if unexpected:
            raise ValueError(f"证书人审第 {index} 行包含未知字段 {sorted(unexpected)}。")
        margin = _probability(row["margin"], "margin")
        if margin > 0.5:
            raise ValueError(f"证书人审第 {index} 行 margin 必须在 [0,0.5]。")
        components.append(
            [
                _probability(row["cov_f"], "cov_f"),
                1.0 - _probability(row["contra"], "contra"),
                margin,
            ]
        )
        error = row["human_error"]
        if not isinstance(error, bool):
            raise ValueError(f"证书人审第 {index} 行 human_error 必须是 boolean。")
        errors.append(error)
        for field in ("conversation_id", "dia_id", "arm"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"证书人审第 {index} 行 {field} 必须是非空字符串。")
        if row["arm"] not in CERTIFICATE_ACTIONS:
            raise ValueError(f"证书人审第 {index} 行 arm 不是需要证书的表示动作。")
        for field in provenance:
            provenance[field].append(row[field].strip())
    return (
        np.asarray(components, dtype=np.float64),
        np.asarray(errors, dtype=bool),
        {field: np.asarray(values) for field, values in provenance.items()},
    )


class CertificateChecker:
    """部署执行层 checker；使用校准时冻结的同一组分权重。"""

    def __init__(
        self,
        nli: NLIScorer,
        llm: LLMClient,
        weights: tuple[float, float, float],
    ):
        self.nli, self.llm, self.weights = nli, llm, weights

    def __call__(self, representation: Repr, turn: dict) -> float:
        raw_text = str(turn.get("source_text") or turn.get("text") or "").strip()
        if not raw_text:
            raise ValueError("事实证书需要 turn.source_text 或 turn.text。")
        source_parts = [
            f"speaker={turn.get('speaker', '')}",
            f"time={turn.get('timestamp', '')}",
            raw_text,
        ]
        raw_window = " | ".join(source_parts)
        material = []
        if representation.arm.startswith("sum"):
            material.append(representation.payload)
        if not representation.arm.endswith("+hq"):
            material.extend(representation.keys)
        payload = "\n".join(material).strip()
        if not payload:
            return 0.0
        components = measure_certificate(
            payload, raw_window, self.nli, self.llm
        )
        return float(
            score_certificate_components(components.features()[None, :], self.weights)[0]
        )
