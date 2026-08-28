"""R2W-GBM 写入侧两阶段决策入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Protocol

import joblib
import numpy as np

from .calibration import DecisionPolicy, load_policy
from .cases import CaseMemory
from .certificate import CertificateChecker, TransformersNLIScorer
from .config import load_config
from .constructor import build_constructor
from .costs import representation_costs
from .embedding import EmbeddingBackend
from .features import FeatureBuilder
from .llm import LLMClient
from .model import GBMEnsemble
from .query_generator import QueryGenerator
from .representations import CERTIFICATE_ACTIONS, Repr, STRUCT_ACTIONS, build_representation
from .text import normalize_relative_time, word_count


class MemoryIndexWriter(Protocol):
    def write(self, action: str, units: list[str], turn: dict) -> None: ...


def load_artifacts(directory: str | Path, *, embedder=None, ner=None):
    directory = Path(directory)
    cfg = load_config(directory / "config.json")
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("R2W-GBM artifact 缺少 manifest.json。")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("artifact_version") != 3 or manifest.get("config_hash") != cfg.hash() or manifest.get("model") != "model.joblib":
        raise RuntimeError("旧工件或不匹配的 artifact 被拒绝加载。")
    if (directory / "model.pt").exists() or (directory / "calibration.json").exists():
        raise RuntimeError("检测到旧 Torch/CRC 工件，R2W-GBM 不提供兼容加载。")
    embedder = embedder or EmbeddingBackend(cfg)
    features = FeatureBuilder.load(directory / "features.joblib", cfg, embedder, ner=ner)
    model = joblib.load(directory / "model.joblib")
    if not isinstance(model, GBMEnsemble):
        raise RuntimeError("model.joblib 不是 R2W GBM ensemble。")
    with (directory / "case_memory.json").open(encoding="utf-8") as handle:
        cases_payload = json.load(handle)
    if cases_payload.get("config_hash") != cfg.hash():
        raise RuntimeError("案例库与 GBM 工件配置不一致。")
    with (directory / "policy.json").open(encoding="utf-8") as handle:
        policy = load_policy(json.load(handle), cfg)
    return cfg, embedder, features, model, CaseMemory.from_list(cfg, cases_payload["cases"]), policy


def _conversation(turn: dict, context: list[dict], speaker_a: str, speaker_b: str) -> dict:
    turns = [dict(value) for value in context] + [dict(turn)]
    for value in turns:
        required = {"dia_id", "speaker", "text", "timestamp", "session"}
        if required - set(value) or any(not isinstance(value[name], str) or not value[name].strip() for name in required - {"session"}) or not isinstance(value["session"], int) or value["session"] < 1:
            raise ValueError("context 与 turn 必须包含有效 dia_id/speaker/text/timestamp/session。")
        value["source_text"] = value.get("source_text", value["text"])
        value["text"] = normalize_relative_time(value["source_text"], value["timestamp"])
    mapping = {value["dia_id"]: index for index, value in enumerate(turns)}
    if len(mapping) != len(turns):
        raise ValueError("context 与 turn 的 dia_id 不可重复。")
    if not speaker_a or not speaker_b or speaker_a == speaker_b:
        raise ValueError("speaker_a 与 speaker_b 必须是两个不同的非空字符串。")
    return {"turns": turns, "qa": [], "speaker_a": speaker_a, "speaker_b": speaker_b, "dia_to_index": mapping}


def _draft_statistics(turn: dict, representations: list[Repr]) -> np.ndarray:
    source_words = max(word_count(turn["text"]), 1)
    rows = []
    for action, representation in zip(STRUCT_ACTIONS, representations, strict=True):
        payload_words = word_count(representation.payload)
        key_words = sum(word_count(value) for value in representation.keys)
        rows.append((payload_words, key_words, len(representation.keys), payload_words / source_words, float(action.startswith("sum"))))
    return np.asarray(rows, dtype=np.float32)[None, :, :]


def _costs(representations: list[Repr]) -> np.ndarray:
    if tuple(representation.arm for representation in representations) != STRUCT_ACTIONS:
        raise ValueError("成本计算需要按固定十臂顺序提供表示。")
    return np.vstack([representation_costs(value) for value in representations])


def decide_turn(cfg, features: FeatureBuilder, model: GBMEnsemble, cases: CaseMemory, calibration: DecisionPolicy, turn: dict, context: list[dict] | None = None, speaker_a: str = "", speaker_b: str = "", qg: QueryGenerator | None = None, llm: LLMClient | None = None, writer: MemoryIndexWriter | None = None, certificate_checker: Callable[[Repr, dict], float] | None = None, constructor=None) -> dict:
    """保持旧入口参数顺序；``calibration`` 现在是普通 validation policy。"""
    conversation = _conversation(turn, context or [], speaker_a, speaker_b)
    stage1 = features.build_memory(conversation)[-1:]
    y_b, sigma_b = model.predict_admission(stage1)
    case_audit = cases.local_audit(stage1[0])
    current = conversation["turns"][-1]
    # 第一关只能使用确定性的 raw 成本。此处绝不能预先构建九个 non-raw
    # 草稿，否则 ``none`` 判决没有任何计算节省。
    raw = build_representation(current, "raw", None, cfg)
    gate_value = calibration.gate(float(y_b[0]), float(sigma_b[0]), representation_costs(raw))
    if gate_value <= 0.0:
        result = {
            "action": "none",
            "stage2_skipped": True,
            "gate_value": gate_value,
            "admission": {"mean": float(y_b[0]), "sigma": float(sigma_b[0])},
            "effects": [],
            "hit_rates": [],
            "sigma": [],
            "costs": {"raw": representation_costs(raw).astype(float).tolist()},
            "scores": [],
            "dropped": {"stage2": "admission_gate"},
            "certificates": {},
            "case_audit": case_audit,
            "units": [],
        }
        if writer is not None:
            if hasattr(writer, "write_decision"):
                writer.write_decision("none", None, current, result)
            else:
                writer.write("none", [], current)
        return result

    qg = qg or QueryGenerator(cfg)
    constructor = constructor or build_constructor(cfg, llm)
    if certificate_checker is None:
        raise RuntimeError(
            "R2W-GBM 第二关要求事实证书 checker；请配置固定 NLI checker，"
            "不能静默跳过受证书约束的候选臂。"
        )
    inputs = features.build_current_stage2(conversation, qg)
    base = np.concatenate((inputs["memory"], inputs["query"], inputs["interaction"], inputs["hot"]), axis=1)
    # 用户要求全量构建：raw 与九个非 raw 草稿全部进入第二关前的精确计数。
    representations = [raw, *[build_representation(current, action, constructor, cfg) for action in STRUCT_ACTIONS[1:]]]
    draft = _draft_statistics(current, representations)
    effects, hits, sigma = model.predict_arms(base, draft)
    costs = _costs(representations)
    feasible = np.ones(len(STRUCT_ACTIONS), dtype=bool)
    dropped: dict[str, str] = {}
    certificates: dict[str, float] = {}
    for index, (action, representation) in enumerate(zip(STRUCT_ACTIONS, representations, strict=True)):
        if action in CERTIFICATE_ACTIONS:
            score = float(certificate_checker(representation, current))
            if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                raise RuntimeError("certificate_checker 必须返回 [0,1] 内的有限分数。")
            certificates[action] = score
            if score < calibration.fidelity_ratio:
                feasible[index] = False
                dropped[action] = "certificate_below_threshold"
    decision = calibration.decide(float(y_b[0]), float(sigma_b[0]), effects[0], hits[0], sigma[0], costs, feasible)
    action = decision.action_index
    action_name = "none" if action == "none" else STRUCT_ACTIONS[int(action)]
    selected = None if action == "none" else representations[int(action)]
    result = {
        "action": action_name,
        "gate_value": decision.gate_value,
        "admission": {"mean": float(y_b[0]), "sigma": float(sigma_b[0])},
        "effects": effects[0].astype(float).tolist(),
        "hit_rates": hits[0].astype(float).tolist(),
        "sigma": sigma[0].astype(float).tolist(),
        "costs": costs.astype(float).tolist(),
        "scores": decision.values.astype(float).tolist(),
        "dropped": dropped,
        "certificates": certificates,
        "case_audit": case_audit,
        "units": [] if selected is None else list(selected.units),
    }
    if selected is not None:
        result["representation"] = {
            "protocol_version": selected.protocol_version,
            "payload": selected.payload,
            "keys": list(selected.keys),
            "key_meta": list(selected.key_meta),
        }
    if writer is not None:
        if hasattr(writer, "write_decision"):
            writer.write_decision(action_name, selected, current, result)
        elif selected is not None:
            writer.write(action_name, result["units"], current)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 R2W-GBM 写时决策")
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--turn", required=True)
    parser.add_argument("--context", default="[]")
    parser.add_argument("--speaker-a", required=True)
    parser.add_argument("--speaker-b", required=True)
    parser.add_argument("--certificate-nli-model", required=True)
    parser.add_argument("--certificate-nli-device", default="cpu")
    parser.add_argument(
        "--certificate-weights",
        default="0.3333333333,0.3333333333,0.3333333334",
        help="固定 cov_f、1-contra、margin 权重，三个非负数且和为 1。",
    )
    arguments = parser.parse_args()
    cfg, _, features, model, cases, policy = load_artifacts(arguments.artifacts)
    try:
        weights = tuple(float(value) for value in arguments.certificate_weights.split(","))
    except ValueError as exc:
        parser.error("--certificate-weights 必须是三个逗号分隔数值。")
        raise AssertionError from exc
    if len(weights) != 3 or any(value < 0.0 for value in weights) or not np.isclose(sum(weights), 1.0):
        parser.error("--certificate-weights 必须是和为 1 的三个非负数。")
    llm = LLMClient(cfg)
    checker = CertificateChecker(
        TransformersNLIScorer(arguments.certificate_nli_model, arguments.certificate_nli_device),
        llm,
        weights,
    )
    print(json.dumps(decide_turn(cfg, features, model, cases, policy, json.loads(arguments.turn), json.loads(arguments.context), arguments.speaker_a, arguments.speaker_b, llm=llm, certificate_checker=checker), ensure_ascii=False))


if __name__ == "__main__":
    main()
