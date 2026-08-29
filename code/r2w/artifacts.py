from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import json
from pathlib import Path
from .config import R2WConfig
from .replay import MemoryReplay
from .costs import ResourceLedger, SelectResources

REQUIRED_VERSION_FIELDS=frozenset({
    "candidate_constructor", "retriever", "reader", "scorer", "prompt",
    "tokenizer", "support_model", "qa_checker", "resource_pricing",
})

@dataclass(frozen=True)
class LabelRow:
    dialogue_id:str; turn_id:str; action_id:str; background_id:str; fold_id:int
    delta_e:float; delta_f:float; se_e:float; se_f:float; hit_rate:float
    sampled_queries:int; changed_queries:int; sampling_seed:int
    select_resources:dict[str,Any]; commit_resources:dict[str,Any]; index_resources:dict[str,Any]; read_resources:dict[str,Any]
    gate:dict[str,Any]; retrieval:dict[str,Any]; versions:dict[str,str]
    config_hash:str

@dataclass(frozen=True)
class SamplingAuditRow:
    dialogue_id:str; turn_id:str; background_id:str; fold_id:int
    changed_queries:tuple[int,...]; probabilities:tuple[float,...]; indicators:tuple[int,...]; seed:int


def rows_from_replay(dialogue_id:str,turn_id:str,background_id:str,fold_id:int,replay:MemoryReplay,cfg:R2WConfig,resources:dict[str,ResourceLedger]|None=None,select:SelectResources|None=None,gates:dict[str,dict]|None=None,retrieval:dict[str,dict]|None=None,versions:dict[str,str]|None=None):
    resources=resources or {}; gates=gates or {}; retrieval=retrieval or {}; versions=versions or {}
    missing=REQUIRED_VERSION_FIELDS-set(versions)
    empty=[key for key in REQUIRED_VERSION_FIELDS if not str(versions.get(key," ")).strip()]
    if missing or empty:
        raise ValueError(f"标签工件缺少冻结版本字段: {sorted(set(missing)|set(empty))}")
    rows=[]
    for x in replay.labels:
        ledger=resources.get(x.action,ResourceLedger())
        sel=select or ledger.select
        rows.append(LabelRow(dialogue_id,turn_id,x.action,background_id,fold_id,x.delta_e,x.delta_f,x.se_e,x.se_f,x.hit_rate,x.sampled_queries,x.changed_queries,replay.audit.seed,asdict(sel),asdict(ledger.commit),asdict(ledger.index),asdict(ledger.read),gates.get(x.action,{}),retrieval.get(x.action,{}),versions,cfg.hash()))
    audit=SamplingAuditRow(dialogue_id,turn_id,background_id,fold_id,replay.audit.changed,replay.audit.probabilities,replay.audit.indicators,replay.audit.seed)
    return rows,audit

def save_rows(rows:list[LabelRow],path:str|Path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf8") as f:
        for r in rows: f.write(json.dumps(asdict(r),ensure_ascii=False)+"\n")

def save_sampling_audits(rows:list[SamplingAuditRow],path:str|Path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf8") as f:
        for r in rows: f.write(json.dumps(asdict(r),ensure_ascii=False)+"\n")
