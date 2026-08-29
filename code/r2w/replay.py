from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import hashlib, json
import numpy as np
from .actions import Representation
from .config import R2WConfig
from .retrieval import ParentRRFIndex

class QueryRunner(Protocol):
    reader_version: str
    scorer_version: str
    prompt_version: str
    seed: int
    def score(self, query:dict, bodies:list[str])->float: ...

@dataclass(frozen=True)
class Snapshot:
    signature:str
    bodies:tuple[str,...]
    target_rank:int

@dataclass(frozen=True)
class ActionLabel:
    action:str
    delta_e:float
    delta_f:float
    se_e:float
    se_f:float
    hit_rate:float
    sampled_queries:int
    changed_queries:int

@dataclass(frozen=True)
class ReplayAudit:
    probabilities:tuple[float,...]
    indicators:tuple[int,...]
    changed:tuple[int,...]
    seed:int

@dataclass(frozen=True)
class MemoryReplay:
    labels:tuple[ActionLabel,...]
    audit:ReplayAudit


def request_signature(query:dict, snap_parents:tuple[str,...], bodies:tuple[str,...], metadata:tuple[dict,...], runner:QueryRunner)->str:
    payload={
        # QueryRunner receives the complete query object, so every query field
        # belongs to the cache key. Hashing only the question could incorrectly
        # reuse a score when, for example, the reference answer or scoring mode
        # differs under the same wording.
        "query":query,"parents":snap_parents,
        "body_hashes":[hashlib.sha256(x.encode()).hexdigest() for x in bodies],
        "metadata":metadata,"prompt":runner.prompt_version,"reader":runner.reader_version,
        "scorer":runner.scorer_version,"seed":runner.seed,
    }
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()

def _snapshot(index:ParentRRFIndex, query:dict, target_id:str, runner:QueryRunner, cfg:R2WConfig)->Snapshot:
    result=index.retrieve(str(query["question"]),cfg.retrieval_k)
    parents=tuple(str(x.parent_id) for x in result); bodies=tuple(x.body for x in result)
    rank=next((i+1 for i,x in enumerate(result) if str(x.parent_id)==str(target_id)),0)
    meta=tuple(index.representations[x.parent_id].metadata for x in result)
    sig=request_signature(query,parents,bodies,meta,runner)
    return Snapshot(sig,bodies,rank)

def _snapshots_by_action(base_index:ParentRRFIndex,target_id:str,action_reprs:dict[str,Representation|None],queries:list[dict],runner:QueryRunner,cfg:R2WConfig)->dict[str,list[Snapshot]]:
    old=base_index.remove_parent(target_id)
    if old is None:
        raise ValueError("目标父记忆不在背景索引。")
    snapshots:dict[str,list[Snapshot]]={}
    try:
        for action,representation in action_reprs.items():
            if representation is not None:
                base_index.add(representation)
            snapshots[action]=[_snapshot(base_index,query,target_id,runner,cfg) for query in queries]
            if representation is not None:
                base_index.remove_parent(target_id)
    finally:
        if target_id not in base_index.representations:
            base_index.add(old)
    return snapshots

def _normalised_weights(query_count:int,weights:list[float]|None)->list[float]:
    result=weights or ([1/query_count]*query_count if query_count else [])
    if len(result)!=query_count or (query_count and not np.isclose(sum(result),1.0)):
        raise ValueError("query weights 必须和为 1。")
    return result

def _changed_queries(snapshots:dict[str,list[Snapshot]],action_reprs:dict[str,Representation|None],query_count:int)->list[int]:
    return [
        query_index for query_index in range(query_count)
        if len({snapshots[action][query_index].signature for action in action_reprs})>1
    ]

def _sampling_probabilities(snapshots:dict[str,list[Snapshot]],action_reprs:dict[str,Representation|None],queries:list[dict],changed:list[int],target_id:str,cfg:R2WConfig)->np.ndarray:
    probabilities=np.zeros(len(queries),dtype=float)
    for query_index in changed:
        risk=max(int(snapshots[action][query_index].target_rank>0) for action in action_reprs)
        evidence=target_id in set(map(str,queries[query_index].get("evidence",[])))
        probabilities[query_index]=min(1.0,cfg.s_min*(4.0 if risk or evidence else 1.0))
        probabilities[query_index]=max(probabilities[query_index],cfg.s_min)
    return probabilities

def _sample_indicators(changed:list[int],probabilities:np.ndarray,target_id:str,cfg:R2WConfig)->np.ndarray:
    seed_offset=int(hashlib.sha256(str(target_id).encode()).hexdigest()[:8],16)%1000000
    rng=np.random.default_rng(cfg.random_seed+seed_offset)
    indicators=np.zeros(len(probabilities),dtype=int)
    for query_index in changed:
        indicators[query_index]=int(rng.random()<probabilities[query_index])
    return indicators

def _score_sampled_snapshots(snapshots:dict[str,list[Snapshot]],action_reprs:dict[str,Representation|None],queries:list[dict],changed:list[int],indicators:np.ndarray,runner:QueryRunner)->dict[str,float]:
    cache:dict[str,float]={}
    for query_index in changed:
        if not indicators[query_index]:
            continue
        for action in action_reprs:
            snapshot=snapshots[action][query_index]
            if snapshot.signature not in cache:
                cache[snapshot.signature]=float(runner.score(queries[query_index],list(snapshot.bodies)))
    return cache

def _estimate_delta(action:str,baseline:str,snapshots:dict[str,list[Snapshot]],changed:list[int],indicators:np.ndarray,cache:dict[str,float],weights:list[float],probabilities:np.ndarray)->tuple[float,float]:
    terms=[]; variances=[]
    for query_index in changed:
        if not indicators[query_index]:
            continue
        difference=cache[snapshots[action][query_index].signature]-cache[snapshots[baseline][query_index].signature]
        probability=probabilities[query_index]; weight=weights[query_index]
        terms.append(weight*difference/probability)
        variances.append(weight*weight*(1-probability)/(probability*probability)*difference*difference)
    return float(sum(terms)),float(np.sqrt(sum(variances)))

def _action_labels(action_reprs:dict[str,Representation|None],snapshots:dict[str,list[Snapshot]],query_count:int,changed:list[int],indicators:np.ndarray,cache:dict[str,float],weights:list[float],probabilities:np.ndarray)->list[ActionLabel]:
    raw_e,_=_estimate_delta("raw","none",snapshots,changed,indicators,cache,weights,probabilities)
    labels=[]
    for action in action_reprs:
        delta_e,se_e=_estimate_delta(action,"none",snapshots,changed,indicators,cache,weights,probabilities) if action!="none" else (0.0,0.0)
        delta_f,se_f=_estimate_delta(action,"raw",snapshots,changed,indicators,cache,weights,probabilities) if action not in {"none","raw"} else ((-raw_e,0.0) if action=="none" else (0.0,0.0))
        hits=[snapshots[action][query_index].target_rank>0 for query_index in range(query_count)] if action!="none" else [False]*query_count
        labels.append(ActionLabel(action,delta_e,delta_f,se_e,se_f,float(np.mean(hits)) if hits else 0.0,int(indicators.sum()),len(changed)))
    return labels

def replay_memory(
    base_index:ParentRRFIndex,
    target_id:str,
    action_reprs:dict[str,Representation|None],
    queries:list[dict],
    runner:QueryRunner,
    cfg:R2WConfig,
    weights:list[float]|None=None,
)->MemoryReplay:
    if "none" not in action_reprs or "raw" not in action_reprs:
        raise ValueError("重放必须包含 none/raw。")
    snapshots=_snapshots_by_action(base_index,target_id,action_reprs,queries,runner,cfg)
    query_count=len(queries)
    normalised_weights=_normalised_weights(query_count,weights)
    changed=_changed_queries(snapshots,action_reprs,query_count)
    probabilities=_sampling_probabilities(snapshots,action_reprs,queries,changed,target_id,cfg)
    indicators=_sample_indicators(changed,probabilities,target_id,cfg)
    cache=_score_sampled_snapshots(snapshots,action_reprs,queries,changed,indicators,runner)
    labels=_action_labels(action_reprs,snapshots,query_count,changed,indicators,cache,normalised_weights,probabilities)
    audit=ReplayAudit(tuple(probabilities.tolist()),tuple(indicators.tolist()),tuple(changed),cfg.random_seed)
    return MemoryReplay(tuple(labels),audit)
