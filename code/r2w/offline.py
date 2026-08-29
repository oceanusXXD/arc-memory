from __future__ import annotations
from dataclasses import dataclass, replace
from .actions import TokenCounter, build_candidate_bundle, build_representation, Representation
from .config import R2WConfig
from .gates import SupportModel,QAChecker,EntityExtractor,filter_hq,gate_representation
from .replay import replay_memory,MemoryReplay,QueryRunner
from .retrieval import ParentRRFIndex

@dataclass(frozen=True)
class PreparedMemory:
    turn_id:str
    representations:dict[str,Representation|None]
    gate_info:dict[str,dict]
    degenerate_to:dict[str,str]

class OfflineLabelProducer:
    def __init__(self,cfg:R2WConfig,embedder,counter:TokenCounter,constructor,support:SupportModel,qa_checker:QAChecker,runner:QueryRunner,entity_extractor:EntityExtractor|None=None):
        self.cfg=cfg; self.embedder=embedder; self.counter=counter; self.constructor=constructor; self.support=support; self.qa=qa_checker; self.runner=runner; self.entities=entity_extractor

    def _raw_state(self,turn:dict):
        raw=build_representation(turn,"raw",None,self.cfg,self.counter)
        reps={"none":None,"raw":raw}; gate={"raw":{"fid_mean":1.0,"fid_min":1.0}}; deg={}
        return raw,reps,gate,deg

    def _candidate_bundle(self,turn:dict,history_prefix:list[dict],gate:dict):
        try:
            return filter_hq(build_candidate_bundle(turn,history_prefix,self.constructor,self.cfg),str(turn["text"]),self.qa)
        except Exception as exc:
            # A failed common candidate path must not remove the raw/none
            # comparison or make a memory disappear from label production.
            gate["candidate_bundle"]={"feasible":False,"reason":f"unavailable:{type(exc).__name__}"}
            return None

    def _add_generated_representations(self,turn:dict,bundle,raw:Representation,reps:dict,gate:dict,deg:dict)->None:
        seen={(raw.body,raw.keys):"raw"}
        for action in self.cfg.storage_actions:
            if action=="raw": continue
            try:
                r=build_representation(turn,action,bundle,self.cfg,self.counter)
                g=gate_representation(r,str(turn["text"]),self.support,self.cfg,self.entities)
                if not g.feasible: continue
                builder_version=str(getattr(self.constructor,"version",type(self.constructor).__name__))
                r=replace(r,metadata={**r.metadata,"fid_mean":g.fid_mean,"fid_min":g.fid_min,"builder_version":builder_version})
                k=(r.body,r.keys)
                if k in seen: deg[action]=seen[k]; continue
                seen[k]=action; reps[action]=r; gate[action]={"fid_mean":g.fid_mean,"fid_min":g.fid_min}
            except Exception:
                continue

    def prepare(self,turn:dict,history_prefix:list[dict])->PreparedMemory:
        raw,reps,gate,deg=self._raw_state(turn)
        bundle=self._candidate_bundle(turn,history_prefix,gate)
        if bundle is not None:
            self._add_generated_representations(turn,bundle,raw,reps,gate,deg)
        return PreparedMemory(str(turn["dia_id"]),reps,gate,deg)

    def prepare_conversation(self,conversation:dict,history_window:int=8)->list[PreparedMemory]:
        turns=conversation["turns"]
        return [self.prepare(t,turns[max(0,i-history_window):i]) for i,t in enumerate(turns)]

    def _background_representations(self,prepared:list[PreparedMemory],background_actions:list[str])->list[Representation]:
        reps=[]
        for i,a in enumerate(background_actions):
            if a=="none": continue
            r=prepared[i].representations.get(a)
            if r is None: raise ValueError(f"背景动作 {a} 对 turn {i} 不可行。")
            reps.append(r)
        return reps

    def _replay_prepared(self,index:ParentRRFIndex,prepared:PreparedMemory,queries:list[dict])->MemoryReplay:
        inserted=prepared.turn_id not in index.representations
        if inserted:
            index.add(prepared.representations["raw"])
        try:
            return replay_memory(index,prepared.turn_id,prepared.representations,queries,self.runner,self.cfg)
        finally:
            if inserted:
                index.remove_parent(prepared.turn_id)

    def replay_conversation(self,conversation:dict,prepared:list[PreparedMemory],background_actions:list[str]|None=None,background_id:str="raw")->dict[str,MemoryReplay]:
        turns=conversation["turns"]
        if len(prepared)!=len(turns): raise ValueError("prepared 必须逐 turn 对齐。")
        background_actions=background_actions or ["raw"]*len(turns)
        if len(background_actions)!=len(turns): raise ValueError("background_actions 长度不匹配。")
        index=ParentRRFIndex(self._background_representations(prepared,background_actions),self.embedder,self.cfg)
        out={}
        for item in prepared:
            out[item.turn_id]=self._replay_prepared(index,item,conversation["qa"])
        return out

def sample_mixed_background(prepared:list[PreparedMemory], action_frequencies:dict[str,float], seed:int=0)->list[str]:
    import numpy as np
    actions=list(action_frequencies); probs=np.asarray([action_frequencies[a] for a in actions],dtype=float)
    if not actions or np.any(probs<0) or probs.sum()<=0: raise ValueError("action_frequencies 非法。")
    probs=probs/probs.sum(); rng=np.random.default_rng(seed); out=[]
    for p in prepared:
        feasible=[a for a in actions if a=="none" or a in p.representations]
        fp=np.asarray([action_frequencies[a] for a in feasible],dtype=float); fp=fp/fp.sum()
        out.append(str(rng.choice(feasible,p=fp)))
    return out

def policy_background(prepared:list[PreparedMemory], policy)->list[str]:
    out=[]
    for p in prepared:
        a=str(policy(p))
        if a!="none" and a not in p.representations: raise ValueError(f"策略选择不可行动作 {a}。")
        out.append(a)
    return out

def needs_background_retrain(action_agreement:float,kendall_tau:float,min_agreement:float,min_tau:float)->bool:
    return action_agreement<min_agreement or kendall_tau<min_tau
