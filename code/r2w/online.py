from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Protocol
from .actions import TokenCounter, build_candidate_bundle, build_representation
from .config import R2WConfig
from .costs import ResourceLedger, SelectResources, post_selection_value
from .features import CandidateDiagnostics, cheap_features, full_action_features
from .gates import SupportModel, QAChecker, EntityExtractor, filter_hq, gate_representation
from .policy import Estimate, Stage2Estimate, early_reject, safe_stage2
from .retrieval import ParentRRFIndex

class ValueModel(Protocol):
    def cheap(self, action:str, features)->Estimate: ...
    def full(self, action:str, features, raw_features)->Stage2Estimate: ...
    def read_cost(self, action:str, features)->float: ...
class ParentWriter(Protocol):
    def atomic_write(self, representation)->None: ...

class ResourceMeasurer(Protocol):
    def select_resources(self)->SelectResources: ...
    def action_resources(self, action, representation)->ResourceLedger: ...

@dataclass(frozen=True)
class Decision:
    action:str
    representation:object|None
    stage1_rejected:bool
    feasible:tuple[str,...]
    predictions:dict
    select_resources:SelectResources

class OnlineWriter:
    def __init__(self,cfg:R2WConfig,embedder,counter:TokenCounter,constructor,support:SupportModel,qa:QAChecker,value_model:ValueModel,resource_measurer:ResourceMeasurer,entity_extractor:EntityExtractor|None=None):
        self.cfg=cfg; self.embedder=embedder; self.counter=counter; self.constructor=constructor; self.support=support; self.qa=qa; self.value_model=value_model; self.resources=resource_measurer; self.entities=entity_extractor

    def _stage1_predictions(self, cheap)->dict:
        return {action:self.value_model.cheap(action,cheap) for action in self.cfg.storage_actions}

    def _raw_arms(self, turn:dict)->tuple[dict,dict,set[str]]:
        raw=build_representation(turn,"raw",None,self.cfg,self.counter)
        return {"raw":raw},{"raw":CandidateDiagnostics()},{"none","raw"}

    def _candidate_bundle(self, turn:dict, history_prefix:list[dict], fallbacks:list[str]):
        try:
            bundle=build_candidate_bundle(turn,history_prefix,self.constructor,self.cfg)
            return filter_hq(bundle,str(turn["text"]),self.qa)
        except Exception as exc:
            # The manuscript specifies raw/none-only selection when candidate
            # generation or its common checks are unavailable.
            fallbacks.append(f"candidate_bundle_unavailable:{type(exc).__name__}")
            return None

    def _add_generated_arms(self, turn:dict, bundle, reps:dict, gates:dict, feasible:set[str], fallbacks:list[str])->None:
        for action in self.cfg.storage_actions:
            if action=="raw":
                continue
            try:
                representation=build_representation(turn,action,bundle,self.cfg,self.counter)
                gate=gate_representation(representation,str(turn["text"]),self.support,self.cfg,self.entities)
                if gate.feasible:
                    builder_version=str(getattr(self.constructor,"version",type(self.constructor).__name__))
                    representation=replace(representation,metadata={
                        **representation.metadata,
                        "fid_mean":gate.fid_mean,
                        "fid_min":gate.fid_min,
                        "builder_version":builder_version,
                    })
                    reps[action]=representation
                    gates[action]=CandidateDiagnostics(gate.fid_mean,gate.fid_min)
                    feasible.add(action)
            except Exception as exc:
                # An unavailable support/QA adapter rejects only the generated
                # arm and leaves the raw/none fallback intact.
                fallbacks.append(f"{action}_unavailable:{type(exc).__name__}")

    @staticmethod
    def _remove_equivalent_arms(reps:dict, gates:dict, feasible:set[str])->None:
        seen={(reps["raw"].body,reps["raw"].keys):"raw"}
        for action in list(feasible-{"none","raw"}):
            key=(reps[action].body,reps[action].keys)
            if key in seen:
                feasible.remove(action); reps.pop(action,None); gates.pop(action,None)
            else:
                seen[key]=action

    def _stage2_predictions(self, cheap, reps:dict, gates:dict, feasible:set[str], index:ParentRRFIndex)->dict:
        raw_tokens=self.counter.count(reps["raw"].body)
        raw_features=full_action_features(cheap,reps["raw"],raw_tokens,index,self.counter,gates["raw"],self.cfg)
        predictions={}
        for action in feasible-{"none"}:
            features=raw_features if action=="raw" else full_action_features(
                cheap,reps[action],raw_tokens,index,self.counter,gates[action],self.cfg,
            )
            prediction=self.value_model.full(action,features,raw_features)
            ledger=self.resources.action_resources(action,reps[action])
            # Future read cost cannot be measured at write time. Use the
            # separately trained read-cost model required by the manuscript;
            # commit and index resources remain exact object measurements.
            predicted_read=float(self.value_model.read_cost(action,features))
            if predicted_read<0:
                raise ValueError("read_cost 预测必须非负。")
            ledger=replace(ledger,read=replace(ledger.read,expected_body_tokens=predicted_read))
            mean=post_selection_value(prediction.mean,ledger,self.cfg)
            predictions[action]=Stage2Estimate(mean,prediction.sigma,prediction.form_sigma)
        return predictions

    def decide(self,turn:dict,history_prefix:list[dict],index:ParentRRFIndex,select_cost_lb:float=0.0)->Decision:
        cheap=cheap_features(turn,history_prefix,index,self.embedder,self.counter,self.cfg)
        try:
            pre=self._stage1_predictions(cheap)
        except Exception as exc:
            raw=build_representation(turn,"raw",None,self.cfg,self.counter)
            action=self.cfg.value_model_fallback
            return Decision(action,None if action=="none" else raw,False,("none","raw"),{"fallbacks":(f"value_model_unavailable:{type(exc).__name__}",)},SelectResources())
        if early_reject(pre,select_cost_lb,self.cfg):
            return Decision("none",None,True,("none","raw"),{"pre":pre},SelectResources())

        reps,gates,feasible=self._raw_arms(turn)
        fallbacks=[]
        bundle=self._candidate_bundle(turn,history_prefix,fallbacks)
        select_res=self.resources.select_resources()
        if bundle is not None:
            self._add_generated_arms(turn,bundle,reps,gates,feasible,fallbacks)
        # Equivalent actions are merged (e.g. raw+hq after all HQ keys fail answerability).
        self._remove_equivalent_arms(reps,gates,feasible)
        try:
            full=self._stage2_predictions(cheap,reps,gates,feasible,index)
        except Exception as exc:
            action=self.cfg.value_model_fallback
            fallbacks.append(f"value_model_unavailable:{type(exc).__name__}")
            return Decision(action,None if action=="none" else reps["raw"],False,tuple(sorted(feasible)),{"pre":pre,"fallbacks":tuple(fallbacks)},select_res)
        action=safe_stage2(full,feasible,self.cfg)
        predictions={"pre":pre,"full":full}
        if fallbacks:
            predictions["fallbacks"]=tuple(fallbacks)
        return Decision(action,None if action=="none" else reps[action],False,tuple(sorted(feasible)),predictions,select_res)

    def decide_and_commit(self,turn:dict,history_prefix:list[dict],index:ParentRRFIndex,writer:ParentWriter,select_cost_lb:float=0.0)->Decision:
        decision=self.decide(turn,history_prefix,index,select_cost_lb)
        if decision.representation is not None:
            # Writer owns the transaction: body, keys, metadata, action/version/fidelity are committed atomically.
            writer.atomic_write(decision.representation)
        return decision
