from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import re
from .actions import CandidateBundle, Representation
from .config import R2WConfig

class SupportModel(Protocol):
    def score(self, claim:str, source:str)->float: ...
class QAChecker(Protocol):
    def answerable(self, question:str, source:str)->bool: ...
class EntityExtractor(Protocol):
    def entities(self, text:str)->set[str]: ...

class RegexEntityExtractor:
    def entities(self,text:str)->set[str]:
        return {x.strip() for x in re.findall(r"\b(?:[A-Z][\w.-]*)(?:\s+[A-Z][\w.-]*)*\b",text) if len(x.strip())>1}

@dataclass(frozen=True)
class GateResult:
    feasible: bool
    fid_mean: float
    fid_min: float
    reason: str=""


def _numbers_dates(text:str)->set[str]:
    return set(re.findall(r"[$€£¥]?\b\d+(?:[.,]\d+)?%?\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",text))

def _claims(repr_:Representation)->list[str]:
    # Body metadata is common; split candidate body into atomic-ish sentences and include cards.
    body=re.sub(r"^\[[^\]]+\]\s*","",repr_.body)
    parts=[x.strip() for x in re.split(r"(?<=[.!?。！？])\s+|[;；]\s*",body) if x.strip()]
    # Base body key equals body and is not an extra claim; only additional cards are checked.
    parts.extend(repr_.keys[1:])
    return parts or [body]

def gate_representation(repr_:Representation, source:str, support:SupportModel, cfg:R2WConfig, entity_extractor:EntityExtractor|None=None)->GateResult:
    entity_extractor=entity_extractor or RegexEntityExtractor()
    claims=_claims(repr_)
    source_nums=_numbers_dates(source); source_entities={e.casefold() for e in entity_extractor.entities(source)}
    for claim in claims:
        if not _numbers_dates(claim).issubset(source_nums): return GateResult(False,0.,0.,"number_or_date_not_supported")
        new={e.casefold() for e in entity_extractor.entities(claim)}-source_entities
        # Metadata can contain speaker/time; those are common and excluded from body above.
        if new: return GateResult(False,0.,0.,"new_entity_not_supported")
    scores=[float(support.score(c,source)) for c in claims]
    if not scores or any(not 0<=x<=1 for x in scores): raise ValueError("support model 必须返回 [0,1]。")
    mean=sum(scores)/len(scores); minimum=min(scores)
    ok=mean>=cfg.tau_mean and minimum>=cfg.tau_min
    return GateResult(ok,mean,minimum,"" if ok else "support_below_threshold")

def filter_hq(bundle:CandidateBundle, source:str, qa:QAChecker)->CandidateBundle:
    kept=tuple(q for q in bundle.hq if qa.answerable(q,source))
    return CandidateBundle(bundle.summary,bundle.kv,bundle.event,kept,bundle.graph,bundle.protocol_version)
