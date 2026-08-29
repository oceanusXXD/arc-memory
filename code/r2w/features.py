from __future__ import annotations
from dataclasses import dataclass
import re, numpy as np
from .actions import Representation, TokenCounter
from .retrieval import ParentRRFIndex
from .config import R2WConfig

@dataclass(frozen=True)
class CandidateDiagnostics:
    fid_mean:float=1.0; fid_min:float=1.0

def _scalar_text(text:str,counter:TokenCounter):
    return np.asarray([
        counter.count(text), len(re.findall(r"[.!?。！？]",text)),
        len(re.findall(r"\d",text)), len(re.findall(r"\b(?:not|never|no)\b|不|没|无",text,re.I)),
        float("?" in text or "？" in text),
    ],dtype=np.float32)

def cheap_features(turn:dict, history_prefix:list[dict], index:ParentRRFIndex, embedder, counter:TokenCounter, cfg:R2WConfig)->np.ndarray:
    text=str(turn["text"]); emb=np.asarray(embedder.encode([text],normalize_embeddings=True)[0],dtype=np.float32)
    scalar=_scalar_text(text,counter)
    results=index.retrieve(text,min(10,max(1,len(index.representations)))) if index.representations else []
    scores=np.asarray([x.rrf for x in results],dtype=np.float32)
    index_stats=np.asarray([
        len(index.representations), len(index.units),
        float(scores[0]) if len(scores) else 0., float(scores.mean()) if len(scores) else 0.,
        float(scores[0]-scores[-1]) if len(scores)>1 else 0.,
        len(history_prefix), int(turn.get("session",0)),
    ],dtype=np.float32)
    return np.concatenate([emb,scalar,index_stats])

def full_action_features(cheap:np.ndarray, representation:Representation, raw_body_tokens:int, index:ParentRRFIndex, counter:TokenCounter, diag:CandidateDiagnostics, cfg:R2WConfig)->np.ndarray:
    body_tokens=counter.count(representation.body); card_tokens=sum(counter.count(k) for k in representation.keys[1:])
    rehearsals=[]
    for key in representation.keys[1:]:
        r=index.retrieve(key,min(5,max(1,len(index.representations)))) if index.representations else []
        rehearsals.append(r[0].rrf if r else 0.)
    action_onehot=np.asarray([float(representation.action==a) for a in cfg.storage_actions],dtype=np.float32)
    candidate=np.asarray([
        body_tokens, body_tokens/max(raw_body_tokens,1), len(representation.keys)-1, card_tokens,
        diag.fid_mean, diag.fid_min, max(rehearsals,default=0.), float(np.mean(rehearsals)) if rehearsals else 0.,
    ],dtype=np.float32)
    return np.concatenate([cheap,action_onehot,candidate])
