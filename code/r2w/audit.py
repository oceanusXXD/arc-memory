from __future__ import annotations
import math
from .replay import MemoryReplay

def audit_replay(replay:MemoryReplay,tolerance:float=1e-6)->dict:
    probs=replay.audit.probabilities; inds=replay.audit.indicators
    bad=[q for q in replay.audit.changed if not (0<probs[q]<=1)]
    bad_i=[q for q,v in enumerate(inds) if v not in (0,1)]
    by={x.action:x for x in replay.labels}; raw=by["raw"]
    identity={a:abs(x.delta_e-(raw.delta_e+x.delta_f)) for a,x in by.items() if a not in {"none","raw"}}
    return {"positive_probability_ok":not bad,"indicator_ok":not bad_i,"double_contrast_identity_ok":all(v<=tolerance for v in identity.values()),"max_identity_error":max(identity.values(),default=0.0)}
