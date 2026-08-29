from __future__ import annotations
from dataclasses import dataclass, asdict
from .config import R2WConfig

@dataclass(frozen=True)
class SelectResources:
    calls: int=0; input_tokens:int=0; output_tokens:int=0; gpu_ms:float=0.; wall_ms:float=0.; cost:float=0.
@dataclass(frozen=True)
class CommitResources:
    calls:int=0; gpu_ms:float=0.; wall_ms:float=0.; cost:float=0.
@dataclass(frozen=True)
class IndexResources:
    body_bytes:int=0; key_bytes:int=0; vector_count:int=0; vector_bytes:int=0; retention_days:float=0.
@dataclass(frozen=True)
class ReadResources:
    hit_probability:float=0.; expected_body_tokens:float=0.; retrieval_ms:float=0.; reader_ms:float=0.; cost:float=0.
@dataclass(frozen=True)
class ResourceLedger:
    select:SelectResources=SelectResources(); commit:CommitResources=CommitResources(); index:IndexResources=IndexResources(); read:ReadResources=ReadResources()
    def to_dict(self): return {"select":asdict(self.select),"commit":asdict(self.commit),"index":asdict(self.index),"read":asdict(self.read)}

def scalar_select(x:SelectResources, unit:str="cost")->float: return float(getattr(x,unit))
def scalar_commit(x:CommitResources, unit:str="cost")->float: return float(getattr(x,unit))
def scalar_index(x:IndexResources, unit:str="byte_day")->float:
    if unit=="byte_day": return float((x.body_bytes+x.key_bytes)*max(x.retention_days,1.0))
    if unit=="bytes": return float(x.body_bytes+x.key_bytes)
    return float(getattr(x,unit))
def scalar_read(x:ReadResources, unit:str="expected_body_tokens")->float: return float(getattr(x,unit))

def post_selection_value(delta_e:float, ledger:ResourceLedger, cfg:R2WConfig)->float:
    return float(delta_e - cfg.lambda_commit*scalar_commit(ledger.commit,cfg.commit_scalar)/cfg.H_Q - cfg.lambda_index*scalar_index(ledger.index,cfg.index_scalar)/cfg.H_Q - cfg.lambda_read*scalar_read(ledger.read,cfg.read_scalar))
def enter_value(values:list[float], select:SelectResources, cfg:R2WConfig)->float:
    return float(max([0.0,*values]) - cfg.lambda_select*scalar_select(select,cfg.select_scalar)/cfg.H_Q)
