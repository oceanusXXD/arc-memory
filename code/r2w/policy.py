from __future__ import annotations
from dataclasses import dataclass
from .config import R2WConfig

@dataclass(frozen=True)
class Estimate:
    mean: float
    sigma: float
@dataclass(frozen=True)
class Stage2Estimate:
    mean: float
    sigma: float
    form_sigma: float=0.0

def early_reject(estimates:dict[str,Estimate], select_cost_lb:float, cfg:R2WConfig)->bool:
    storage=[a for a in cfg.storage_actions if a in estimates]
    if not storage: raise ValueError("第一级缺少可部署存储动作预测。")
    best=max(estimates[a].mean+cfg.kappa_e*estimates[a].sigma for a in storage)
    return best - cfg.lambda_select*select_cost_lb/cfg.H_Q <= 0

def safe_stage2(estimates:dict[str,Stage2Estimate], feasible:set[str], cfg:R2WConfig)->str:
    if "raw" not in estimates: raise ValueError("第二级必须包含 raw。")
    safe=[]
    raw=estimates["raw"]
    if "raw" in feasible and raw.mean+cfg.kappa_e*raw.sigma>0: safe.append("raw")
    for a in feasible:
        if a in {"none","raw"} or a not in estimates: continue
        e=estimates[a]
        u=e.mean+cfg.kappa_e*e.sigma
        lf=(e.mean-raw.mean)-cfg.kappa_f*e.form_sigma
        if u>0 and lf>0: safe.append(a)
    return "none" if not safe else max(safe,key=lambda a:(estimates[a].mean,a))
