from __future__ import annotations
import re

def _natural_key(value:str):
    return tuple(int(x) if x.isdigit() else x.casefold() for x in re.split(r"(\d+)", value))

def locomo_five_folds(ids:list[str]):
    values=sorted(ids,key=_natural_key)
    if len(values)!=10: raise ValueError("LoCoMo 主实验要求恰好 10 个独立长对话。")
    folds=[]
    for f in range(5):
        t=2*f; v=(t+2)%10
        test=(values[t],values[t+1]); val=(values[v],values[(v+1)%10])
        train=tuple(x for x in values if x not in test and x not in val)
        folds.append({"fold":f+1,"train":train,"val":val,"test":test})
    return folds
