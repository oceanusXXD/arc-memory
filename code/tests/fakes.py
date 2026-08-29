from __future__ import annotations
import numpy as np
from r2w.policy import Estimate, Stage2Estimate
from r2w.costs import SelectResources, CommitResources, IndexResources, ReadResources, ResourceLedger

class FakeEmbedder:
    def __init__(self,dim=8): self.dim=dim
    def encode(self,texts,normalize_embeddings=True,batch_size=128):
        out=[]
        for text in texts:
            v=np.zeros(self.dim,dtype=np.float32)
            for i,b in enumerate(str(text).encode()): v[i%self.dim]+=((b%31)+1)
            if normalize_embeddings and np.linalg.norm(v): v/=np.linalg.norm(v)
            out.append(v)
        return np.asarray(out)

class FakeConstructor:
    def __init__(self): self.calls=0
    def json(self,prompt):
        self.calls+=1
        return {"summary":"value-a.","kv":[{"entity":"subject-1","attribute":"status","value":"value-a"}],"event":[{"time":"2001-02-03","subject":"subject-1","event":"status value-a"}],"hq":["Which status has subject-1?"],"graph":[{"subject":"subject-1","relation":"has_status","object":"value-a"}]}

class FakeSupport:
    def score(self,claim,source): return 1.0 if "invalid-token" not in claim else 0.0
class FakeQA:
    def answerable(self,question,source): return "subject-1" in question

class FakeRunner:
    reader_version="reader-test"; scorer_version="scorer-test"; prompt_version="prompt-test"; seed=0
    def __init__(self): self.calls=0
    def score(self,query,bodies):
        self.calls+=1
        gold=str(query.get("answer","")).casefold()
        joined=" ".join(bodies).casefold()
        return float(bool(gold) and gold in joined)

class FakeValueModel:
    def cheap(self,action,features):
        return Estimate({"raw":0.2,"sum":0.1,"raw+kv":0.3,"raw+event":0.2,"raw+hq":0.1}.get(action,.0),0.01)
    def full(self,action,features,raw_features):
        means={"raw":0.2,"sum":0.15,"raw+kv":0.35,"raw+event":0.25,"raw+hq":0.1}
        return Stage2Estimate(means[action],.01,.01)
    def read_cost(self,action,features): return 0.0

class FakeResources:
    def select_resources(self): return SelectResources(calls=1)
    def action_resources(self,action,representation):
        return ResourceLedger(commit=CommitResources(calls=1),index=IndexResources(body_bytes=len(representation.body),key_bytes=sum(map(len,representation.keys)),vector_count=len(representation.keys)),read=ReadResources())
