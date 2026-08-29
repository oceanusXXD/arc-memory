import unittest
from dataclasses import asdict
import numpy as np
from r2w.config import R2WConfig
from r2w.model import train_double_contrast, DoubleContrastValueModel
from r2w.artifacts import rows_from_replay, REQUIRED_VERSION_FIELDS
from r2w.actions import Representation
from r2w.retrieval import ParentRRFIndex
from r2w.replay import replay_memory
from tests.fakes import FakeEmbedder, FakeRunner

class TestModelArtifact(unittest.TestCase):
    def test_double_contrast_training_and_prediction(self):
        cfg=R2WConfig(); rng=np.random.default_rng(1); n=10; d=7
        groups=np.array([f"dialogue-{i//2+1:02d}" for i in range(n)])
        raw_x=rng.normal(size=(n,d)); raw_y=rng.normal(size=n)
        form=[]; fy=[]; read=[]; ry=[]
        cheap=[]; cy=[]
        for i in range(n):
            for a in cfg.storage_actions:
                x=rng.normal(size=d)
                form.append([i,*x]); fy.append(rng.normal())
                read.append([i,*x]); ry.append(abs(rng.normal()))
                c=rng.normal(size=5+len(cfg.storage_actions)); cheap.append([i,*c]); cy.append(rng.normal())
        ens=train_double_contrast(cfg,raw_x,raw_y,np.asarray(form),np.asarray(fy),np.asarray(read),np.asarray(ry),np.asarray(cheap),np.asarray(cy),groups)
        self.assertEqual(len(ens.raw_models),5)
        model=DoubleContrastValueModel(ens)
        pred=model.full("raw",raw_x[0],raw_x[0])
        self.assertTrue(np.isfinite(pred.mean))

    def test_artifact_contains_v4_fields(self):
        cfg=R2WConfig(s_min=1.0,retrieval_k=1)
        raw=Representation("raw","parent-001","value-a",("value-a",),{})
        idx=ParentRRFIndex([raw],FakeEmbedder(),cfg)
        rep=replay_memory(idx,"parent-001",{"none":None,"raw":raw},[{"question":"attribute","answer":"value-a"}],FakeRunner(),cfg)
        versions={key:f"{key}-v1" for key in REQUIRED_VERSION_FIELDS}
        rows,audit=rows_from_replay("dialogue-001","parent-001","raw",1,rep,cfg,versions=versions)
        d=asdict(rows[0])
        self.assertIn("select_resources",d); self.assertIn("commit_resources",d); self.assertEqual(d["config_hash"],cfg.hash())
        self.assertEqual(audit.fold_id,1)

    def test_artifact_rejects_missing_frozen_versions(self):
        cfg=R2WConfig(s_min=1.0,retrieval_k=1)
        raw=Representation("raw","parent-001","value-a",("value-a",),{})
        idx=ParentRRFIndex([raw],FakeEmbedder(),cfg)
        rep=replay_memory(idx,"parent-001",{"none":None,"raw":raw},[{"question":"attribute","answer":"value-a"}],FakeRunner(),cfg)
        with self.assertRaises(ValueError):
            rows_from_replay("dialogue-001","parent-001","raw",1,rep,cfg)
