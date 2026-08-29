from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import R2WConfig


def _regressor(cfg:R2WConfig,seed:int):
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(objective="huber",n_estimators=200,learning_rate=.05,num_leaves=15,min_child_samples=10,random_state=seed,verbosity=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(loss="absolute_error",random_state=seed)

@dataclass
class DoubleContrastEnsemble:
    raw_models:list
    form_models:list
    read_models:list
    cheap_models:list
    actions:tuple[str,...]

    @staticmethod
    def _pred(models,x):
        p=np.vstack([np.asarray(m.predict(x),dtype=float) for m in models]);
        return p.mean(0),p.std(0,ddof=1) if len(models)>1 else np.zeros(p.shape[1])

    def predict_raw(self,x): return self._pred(self.raw_models,x)
    def predict_form(self,x): return self._pred(self.form_models,x)
    def predict_read(self,x): return self._pred(self.read_models,x)
    def predict_cheap(self,x): return self._pred(self.cheap_models,x)


def train_double_contrast(cfg:R2WConfig, raw_x, raw_y, form_x, form_y, read_x, read_y, cheap_x, cheap_y, groups, n_models=5)->DoubleContrastEnsemble:
    from sklearn.model_selection import GroupKFold
    groups=np.asarray(groups); unique=np.unique(groups)
    if len(unique)<n_models: raise ValueError("训练至少需要与 ensemble 数相同的独立对话。")
    raw_x=np.asarray(raw_x); raw_y=np.asarray(raw_y); form_x=np.asarray(form_x); form_y=np.asarray(form_y); read_x=np.asarray(read_x); read_y=np.asarray(read_y); cheap_x=np.asarray(cheap_x); cheap_y=np.asarray(cheap_y)
    # form/read/cheap rows carry a first column source-row index to map group without leaking it into model.
    folds=GroupKFold(n_splits=n_models); raws=[]; forms=[]; reads=[]; cheaps=[]
    for f,(tr,_) in enumerate(folds.split(raw_x,groups=groups)):
        allowed=set(tr.tolist())
        rm=_regressor(cfg,cfg.random_seed+f); rm.fit(raw_x[tr],raw_y[tr]); raws.append(rm)
        def fit_rows(x,y,seed):
            idx=x[:,0].astype(int); mask=np.array([i in allowed for i in idx]); m=_regressor(cfg,seed); m.fit(x[mask,1:],y[mask]); return m
        forms.append(fit_rows(form_x,form_y,cfg.random_seed+100+f)); reads.append(fit_rows(read_x,read_y,cfg.random_seed+200+f)); cheaps.append(fit_rows(cheap_x,cheap_y,cfg.random_seed+300+f))
    return DoubleContrastEnsemble(raws,forms,reads,cheaps,cfg.storage_actions)


@dataclass
class DoubleContrastValueModel:
    """Adapter used by OnlineWriter. Cheap rows are pooled action rows; full model enforces DeltaE=DeltaE_raw+DeltaF."""
    ensemble: DoubleContrastEnsemble

    def cheap(self, action:str, features):
        from .policy import Estimate
        # append action one-hot so one shared cheap model predicts every deployable arm
        one=np.asarray([float(action==a) for a in self.ensemble.actions],dtype=float)
        x=np.concatenate([np.asarray(features,dtype=float),one])[None,:]
        mean,sigma=self.ensemble.predict_cheap(x)
        return Estimate(float(mean[0]),float(sigma[0]))

    def full(self, action:str, features, raw_features):
        from .policy import Stage2Estimate
        if raw_features is None:
            raw_features=features
        raw_mean,raw_sigma=self.ensemble.predict_raw(np.asarray(raw_features,dtype=float)[None,:])
        if action=="raw":
            return Stage2Estimate(float(raw_mean[0]),float(raw_sigma[0]),0.0)
        # Independent model ensembles use matched model index; compute paired DeltaE predictions.
        raw_preds=np.asarray([m.predict(np.asarray(raw_features,dtype=float)[None,:])[0] for m in self.ensemble.raw_models],dtype=float)
        form_preds=np.asarray([m.predict(np.asarray(features,dtype=float)[None,:])[0] for m in self.ensemble.form_models],dtype=float)
        de_preds=raw_preds+form_preds
        mean=float(de_preds.mean()); sigma=float(de_preds.std(ddof=1)) if len(de_preds)>1 else 0.0
        fs=float(form_preds.std(ddof=1)) if len(form_preds)>1 else 0.0
        return Stage2Estimate(mean,sigma,fs)

    def read_cost(self,action:str,features)->float:
        mean,_=self.ensemble.predict_read(np.asarray(features,dtype=float)[None,:])
        return max(0.0,float(mean[0]))
