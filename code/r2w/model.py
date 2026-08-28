"""按 conversation 分组的 R2W pooled LightGBM 集成。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupKFold

from .config import R2WConfig


def arm_features(action_count: int) -> np.ndarray:
    """十臂的固定特征：压缩位、索引卡 one-hot 与 arm one-hot。"""
    if action_count != 10:
        raise ValueError("R2W 的 arm 特征固定对应十个存储动作。")
    rows = []
    for index in range(action_count):
        compression = float(index >= 5)
        key = index % 5
        key_vector = [float(position == key) for position in range(5)]
        arm_vector = [float(position == index) for position in range(action_count)]
        rows.append([compression, *key_vector, *arm_vector])
    return np.asarray(rows, dtype=np.float32)


def stage2_matrix(
    base_features: np.ndarray, draft_statistics: np.ndarray, action_count: int = 10
) -> np.ndarray:
    """把一条记忆展开为十条 pooled arm 样本。"""
    base = np.asarray(base_features, dtype=np.float32)
    draft = np.asarray(draft_statistics, dtype=np.float32)
    if base.ndim != 2 or draft.shape[:2] != (len(base), action_count):
        raise ValueError("stage2 特征与草稿统计必须按 [turn, arm] 对齐。")
    arms = arm_features(action_count)
    return np.concatenate(
        (
            np.repeat(base, action_count, axis=0),
            np.tile(arms, (len(base), 1)),
            draft.reshape(len(base) * action_count, -1),
        ),
        axis=1,
    ).astype(np.float32)


@dataclass
class GBMEnsemble:
    admission_models: list
    effect_models: list
    hit_models: list
    action_count: int = 10

    @staticmethod
    def _predict(models: list, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not models:
            raise RuntimeError("GBM ensemble 为空。")
        predictions = np.vstack(
            [np.asarray(model.predict(values), dtype=np.float32) for model in models]
        )
        return predictions.mean(axis=0), predictions.std(axis=0, ddof=1)

    def predict_admission(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._predict(self.admission_models, values)

    def predict_arms(
        self, base_features: np.ndarray, draft_statistics: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        matrix = stage2_matrix(base_features, draft_statistics, self.action_count)
        effect, sigma = self._predict(self.effect_models, matrix)
        hit, _ = self._predict(self.hit_models, matrix)
        count = len(base_features)
        return (
            effect.reshape(count, self.action_count),
            np.clip(hit.reshape(count, self.action_count), 0.0, 1.0),
            sigma.reshape(count, self.action_count),
        )


def _regressor(cfg: R2WConfig, objective: str, seed: int):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise RuntimeError("R2W-GBM 需要 lightgbm；请安装 requirements.txt。") from exc
    return LGBMRegressor(
        objective=objective,
        n_estimators=cfg.gbm_estimators,
        learning_rate=cfg.gbm_learning_rate,
        num_leaves=cfg.gbm_num_leaves,
        min_child_samples=cfg.gbm_min_child_samples,
        random_state=seed,
        verbosity=-1,
    )


def train_gbm_ensemble(
    cfg: R2WConfig,
    admission_features: np.ndarray,
    admission_target: np.ndarray,
    arm_base_features: np.ndarray,
    draft_statistics: np.ndarray,
    effects: np.ndarray,
    hit_rates: np.ndarray,
    measured: np.ndarray,
    standard_errors: np.ndarray,
    conversation_ids: np.ndarray,
) -> GBMEnsemble:
    """训练五个互斥 conversation fold 模型，不把最优 arm 压成类别标签。"""
    groups = np.asarray(conversation_ids)
    unique_groups = np.unique(groups)
    if len(unique_groups) < cfg.gbm_folds:
        raise ValueError("R2W-GBM 训练至少需要五段独立对话。")
    admission_features = np.asarray(admission_features, dtype=np.float32)
    admission_target = np.asarray(admission_target, dtype=np.float32)
    base = np.asarray(arm_base_features, dtype=np.float32)
    effects = np.asarray(effects, dtype=np.float32)
    hit_rates = np.asarray(hit_rates, dtype=np.float32)
    measured = np.asarray(measured, dtype=bool)
    standard_errors = np.asarray(standard_errors, dtype=np.float32)
    if effects.shape != hit_rates.shape or effects.shape != measured.shape:
        raise ValueError("逐臂标签必须有相同 [turn, arm] 形状。")
    if len(base) != len(groups) or len(admission_features) != len(groups):
        raise ValueError("特征与 conversation_ids 必须逐 turn 对齐。")

    matrix = stage2_matrix(base, draft_statistics, effects.shape[1])
    flat_groups = np.repeat(groups, effects.shape[1])
    flat_mask = measured.reshape(-1)
    flat_effects = effects.reshape(-1)
    flat_hits = np.clip(hit_rates.reshape(-1), 0.0, 1.0)
    flat_se = standard_errors.reshape(-1)
    folds = GroupKFold(n_splits=cfg.gbm_folds)
    admissions, effect_models, hit_models = [], [], []
    for fold, (train_index, _) in enumerate(folds.split(admission_features, groups=groups)):
        admission = _regressor(cfg, "huber", cfg.random_seed + fold)
        admission.fit(admission_features[train_index], admission_target[train_index])
        admissions.append(admission)

        train_groups = set(groups[train_index].tolist())
        arm_train = flat_mask & np.isin(flat_groups, list(train_groups))
        if not arm_train.any():
            raise RuntimeError("一个 GBM fold 没有 measured arm 标签。")
        weights = None
        if cfg.inverse_variance_weights:
            weights = 1.0 / np.maximum(flat_se[arm_train] ** 2, 1e-6)
        effect = _regressor(cfg, "huber", cfg.random_seed + 100 + fold)
        effect.fit(matrix[arm_train], flat_effects[arm_train], sample_weight=weights)
        effect_models.append(effect)
        hit = _regressor(cfg, "binary", cfg.random_seed + 200 + fold)
        hit.fit(matrix[arm_train], flat_hits[arm_train])
        hit_models.append(hit)
    return GBMEnsemble(admissions, effect_models, hit_models, effects.shape[1])
