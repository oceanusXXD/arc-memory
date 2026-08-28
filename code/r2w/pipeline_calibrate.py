"""不重训 GBM 地重新选择 λ/κ 决策配置。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np

from .calibration import CostStatistics, select_policy, save_policy
from .config import R2WConfig, load_config
from .model import GBMEnsemble


def recalibrate(artifacts: str | Path, lambda_w: float, lambda_r: float, lambda_s: float) -> dict:
    directory = Path(artifacts)
    cfg = load_config(directory / "config.json")
    updated = R2WConfig(**(asdict(cfg) | {"lambda_w": lambda_w, "lambda_r": lambda_r, "lambda_s": lambda_s}))
    data = np.load(directory / "measurement_data.npz", allow_pickle=False)
    if str(data["config_hash"]) != cfg.hash():
        raise RuntimeError("measurement_data 与训练工件配置不一致。")
    model = joblib.load(directory / "model.joblib")
    if not isinstance(model, GBMEnsemble):
        raise RuntimeError("model.joblib 不是 GBM artifact。")
    admission, admission_sigma = model.predict_admission(data["validation_stage1"])
    effect, hit, sigma = model.predict_arms(data["validation_stage2"], data["validation_draft"])
    policy = select_policy(
        updated,
        CostStatistics.fit(data["validation_costs"], data["validation_hits"]),
        admission,
        admission_sigma,
        effect,
        hit,
        sigma,
        data["validation_costs"],
        data["validation_effects"],
        data["validation_hits"],
        data["validation_measured"].astype(bool),
    )
    with (directory / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(updated.to_dict(), handle, ensure_ascii=False, indent=2)
    with (directory / "policy.json").open("w", encoding="utf-8") as handle:
        json.dump(save_policy(policy), handle, ensure_ascii=False, indent=2)
    return {"config_hash": updated.hash(), "policy": save_policy(policy)}


def main() -> None:
    parser = argparse.ArgumentParser(description="重新选择 R2W-GBM 运营汇率")
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--lambda-w", type=float, required=True)
    parser.add_argument("--lambda-r", type=float, required=True)
    parser.add_argument("--lambda-s", type=float, required=True)
    arguments = parser.parse_args()
    print(json.dumps(recalibrate(arguments.artifacts, arguments.lambda_w, arguments.lambda_r, arguments.lambda_s), ensure_ascii=False))


if __name__ == "__main__":
    main()
