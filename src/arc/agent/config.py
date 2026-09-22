"""Configuration loading and invariant checks."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "run_id": "locomo_arc", "seed": 7,
    "paths": {"raw_data": "data/LoCoMo/data/locomo10.json", "split_manifest": "data/splits/locomo_group_split.json", "processed_dir": "data/processed", "cache_dir": "data/cache", "models_dir": "models", "runs_dir": "runs"},
    "api": {"protocol": "claude_code", "timeout_seconds": 900},
    "models": {"agent": "${ARC_AGENT_MODEL_ID:Qwen/Qwen3.5-4B}", "builder": "${ARC_BUILDER_MODEL_ID:Qwen/Qwen3.5-4B}", "auditor": "${ARC_AUDITOR_MODEL_ID:Qwen/Qwen3.5-4B}", "teacher": "${ARC_TEACHER_MODEL_ID:grok-4.6}", "compiler": "${ARC_COMPILER_MODEL_ID:Qwen/Qwen3.5-4B}"},
    "claude_code": {"command": "${CLAUDE_CODE_COMMAND:claude}", "model": "${CLAUDE_CODE_MODEL:Qwen/Qwen3.5-4B}", "timeout_seconds": 900, "require_credentials": True, "base_url": "${ANTHROPIC_BASE_URL:https://api.siliconflow.cn}", "api_key_env": "${ARC_API_KEY_ENV:SILICONFLOW_API_KEY}", "thinking": True, "thinking_budget_tokens": 1024, "extra_args": []},
    "agent": {"provider": "claude_code", "model": "${ARC_AGENT_MODEL_ID:Qwen/Qwen3.5-4B}", "max_tokens": 2048, "temperature": 0.7, "timeout_seconds": 900},
    "compiler": {"provider": "claude_code", "evaluate_queries": True, "tau_query": 0.5, "tau_utility": 0.5, "tau_coverage": 0.5},
    "decoder": {"temperature": 0.7, "constructor_max_tokens": 2048, "relation_render_tokens": 512},
    "tokenizer": {"enabled": True, "model": "${ARC_BUILDER_MODEL_ID:Qwen/Qwen3.5-4B}", "trust_remote_code": True, "chat_template_kwargs": {"enable_thinking": True}},
    "retrieval": {"embedding_model": "Qwen/Qwen3-Embedding-0.6B", "query_prompt_name": "query", "cache_vectors": True, "backend": "hybrid", "top_k": 128, "rrf_k": 60, "source_cap_tokens": 8192, "max_blocks": 64},
    "budgets": {"agent_context_tokens": 16384, "builder_input_limit": 15872, "builder_input_tokens": 4096, "deployment": [1024, 2048, 4096]},
    "experiment_groups": {"D_seed": ["conv-26", "conv-30"], "D_value": ["conv-41", "conv-42"], "D_dev": ["conv-43", "conv-44"], "D_cal": ["conv-47", "conv-48"], "D_eval": ["conv-49", "conv-50"]},
    "splits": {"train": ["conv-26", "conv-30", "conv-41", "conv-42", "conv-43", "conv-44"], "dev": ["conv-47", "conv-48"], "final": ["conv-49", "conv-50"]},
    "selector": {"checkpoint": "", "beam_width": 4, "width": 128, "heads": 4, "ff": 512, "layers": 2, "dropout": 0.0,
                 "architectures": ["Flat", "Chain", "Tree", "Graph", "Cluster"]},
    "baselines": {"fixed_top_k": 4},
}
_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")

def load_dotenv(path: str | Path = ".env") -> None:
    target = Path(path)
    if not target.exists(): return
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'").strip())

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(base)
    for key, value in override.items():
        output[key] = _deep_merge(output[key], value) if isinstance(value, dict) and isinstance(output.get(key), dict) else value
    return output

def _expand(value: Any) -> Any:
    if isinstance(value, dict): return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list): return [_expand(v) for v in value]
    if not isinstance(value, str): return value
    return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)

def validate_config(config: dict[str, Any]) -> None:
    for section in ("paths", "api", "models", "claude_code", "agent", "compiler", "decoder", "retrieval", "budgets", "experiment_groups", "selector", "baselines"):
        if not isinstance(config.get(section), dict): raise ValueError(f"config.{section} must be a mapping")
    if str(config["compiler"].get("provider")) != "claude_code": raise ValueError("compiler.provider must be claude_code")
    if str(config["agent"].get("provider")) != "claude_code": raise ValueError("agent.provider must be claude_code")
    seen: dict[str, str] = {}
    for group in ("D_seed", "D_value", "D_dev", "D_cal", "D_eval"):
        if group not in config["experiment_groups"]: raise ValueError(f"missing experiment group {group}")
        for episode in config["experiment_groups"][group]:
            episode = str(episode)
            if episode in seen: raise ValueError(f"episode {episode} appears in both {seen[episode]} and {group}")
            seen[episode] = group

def load_config(path: str | Path | None = None) -> dict[str, Any]:
    load_dotenv(); config = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        source = Path(path)
        if not source.exists(): raise FileNotFoundError(source)
        config = _deep_merge(config, yaml.safe_load(source.read_text(encoding="utf-8")) or {})
    config = _expand(config); validate_config(config); return config

def run_dir(config: dict[str, Any]) -> Path:
    target = Path(config["paths"]["runs_dir"]) / str(config["run_id"]); target.mkdir(parents=True, exist_ok=True); return target
def processed_path(config: dict[str, Any], name: str) -> Path:
    target = Path(config["paths"]["processed_dir"]); target.mkdir(parents=True, exist_ok=True); return target / name
def cache_path(config: dict[str, Any], *parts: str) -> Path:
    target = Path(config["paths"]["cache_dir"]).joinpath(*parts); target.parent.mkdir(parents=True, exist_ok=True); return target
def model_path(config: dict[str, Any], *parts: str) -> Path:
    target = Path(config["paths"]["models_dir"]).joinpath(*parts); target.parent.mkdir(parents=True, exist_ok=True); return target
def file_sha256(path: str | Path) -> str | None:
    target = Path(path)
    if not target.exists(): return None
    digest = hashlib.sha256()
    with target.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()
def write_manifest(config: dict[str, Any], stage: str, extra: dict[str, Any] | None = None) -> Path:
    base = {"run_id": config["run_id"], "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "config": public_config(config)}
    # Persist the effective frozen counting basis alongside the expanded config.
    from arc.algorithm.memory import tokenizer_basis
    base["tokenizer_basis"] = tokenizer_basis(config)
    manifest = {**base, "stage": stage, **(extra or {})}; folder = run_dir(config); stable = folder / "manifest.json"
    if not stable.exists(): stable.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    output = folder / f"manifest.{stage}.json"; output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"); return output

def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Persist settings without expanded credentials or child-process secrets."""
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: ("<redacted>" if key == "env" or any(s in key.lower() for s in ("api_key", "auth_token", "secret", "password", "authorization")) else clean(item)) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value
    return clean(config)

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/locomo.yaml"); args = parser.parse_args(); print(write_manifest(load_config(args.config), "manifest"))
if __name__ == "__main__": main()
