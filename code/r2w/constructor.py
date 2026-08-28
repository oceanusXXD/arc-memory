"""结构化草稿构建器。

两种后端只在生成方式上不同；表示契约、长度限制和解析全部仍在
``representations`` 中执行。生产调用方必须显式选择其中一种。
"""

from __future__ import annotations

import json
from typing import Protocol

from .config import R2WConfig
from .llm import LLMClient


class StructuredConstructor(Protocol):
    def json(self, prompt: str) -> dict: ...


class APIStructuredConstructor:
    def __init__(self, client: LLMClient):
        self.client = client

    def json(self, prompt: str) -> dict:
        return self.client.json(prompt, purpose="constructor")


class LocalStructuredConstructor:
    """本地 seq2seq JSON 构建器。

    模型必须针对仓库的 JSON prompts 微调。这里故意不提供规则回退，避免把
    不合约的本地模型输出伪装成草稿。
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        if not model_name.strip():
            raise ValueError("local 构建器需要 constructor_local_model。")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        self.model.eval()

    def json(self, prompt: str) -> dict:
        import torch

        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        with torch.no_grad():
            generated = self.model.generate(**encoded, do_sample=False, max_new_tokens=256)
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("本地结构化构建器没有生成有效 JSON。") from exc
        if not isinstance(value, dict):
            raise RuntimeError("本地结构化构建器 JSON 根节点必须是 object。")
        return value


def build_constructor(cfg: R2WConfig, llm: LLMClient | None = None) -> StructuredConstructor:
    if cfg.constructor_backend == "api":
        return APIStructuredConstructor(llm or LLMClient(cfg))
    return LocalStructuredConstructor(cfg.constructor_local_model, cfg.constructor_local_device)
