"""冻结的本地假设 query 生成器。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch

from .config import R2WConfig


@dataclass(frozen=True)
class GeneratedQueries:
    texts: tuple[str, str]
    mean_log_likelihood: float


class QueryGenerator:
    """使用确定性 beam 解码一次生成两条固定候选 query。"""

    def __init__(self, cfg: R2WConfig):
        if cfg.qg_num_queries != 2 or cfg.qg_num_beams < cfg.qg_num_queries:
            raise ValueError(
                "FINAL 算法固定使用两条 query，且 beam 数不能小于 query 数。"
            )
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.cfg = cfg
        torch.manual_seed(cfg.random_seed)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.qg_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.qg_model)
        self.model.eval()
        self._cache: dict[str, GeneratedQueries] = {}

    @staticmethod
    def context(previous_text: str, text: str) -> str:
        return f"{previous_text}\n{text}" if previous_text else text

    def generate(self, previous_text: str, text: str) -> GeneratedQueries:
        source = self.context(previous_text, text)
        if source in self._cache:
            return self._cache[source]
        encoded = self.tokenizer(
            [source], return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            generated = self.model.generate(
                **encoded,
                do_sample=False,
                num_beams=self.cfg.qg_num_beams,
                num_return_sequences=self.cfg.qg_num_queries,
                max_new_tokens=64,
                return_dict_in_generate=True,
                output_scores=True,
            )
        texts = tuple(
            value.strip()
            for value in self.tokenizer.batch_decode(
                generated.sequences, skip_special_tokens=True
            )
        )
        if len(texts) != 2 or any(not value for value in texts):
            raise RuntimeError("本地 QG 未生成两条非空假设 query。")
        transition_scores = self.model.compute_transition_scores(
            generated.sequences,
            generated.scores,
            generated.beam_indices,
            normalize_logits=True,
        )
        log_values = transition_scores[transition_scores < 0].tolist()
        if not log_values:
            raise RuntimeError("本地 QG 未返回可计算的 token 对数似然。")
        result = GeneratedQueries(
            texts=(texts[0], texts[1]), mean_log_likelihood=float(np.mean(log_values))
        )
        self._cache[source] = result
        return result

    def generate_many(self, pairs: Iterable[tuple[str, str]]) -> list[GeneratedQueries]:
        return [self.generate(previous, text) for previous, text in pairs]
