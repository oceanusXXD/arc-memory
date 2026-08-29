"""Minimal deterministic tokenization and BM25 primitives for R2W retrieval."""

from __future__ import annotations

import collections
import math
import re


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9']*|\d+(?:[.,]\d+)?|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return [token.lower().removesuffix("'s") for token in TOKEN_RE.findall(text)]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("BM25 至少需要一个文档。")
        self.k1, self.b = k1, b
        self.tokens = [tokenize(document) for document in documents]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.avgdl = sum(self.lengths) / len(self.lengths)
        frequencies = collections.Counter(
            token for document in self.tokens for token in set(document)
        )
        self.idf = {
            token: math.log(1 + (len(documents) - count + 0.5) / (count + 0.5))
            for token, count in frequencies.items()
        }

    def scores(self, query: str) -> list[float]:
        query_tokens = set(tokenize(query))
        scores: list[float] = []
        for document, length in zip(self.tokens, self.lengths):
            frequencies = collections.Counter(document)
            score = 0.0
            for token in query_tokens:
                frequency = frequencies[token]
                if frequency:
                    denominator = frequency + self.k1 * (
                        1 - self.b + self.b * length / max(self.avgdl, 1e-12)
                    )
                    score += self.idf.get(token, 0.0) * frequency * (self.k1 + 1) / denominator
            scores.append(score)
        return scores
