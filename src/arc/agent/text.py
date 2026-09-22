from __future__ import annotations

import math
import re
import string
from collections import Counter

try:
    from nltk.stem import PorterStemmer
except Exception:  # pragma: no cover - import-time optional before install
    PorterStemmer = None


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_stemmer = PorterStemmer() if PorterStemmer else None


def word_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


def token_set(text: str) -> set[str]:
    return set(word_tokens(text))


def approx_tokens(text: str) -> int:
    words = len(re.findall(r"\S+", str(text)))
    return max(1, math.ceil(words * 1.3))


def split_by_token_budget(text: str, max_tokens: int) -> list[str]:
    words = str(text).split()
    if not words:
        return [""]
    chunk_words = max(1, int(max_tokens / 1.3))
    return [" ".join(words[i : i + chunk_words]) for i in range(0, len(words), chunk_words)]


def normalize_answer(value: str) -> str:
    text = str(value).replace(",", "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return " ".join(text.split())


def _stem(token: str) -> str:
    return _stemmer.stem(token) if _stemmer else token


def f1_score(prediction: str, ground_truth: str) -> float:
    pred = [_stem(w) for w in normalize_answer(prediction).split()]
    gold = [_stem(w) for w in normalize_answer(ground_truth).split()]
    if not pred or not gold:
        return 0.0
    common = Counter(pred) & Counter(gold)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred)
    recall = same / len(gold)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(set(normalize_answer(prediction).split()) == set(normalize_answer(ground_truth).split()))


def multi_answer_f1(prediction: str, ground_truth: str) -> float:
    preds = [p.strip() for p in str(prediction).split(",") if p.strip()]
    golds = [g.strip() for g in str(ground_truth).split(",") if g.strip()]
    if not preds or not golds:
        return 0.0
    return sum(max(f1_score(pred, gold) for pred in preds) for gold in golds) / len(golds)


def lexical_score(query: str, text: str) -> float:
    q = token_set(query)
    if not q:
        return 0.0
    t = token_set(text)
    overlap = len(q & t)
    return overlap / math.sqrt(max(1, len(q)) * max(1, len(t)))
