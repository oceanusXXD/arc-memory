"""R2W FINAL 使用的确定性文本统计与 BM25 原语。"""

from __future__ import annotations

import collections
import math
import re
from datetime import datetime, timedelta

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9']*|\d+(?:[.,]\d+)?|[\u4e00-\u9fff]")
SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s*|\n+")
DATE_RE = re.compile(
    r"\b(?:\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{2,4})?|"
    r"(?:19|20)\d{2}|\d{1,2}:\d{2}\s*(?:am|pm))\b",
    re.IGNORECASE,
)
RELATIVE_TIME_RE = re.compile(
    r"\b(?:yesterday|today|tomorrow|tonight|last\s+\w+|next\s+\w+|this\s+\w+|"
    r"\d+\s+(?:day|week|month|year)s?\s+ago|earlier|later|recently)\b|昨天|今天|明天|上周|下周|本周|去年|明年",
    re.IGNORECASE,
)
MEASUREMENT_RE = re.compile(
    r"(?:[$€£¥￥%°]|\b(?:dollars?|usd|eur|pounds?|kg|kilograms?|grams?|kilometers?|kilometres?|"
    r"meters?|metres?|centimeters?|centimetres?|miles?|hours?|minutes?|degrees?)\b|"
    r"人民币|元|公斤|千克|克|公里|千米|厘米|小时|分钟|百分之|度)",
    re.IGNORECASE,
)
PRONOUNS = {
    "i",
    "me",
    "my",
    "mine",
    "we",
    "us",
    "our",
    "ours",
    "you",
    "your",
    "yours",
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "they",
    "them",
    "their",
    "theirs",
    "it",
    "its",
}
STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "that",
    "this",
    "it",
    "i",
    "you",
    "we",
    "he",
    "she",
    "they",
}
QUESTION_WORDS = {
    "what",
    "when",
    "who",
    "how",
    "why",
    "where",
    "which",
    "多少",
    "何时",
    "谁",
    "什么",
    "怎么",
}

_RELATIVE_DATE = re.compile(
    r"\b(yesterday|today|tomorrow|last week|next week|last "
    r"monday|last tuesday|last wednesday|last thursday|last friday|last saturday|last sunday|"
    r"next monday|next tuesday|next wednesday|next thursday|next friday|next saturday|next sunday)\b(?!\(=)",
    re.IGNORECASE,
)
_WEEKDAYS = {
    name: index
    for index, name in enumerate(
        ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    )
}


def tokenize(text: str) -> list[str]:
    return [token.lower().removesuffix("'s") for token in TOKEN_RE.findall(text)]


def word_count(text: str) -> int:
    return len(tokenize(text))


def sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_RE.split(text.strip()) if part.strip()]


def relative_time_count(text: str) -> int:
    return len(RELATIVE_TIME_RE.findall(text))


def date_count(text: str) -> int:
    return len(DATE_RE.findall(text)) + relative_time_count(text)


def lexical_overlap(left: str, right: str) -> float:
    left_tokens, right_tokens = set(tokenize(left)), set(tokenize(right))
    return (
        len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        if left_tokens and right_tokens
        else 0.0
    )


def _timestamp_date(value: str) -> datetime:
    for pattern in (
        "%I:%M %p on %d %B, %Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(value, pattern)
        except ValueError:
            continue
    raise ValueError(f"无法解析 session 时间锚：{value!r}。")


def normalize_relative_time(text: str, timestamp: str) -> str:
    """保留相对表达并附绝对日期，供全部表示臂与在线输入共用。"""
    if not _RELATIVE_DATE.search(text):
        return text
    anchor = _timestamp_date(timestamp)

    def replace(match: re.Match) -> str:
        phrase = match.group(0)
        lowered = phrase.casefold()
        fixed = {
            "yesterday": -1,
            "today": 0,
            "tomorrow": 1,
            "last week": -7,
            "next week": 7,
        }
        if lowered in fixed:
            resolved = anchor + timedelta(days=fixed[lowered])
        else:
            direction, weekday = lowered.split()
            target = _WEEKDAYS[weekday]
            if direction == "last":
                resolved = anchor - timedelta(
                    days=(anchor.weekday() - target) % 7 or 7
                )
            else:
                resolved = anchor + timedelta(
                    days=(target - anchor.weekday()) % 7 or 7
                )
        return f"{phrase}(={resolved:%Y-%m-%d})"

    return _RELATIVE_DATE.sub(replace, text)


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("BM25 至少需要一个文档。")
        self.k1, self.b = k1, b
        self.tokens = [tokenize(doc) for doc in documents]
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
                    score += (
                        self.idf.get(token, 0.0)
                        * frequency
                        * (self.k1 + 1)
                        / denominator
                    )
            scores.append(score)
        return scores
