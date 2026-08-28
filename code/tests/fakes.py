from __future__ import annotations

import hashlib

import numpy as np

from r2w.query_generator import GeneratedQueries
from r2w.teacher import ReaderAnswer


class FakeEmbedder:
    def __init__(self, dim: int):
        self.dim = dim

    def encode(self, texts, normalize_embeddings=True, batch_size=128):
        rows = []
        for text in texts:
            digest = hashlib.sha256(str(text).encode("utf-8")).digest()
            row = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
            row = np.resize(row, self.dim)
            row /= np.linalg.norm(row) or 1.0
            rows.append(row)
        return np.asarray(rows, dtype=np.float32)


class FakeLLM:
    def __init__(self):
        self.text_calls = 0
        self.json_calls = 0

    def text(self, prompt, *, purpose="unspecified"):
        self.text_calls += 1
        return "Alice moved to Boston."

    def json(self, prompt, *, purpose="unspecified"):
        self.json_calls += 1
        if "键值" in prompt:
            return {
                "items": [
                    {"key": "Alice location", "value": "Alice moved to Boston"},
                    {"key": "move date", "value": "8 May 2023"},
                ]
            }
        if "可检索事件" in prompt:
            return {
                "items": [
                    {"time": "2023-05-08", "event": "Alice moved to Boston"},
                    {"time": "2023-05-08", "event": "Alice told Bob"},
                ]
            }
        if "实体关系" in prompt:
            return {
                "relations": [
                    {
                        "subject": "Alice",
                        "relation": "moved_to",
                        "object": "Boston",
                        "value": "Boston",
                        "valid_from": "2023-05-08",
                        "valid_to": None,
                    }
                ]
            }
        return {"answer": "Boston", "cites": [0]}


class FakeQG:
    def generate(self, previous, text):
        return GeneratedQueries(
            ("Where did Alice move?", "What is the weather on Mars?"), -1.0
        )


class FakeReader:
    def answer(self, question, payloads, current_date):
        found = next(
            (index for index, payload in enumerate(payloads) if "Boston" in payload),
            None,
        )
        return ReaderAnswer(
            "Boston" if found is not None else "unknown",
            () if found is None else (found,),
        )


def small_conversation():
    turns = [
        {
            "dia_id": "D1:1",
            "speaker": "Alice",
            "text": "I moved to Boston yesterday.",
            "timestamp": "1:00 pm on 8 May, 2023",
            "session": 1,
        },
        {
            "dia_id": "D1:2",
            "speaker": "Bob",
            "text": "I like tea.",
            "timestamp": "1:00 pm on 8 May, 2023",
            "session": 1,
        },
        {
            "dia_id": "D1:3",
            "speaker": "Alice",
            "text": "Boston is cold.",
            "timestamp": "2:00 pm on 9 May, 2023",
            "session": 2,
        },
    ]
    return {
        "turns": turns,
        "qa": [
            {
                "question": "Where did Alice move?",
                "answer": "Boston",
                "evidence": ["D1:1"],
                "category": 1,
            },
            {
                "question": "What city is cold?",
                "answer": "Boston",
                "evidence": ["D1:3"],
                "category": 1,
            },
        ],
        "dia_to_index": {"D1:1": 0, "D1:2": 1, "D1:3": 2},
        "speaker_a": "Alice",
        "speaker_b": "Bob",
        "conversation_id": "fake-conversation",
    }
