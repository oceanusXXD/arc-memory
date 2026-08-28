"""R2W FINAL 的 M/Q/I 三族特征。"""

from __future__ import annotations

import collections
import math
from itertools import pairwise
from pathlib import Path

import joblib
import numpy as np

from .config import R2WConfig
from .query_generator import QueryGenerator
from .representations import attach_hypothetical_queries, hypothetical_queries, raw_unit
from .retrieval import IndexUnit, UnifiedIndex
from .text import (
    DATE_RE,
    MEASUREMENT_RE,
    PRONOUNS,
    STOPWORDS,
    date_count,
    lexical_overlap,
    relative_time_count,
    sentences,
    tokenize,
)

BM25_DIM = 15
STRUCT_DIM = 13
POSITION_DIM = 5
QUERY_STAT_DIM = 12
INTERACTION_DIM = 18


def _safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        left @ right / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12)
    )


class _NamedEntityExtractor:
    """使用冻结 NER 模型，避免用大小写或正则猜测实体。"""

    def __init__(self, model_name: str):
        from transformers import pipeline

        self._pipeline = pipeline(
            "token-classification", model=model_name, aggregation_strategy="simple"
        )
        self._cache: dict[str, tuple[int, int, int]] = {}

    def counts(self, text: str) -> tuple[int, int, int]:
        if text not in self._cache:
            tokenizer = self._pipeline.tokenizer
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_offsets_mapping=True,
            )
            offsets = [
                offset
                for offset in encoded.get("offset_mapping", [])
                if offset != (0, 0)
            ]
            clipped = text[: offsets[-1][1]] if offsets else text
            entities = self._pipeline(clipped)
            people = sum(
                str(item.get("entity_group", "")).upper() in {"PER", "PERSON"}
                for item in entities
            )
            place_org = sum(
                str(item.get("entity_group", "")).upper()
                in {"LOC", "LOCATION", "ORG", "ORGANIZATION"}
                for item in entities
            )
            self._cache[text] = (people, place_org, len(entities))
        return self._cache[text]


class FeatureBuilder:
    """所有统计只在训练对话 fit，训练与推理共享相同参考索引规则。"""

    def __init__(self, cfg: R2WConfig, embedder, ner=None):
        self.cfg = cfg
        self.embedder = embedder
        self.idf: dict[str, float] = {}
        self.bigram_idf: dict[str, float] = {}
        self.avgdl = 0.0
        self.means: dict[str, np.ndarray] = {}
        self.stds: dict[str, np.ndarray] = {}
        self.ner = ner if ner is not None else _NamedEntityExtractor(cfg.ner_model)
        self._corpus_fitted = False
        self._fitted = False

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(
            self.embedder.encode(texts, normalize_embeddings=True, batch_size=128),
            dtype=np.float32,
        )
        if vectors.ndim != 2 or vectors.shape[1] != self.cfg.embedding_dim:
            raise RuntimeError(
                f"检索编码器维度为 {vectors.shape[1] if vectors.ndim == 2 else 'invalid'}，与 config.embedding_dim={self.cfg.embedding_dim} 不一致。"
            )
        return vectors

    def _fit_corpus_statistics(self, conversations: list[dict]) -> None:
        documents = [
            turn["text"]
            for conversation in conversations
            for turn in conversation["turns"]
        ]
        if not documents:
            raise ValueError("至少需要一个训练 turn。")
        token_documents = [tokenize(text) for text in documents]
        self.avgdl = float(np.mean([len(tokens) for tokens in token_documents]))
        count = len(token_documents)
        document_frequency = collections.Counter(
            token for tokens in token_documents for token in set(tokens)
        )
        bigram_frequency = collections.Counter(
            "\x1f".join(pair)
            for tokens in token_documents
            for pair in set(pairwise(tokens))
        )
        self.idf = {
            token: math.log(1 + (count - value + 0.5) / (value + 0.5))
            for token, value in document_frequency.items()
        }
        self.bigram_idf = {
            token: math.log(1 + (count - value + 0.5) / (value + 0.5))
            for token, value in bigram_frequency.items()
        }
        self._corpus_fitted = True

    @staticmethod
    def _value_summary(values: list[float]) -> tuple[float, float, float]:
        if not values:
            return 0.0, 0.0, 0.0
        array = np.asarray(values, dtype=np.float32)
        return float(array.max()), float(array.mean()), float(np.percentile(array, 90))

    def _bm25_statistics(
        self, text: str, previous_text: str, previous_session_texts: list[str]
    ) -> list[float]:
        tokens = tokenize(text)
        frequencies = collections.Counter(tokens)
        idfs = [self.idf[token] for token in tokens if token in self.idf]
        idf_max, idf_mean, idf_p90 = self._value_summary(idfs)
        rare = sum(value > self.cfg.rare_idf_threshold for value in idfs)
        unique_ratio = len(frequencies) / max(len(tokens), 1)
        normalization = len(tokens) / max(self.avgdl, 1e-12)
        saturation = [
            value
            / (
                value
                + self.cfg.bm25_k1
                * (1 - self.cfg.bm25_b + self.cfg.bm25_b * normalization)
            )
            for value in frequencies.values()
        ]
        max_saturation = max(saturation, default=0.0)
        numeric_idfs = [
            self.idf[token]
            for token in tokens
            if any(character.isdigit() for character in token) and token in self.idf
        ]
        date_tokens = tokenize(" ".join(DATE_RE.findall(text)))
        date_idfs = [self.idf[token] for token in date_tokens if token in self.idf]
        bigrams = ["\x1f".join(pair) for pair in pairwise(tokens)]
        bigram_values = [
            self.bigram_idf[pair] for pair in bigrams if pair in self.bigram_idf
        ]
        oov_ratio = sum(token not in self.idf for token in tokens) / max(len(tokens), 1)
        redundancy = (
            float(
                np.mean(
                    [lexical_overlap(text, other) for other in previous_session_texts]
                )
            )
            if previous_session_texts
            else 0.0
        )
        return [
            idf_max,
            idf_mean,
            idf_p90,
            float(rare),
            unique_ratio,
            normalization,
            max_saturation,
            max(numeric_idfs, default=0.0),
            max(date_idfs, default=0.0),
            lexical_overlap(text, previous_text),
            sum(token in STOPWORDS for token in tokens) / max(len(tokens), 1),
            max(bigram_values, default=0.0),
            float(len(bigrams)),
            oov_ratio,
            redundancy,
        ]

    def _structure_statistics(self, turn: dict, conversation: dict) -> list[float]:
        text = turn["text"]
        tokens = tokenize(text)
        sentence_values = sentences(text)
        people, place_org, entity_count = self.ner.counts(text)
        pronouns = sum(token in PRONOUNS for token in tokens)
        absolute_dates = max(date_count(text) - relative_time_count(text), 0)
        # 两个说话人用一个二值维度表达，保证 M2--M4 为算法规定的 33 个标量。
        speaker_bit = float(turn.get("speaker") == conversation.get("speaker_a"))
        return [
            float(len(tokens)),
            float(len(sentence_values)),
            len(tokens) / max(len(sentence_values), 1),
            float(
                sum(any(character.isdigit() for character in token) for token in tokens)
            ),
            float(bool(MEASUREMENT_RE.search(text))),
            float(absolute_dates),
            float(relative_time_count(text)),
            float(people),
            float(place_org),
            float(pronouns),
            pronouns / max(entity_count, 1),
            float("?" in text or "？" in text),
            speaker_bit,
        ]

    @staticmethod
    def _position_statistics(
        index: int, session_indices: list[int], session_order: list[int], session: int
    ) -> list[float]:
        return [
            session_indices.index(index) / max(len(session_indices) - 1, 1),
            session_order.index(session) / max(len(session_order) - 1, 1),
            math.log1p(len(session_indices)),
            float(index == session_indices[0]),
            float(index == session_indices[-1]),
        ]

    def _memory_features(self, conversation: dict) -> tuple[np.ndarray, np.ndarray]:
        turns = conversation["turns"]
        context_texts = [
            f"{turns[index - 1]['text']}\n{turn['text']}" if index else turn["text"]
            for index, turn in enumerate(turns)
        ]
        embeddings = self._encode(context_texts)
        sessions = [int(turn.get("session", 0)) for turn in turns]
        session_order = list(dict.fromkeys(sessions))
        session_indices = {
            session: [index for index, value in enumerate(sessions) if value == session]
            for session in session_order
        }
        rows = []
        for index, turn in enumerate(turns):
            session = sessions[index]
            previous_same_session = [
                turns[value]["text"]
                for value in session_indices[session]
                if value < index
            ]
            rows.append(
                np.concatenate(
                    (
                        embeddings[index],
                        np.asarray(
                            self._bm25_statistics(
                                turn["text"],
                                turns[index - 1]["text"] if index else "",
                                previous_same_session,
                            ),
                            dtype=np.float32,
                        ),
                        np.asarray(
                            self._structure_statistics(turn, conversation),
                            dtype=np.float32,
                        ),
                        np.asarray(
                            self._position_statistics(
                                index, session_indices[session], session_order, session
                            ),
                            dtype=np.float32,
                        ),
                    )
                )
            )
        return np.asarray(rows, dtype=np.float32), embeddings

    @staticmethod
    def _question_type(query: str) -> list[float]:
        lowered = query.casefold()
        if "how much" in lowered or "多少" in query:
            index = 3
        elif "when" in lowered or "何时" in query or "什么时候" in query:
            index = 1
        elif "who" in lowered or "谁" in query:
            index = 2
        elif "what" in lowered or "什么" in query:
            index = 0
        else:
            index = 4
        return [float(position == index) for position in range(5)]

    def _query_features(self, turn: dict) -> tuple[np.ndarray, np.ndarray]:
        generated = hypothetical_queries(turn)
        queries = list(generated.texts)
        query_embeddings = self._encode(queries)
        tokens = tokenize(" ".join(queries))
        qstat = [
            float(np.mean([len(tokenize(query)) for query in queries])),
            float(abs(len(tokenize(queries[0])) - len(tokenize(queries[1])))),
            _safe_cosine(query_embeddings[0], query_embeddings[1]),
            generated.mean_log_likelihood,
            *self._question_type(queries[0]),
            float(
                any(any(character.isdigit() for character in token) for token in tokens)
                or any(
                    term in " ".join(queries).casefold()
                    for term in ("how much", "how many", "多少")
                )
            ),
            float(
                date_count(" ".join(queries)) > 0
                or relative_time_count(" ".join(queries)) > 0
            ),
            float(
                any(
                    term in " ".join(queries).casefold()
                    for term in ("whose", "with", "related", "谁的", "和谁")
                )
            ),
        ]
        return np.concatenate(
            (query_embeddings.mean(axis=0), np.asarray(qstat, dtype=np.float32))
        ), query_embeddings.mean(axis=0)

    @staticmethod
    def _entropy(values: list[float]) -> float:
        array = np.asarray(values, dtype=np.float32)
        if not len(array) or float(array.sum()) <= 0.0:
            return 0.0
        probabilities = array / array.sum()
        return float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())

    def _interaction_row(
        self,
        conversation: dict,
        index: int,
        memory_embedding: np.ndarray,
        query_embedding: np.ndarray,
    ) -> np.ndarray:
        turns = conversation["turns"]
        turn = turns[index]
        session = int(turn.get("session", 0))
        generated = hypothetical_queries(turn)
        q1 = generated.texts[0]
        content_tokens = set(tokenize(turn["text"]))
        query_tokens = set(tokenize(q1))
        missing = query_tokens - content_tokens
        missing_idfs = [self.idf[token] for token in missing if token in self.idf]
        current_unit = IndexUnit(
            turn_index=index, text=raw_unit(turn, self.cfg), session=session
        )
        session_order = list(
            dict.fromkeys(int(value.get("session", 0)) for value in turns[: index + 1])
        )
        prior_sessions = set(session_order[: session_order.index(session)])
        reference_units = [
            IndexUnit(
                turn_index=prior,
                text=raw_unit(previous, self.cfg),
                session=int(previous.get("session", 0)),
            )
            for prior, previous in enumerate(turns[:index])
            if int(previous.get("session", 0)) in prior_sessions
        ]
        # I_ref 只含当前 session 之前的全部 raw turn，不含当前 session 内的历史 turn。
        raw_index = UnifiedIndex(
            [current_unit, *reference_units], self.embedder, self.cfg
        )
        query_scored = raw_index.scored(q1)
        own_query_entry = next(item for item in query_scored if item.unit_index == 0)
        inter1 = [
            _safe_cosine(memory_embedding, query_embedding),
            own_query_entry.bm25,
            len(missing) / max(len(query_tokens), 1),
            float(np.mean(missing_idfs)) if missing_idfs else 0.0,
        ]
        if reference_units:
            content_scored = raw_index.scored(turn["text"])
            competition = [item for item in content_scored if item.unit_index != 0]
            top = competition[0]
            own = next(item for item in content_scored if item.unit_index == 0)
            top_five = competition[:5]
            inter2 = [
                top.bm25,
                top.dense,
                top.rrf,
                own.rrf / max(top.rrf, 1e-12),
                float(
                    np.mean(
                        [
                            reference_units[item.unit_index - 1].session == session
                            for item in top_five
                        ]
                    )
                ),
                float(np.mean([item.dense for item in top_five])),
                self._entropy([item.rrf for item in competition]),
                top.rrf - (top_five[-1].rrf if len(top_five) == 5 else top.rrf),
                math.log(len(reference_units)),
            ]
        else:
            inter2 = [0.0] * 9
        raw_probe = query_scored[:10]
        full_own_rank = next(
            position
            for position, item in enumerate(query_scored, start=1)
            if item.unit_index == 0
        )
        own_rank = full_own_rank if full_own_rank <= 10 else 0
        inter3 = [
            float(own_rank),
            own_query_entry.rrf / max(raw_probe[0].rrf, 1e-12) if own_rank else 0.0,
            float(own_rank <= 10 and own_rank > 0),
            float(raw_probe[0].unit_index != 0),
            float(
                np.mean(
                    [
                        (
                            [current_unit, *reference_units][item.unit_index].session
                            == session
                        )
                        for item in raw_probe[:5]
                    ]
                )
            ),
        ]
        return np.asarray([*inter1, *inter2, *inter3], dtype=np.float32)

    def _interaction_features(
        self,
        conversation: dict,
        memory_embeddings: np.ndarray,
        query_embeddings: list[np.ndarray],
    ) -> np.ndarray:
        return np.asarray(
            [
                self._interaction_row(
                    conversation,
                    index,
                    memory_embeddings[index],
                    query_embeddings[index],
                )
                for index in range(len(conversation["turns"]))
            ],
            dtype=np.float32,
        )

    def _hot_state_features(self, conversation: dict, memory_embeddings: np.ndarray) -> np.ndarray:
        """E 族：只读取当前写入前的真实 raw 热索引状态。"""
        turns = conversation["turns"]
        rows: list[list[float]] = []
        for index, turn in enumerate(turns):
            previous = turns[:index]
            units = [
                IndexUnit(
                    turn_index=prior_index,
                    text=raw_unit(previous_turn, self.cfg),
                    session=int(previous_turn.get("session", 0)),
                )
                for prior_index, previous_turn in enumerate(previous)
            ]
            if not units:
                rows.append([0.0] * 10)
                continue
            hot = UnifiedIndex(units, self.embedder, self.cfg)
            probe = hot.probe(turn["text"], min(10, len(units)))
            dense = [item.dense for item in probe]
            rrf = [item.rrf for item in probe]
            top = probe[0]
            second = probe[1] if len(probe) > 1 else top
            same_session = np.mean(
                [
                    float(units[item.unit_index].session == int(turn.get("session", 0)))
                    for item in probe
                ]
            )
            local_embeddings = memory_embeddings[:index]
            similarities = local_embeddings @ memory_embeddings[index]
            rows.append(
                [
                    math.log1p(len(units)),
                    float(top.dense),
                    float(np.mean(dense)),
                    float(np.percentile(dense, 90)),
                    float(top.rrf),
                    float(top.rrf - second.rrf),
                    float(np.mean(rrf)),
                    float(np.max(similarities)),
                    float(np.mean(similarities)),
                    float(same_session),
                ]
            )
        return np.asarray(rows, dtype=np.float32)

    def raw_features(
        self, conversation: dict, qg: QueryGenerator
    ) -> dict[str, np.ndarray]:
        if not self._corpus_fitted:
            raise RuntimeError("FeatureBuilder 必须先拟合训练语料 BM25 统计。")
        attach_hypothetical_queries(conversation["turns"], qg)
        memory, memory_embeddings = self._memory_features(conversation)
        query_parts = [self._query_features(turn) for turn in conversation["turns"]]
        query = np.asarray([item[0] for item in query_parts], dtype=np.float32)
        query_embeddings = [item[1] for item in query_parts]
        interaction = self._interaction_features(
            conversation, memory_embeddings, query_embeddings
        )
        hot = self._hot_state_features(conversation, memory_embeddings)
        return {"memory": memory, "query": query, "interaction": interaction, "hot": hot}

    def fit(self, conversations: list[dict], qg: QueryGenerator) -> FeatureBuilder:
        if not conversations:
            raise ValueError("至少需要一段训练对话。")
        self._fit_corpus_statistics(conversations)
        raw = [self.raw_features(conversation, qg) for conversation in conversations]
        for name in ("memory", "query", "interaction", "hot"):
            values = np.vstack([item[name] for item in raw])
            self.means[name] = values.mean(axis=0).astype(np.float32)
            self.stds[name] = np.maximum(values.std(axis=0), 1e-6).astype(np.float32)
        self._fitted = True
        return self

    def _normalise(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("FeatureBuilder 必须先 fit。")
        return {
            name: ((value - self.means[name]) / self.stds[name]).astype(np.float32)
            for name, value in values.items()
        }

    def build(self, conversation: dict, qg: QueryGenerator) -> dict[str, np.ndarray]:
        return self._normalise(self.raw_features(conversation, qg))

    def build_current_stage2(
        self, conversation: dict, qg: QueryGenerator
    ) -> dict[str, np.ndarray]:
        """仅为当前 turn 计算 Q/I；历史 turn 只以 raw 形式充当固定参考索引。"""
        if not self._fitted:
            raise RuntimeError("FeatureBuilder 必须先 fit。")
        index = len(conversation["turns"]) - 1
        turn = conversation["turns"][index]
        if "_r2w_hq" not in turn:
            previous = conversation["turns"][index - 1]["text"] if index else ""
            turn["_r2w_hq"] = qg.generate(previous, turn["text"])
        memory, memory_embeddings = self._memory_features(conversation)
        query, query_embedding = self._query_features(turn)
        interaction = self._interaction_row(
            conversation, index, memory_embeddings[index], query_embedding
        )
        hot = self._hot_state_features(conversation, memory_embeddings)[index : index + 1]
        return self._normalise(
            {
                "memory": memory[index : index + 1],
                "query": query[None, :],
                "interaction": interaction[None, :],
                "hot": hot,
            }
        )

    def build_memory(self, conversation: dict) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureBuilder 必须先 fit。")
        memory, embeddings = self._memory_features(conversation)
        hot = self._hot_state_features(conversation, embeddings)
        return np.concatenate(
            (
                (memory - self.means["memory"]) / self.stds["memory"],
                (hot - self.means["hot"]) / self.stds["hot"],
            ),
            axis=1,
        ).astype(np.float32)

    def save(self, path: str | Path) -> None:
        if not self._fitted:
            raise RuntimeError("未 fit 的特征器不能保存。")
        joblib.dump(
            {
                "config_hash": self.cfg.hash(),
                "artifact_version": self.cfg.artifact_version,
                "repr_protocol_version": self.cfg.repr_protocol_version,
                "idf": self.idf,
                "bigram_idf": self.bigram_idf,
                "avgdl": self.avgdl,
                "means": self.means,
                "stds": self.stds,
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str | Path, cfg: R2WConfig, embedder, ner=None
    ) -> FeatureBuilder:
        state = joblib.load(path)
        if state["config_hash"] != cfg.hash():
            raise RuntimeError("特征器与 FINAL 配置哈希不一致。")
        instance = cls(cfg, embedder, ner=ner)
        instance.idf = state["idf"]
        instance.bigram_idf = state["bigram_idf"]
        instance.avgdl = float(state["avgdl"])
        instance.means = state["means"]
        instance.stds = state["stds"]
        instance._corpus_fitted = True
        instance._fitted = True
        return instance
