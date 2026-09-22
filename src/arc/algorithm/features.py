"""Train-fitted feature schema for the budget-conditioned pointer selector."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import torch


NUMERIC_FEATURES = (
    "position", "token_count", "relative_time_days", "retrieval_score",
    "bm25_rank", "dense_rank", "rrf_score", "cosine_similarity",
)
FEATURE_SCHEMA = "vectors"


class FeatureError(ValueError):
    pass


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _date(value: Any) -> datetime | None:
    text = str(value or "").strip()
    try:
        result = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        result = None
        for pattern in ("%I:%M %p on %d %B, %Y", "%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M"):
            try:
                result = datetime.strptime(text, pattern)
                break
            except ValueError:
                continue
        if result is None:
            return None
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def _metadata(source: Mapping[str, Any], query_time: Any = None) -> list[float | None]:
    age = _number(source.get("relative_time_days"))
    if age is None:
        now = _date(query_time if query_time is not None else source.get("query_time"))
        then = _date(source.get("time") or source.get("session_datetime"))
        if now is not None and then is not None:
            age = (now - then).total_seconds() / 86400.0
    return [
        _number(source.get("position", source.get("position_in_session"))),
        _number(source.get("token_count")),
        age,
        _number(source.get("retrieval_score", source.get("score"))),
        _number(source.get("bm25_rank")),
        _number(source.get("dense_rank")),
        _number(source.get("rrf_score", source.get("retrieval_score", source.get("score")))),
        _number(source.get("cosine_similarity")),
    ]


def _vector(value: Any, dimension: int, name: str) -> torch.Tensor:
    if value is None:
        raise FeatureError(f"missing_{name}")
    try:
        vector = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise FeatureError(f"invalid_{name}") from exc
    if vector.shape != (dimension,) or not torch.isfinite(vector).all():
        raise FeatureError(f"invalid_{name}")
    norm = torch.linalg.vector_norm(vector)
    if norm <= 0 or not torch.isfinite(norm):
        raise FeatureError(f"invalid_{name}")
    return vector / norm


@dataclass
class FeatureSchema:
    dimension: int
    means: list[float]
    scales: list[float]
    speakers: list[str]
    embedding_metadata: dict[str, Any]

    @property
    def input_size(self) -> int:
        """Dimension of ``[z_i; z_q; z_i*z_q; |z_i-z_q|; phi_i; b]``.

        The interaction terms are part of the paper specification.  The
        final scalar is the normalized deployment budget; keeping it in the
        feature tensor makes the selector checkpoint self-contained.
        """
        return 4 * self.dimension + 2 * len(NUMERIC_FEATURES) + len(self.speakers) + 2

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "schema": FEATURE_SCHEMA, "numeric_features": list(NUMERIC_FEATURES)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FeatureSchema:
        if value.get("schema") != FEATURE_SCHEMA or value.get("numeric_features") != list(NUMERIC_FEATURES):
            raise FeatureError("incompatible_feature_schema")
        schema = cls(**{key: value[key] for key in ("dimension", "means", "scales", "speakers", "embedding_metadata")})
        if schema.dimension <= 0 or len(schema.means) != len(NUMERIC_FEATURES) or len(schema.scales) != len(NUMERIC_FEATURES):
            raise FeatureError("invalid_feature_schema")
        if any(not math.isfinite(x) for x in schema.means + schema.scales) or any(x <= 0 for x in schema.scales):
            raise FeatureError("invalid_feature_normalization")
        if len(set(schema.speakers)) != len(schema.speakers):
            raise FeatureError("duplicate_speaker_categories")
        return schema

    @classmethod
    def fit(cls, records: Iterable[Mapping[str, Any]], dimension: int = 1024) -> FeatureSchema:
        # Reiterable JSONL inputs keep the frozen vectors for only one QA in
        # memory. Materialize only a genuinely one-shot input iterator.
        if iter(records) is records:
            records = list(records)
        metadata = None
        speaker_set = set()
        columns: list[list[float]] = [[] for _ in NUMERIC_FEATURES]
        for row in records:
            for source in row.get("sources") or []:
                if metadata is None:
                    metadata = source.get("embedding_metadata")
                    if not isinstance(metadata, dict) or metadata.get("dimension") != dimension:
                        raise FeatureError("missing_or_incompatible_embedding_metadata")
                    for key in ("model", "revision", "query_prompt_name"):
                        if not metadata.get(key):
                            raise FeatureError("incomplete_embedding_metadata")
                if source.get("speaker") not in (None, ""):
                    speaker_set.add(str(source["speaker"]))
                for column, value in zip(columns, _metadata(source, row.get("query_time"))):
                    if value is not None:
                        column.append(value)
        if metadata is None:
            raise FeatureError("no_training_sources")
        means = [sum(column) / len(column) if column else 0.0 for column in columns]
        scales = [math.sqrt(sum((value - mean) ** 2 for value in column) / len(column)) if column else 1.0
                  for column, mean in zip(columns, means)]
        speakers = sorted(speaker_set)
        schema = cls(dimension, means, [max(scale, 1e-6) for scale in scales], speakers, dict(metadata))
        for row in records:
            if row.get("sources"):
                schema.transform(None, list(row["sources"]), row.get("query_time"))
        return schema

    def transform(
        self,
        question: str | None,
        sources: list[Mapping[str, Any]],
        query_time: Any = None,
        budget: float = 4096.0,
        budget_scale: float = 4096.0,
        query_vector: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not sources:
            raise FeatureError("empty_sources_have_no_vector_features")
        # A write-stage candidate set has no query.  Retrieval rows may still
        # carry a cached query vector for diagnostics, so only consume it when
        # the caller explicitly supplies a query/question.  This prevents the
        # selector from learning a future query through stale cache fields.
        query_rows = [source for source in sources if source.get("query_vector") is not None]
        if query_vector is not None:
            query = _vector(query_vector, self.dimension, "query_vector")
        elif question not in (None, "") and query_rows:
            query = _vector(query_rows[0]["query_vector"], self.dimension, "query_vector")
        else:
            # Write-stage candidates have no query by construction.  The
            # candidate-set centroid is an observable, query-independent
            # context representation and keeps the feature dimensionality
            # identical between training and deployment.
            vectors = [_vector(source.get("embedding"), self.dimension, "source_vector") for source in sources]
            query = _vector(torch.stack(vectors).mean(dim=0), self.dimension, "query_vector")
        if question not in (None, "") or query_vector is not None:
            for source in query_rows:
                if question not in (None, "") and source.get("query_vector_question") not in (None, question):
                    raise FeatureError("query_vector_question_mismatch")
                if not torch.allclose(query, _vector(source["query_vector"], self.dimension, "query_vector")) and query_vector is None:
                    raise FeatureError("inconsistent_query_vectors")
        rows = []
        speaker_ids = {speaker: index + 1 for index, speaker in enumerate(self.speakers)}
        for source in sources:
            if source.get("embedding_metadata") != self.embedding_metadata:
                raise FeatureError("embedding_metadata_mismatch")
            vector = _vector(source.get("embedding"), self.dimension, "source_vector")
            values = _metadata(source, query_time)
            numerical = [(value - mean) / scale if value is not None else 0.0
                         for value, mean, scale in zip(values, self.means, self.scales)]
            missing = [float(value is None) for value in values]
            speaker = [0.0] * (len(self.speakers) + 1)
            speaker[speaker_ids.get(str(source.get("speaker") or ""), 0)] = 1.0
            interaction = vector * query
            difference = torch.abs(vector - query)
            normalized_budget = float(budget) / max(float(budget_scale), 1.0)
            rows.append(torch.cat((
                vector,
                query,
                interaction,
                difference,
                torch.tensor(numerical + missing + speaker + [normalized_budget], dtype=torch.float32),
            )))
        features = torch.stack(rows)
        if not torch.isfinite(features).all():
            raise FeatureError("non_finite_metadata_features")
        return features, query
