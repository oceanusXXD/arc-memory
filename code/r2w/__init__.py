"""Read2Write 核心算法包。"""

from .config import R2WConfig, default_config, load_config
from .embedding import EmbeddingBackend
from .llm import LLMClient
from .representations import MetaBlock, Repr


def train_pipeline(*args, **kwargs):
    from .pipeline_train import train_pipeline as implementation

    return implementation(*args, **kwargs)


def load_artifacts(*args, **kwargs):
    from .pipeline_infer import load_artifacts as implementation

    return implementation(*args, **kwargs)


def decide_turn(*args, **kwargs):
    from .pipeline_infer import decide_turn as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "EmbeddingBackend",
    "LLMClient",
    "R2WConfig",
    "MetaBlock",
    "Repr",
    "decide_turn",
    "default_config",
    "load_artifacts",
    "load_config",
    "train_pipeline",
]
