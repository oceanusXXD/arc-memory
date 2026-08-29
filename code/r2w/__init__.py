"""R2W v4 implementation for the current manuscript."""

from .config import R2WConfig, CORE_ACTIONS, EXTENDED_ACTIONS
from .actions import CandidateBundle, Representation, build_candidate_bundle, build_representation
from .retrieval import ParentRRFIndex
from .policy import Estimate, Stage2Estimate, early_reject, safe_stage2
from .replay import replay_memory
from .data import load_benchmark_conversations

__all__ = [
    "R2WConfig", "CORE_ACTIONS", "EXTENDED_ACTIONS", "CandidateBundle",
    "Representation", "build_candidate_bundle", "build_representation",
    "ParentRRFIndex", "Estimate", "Stage2Estimate", "early_reject",
    "safe_stage2", "replay_memory", "load_benchmark_conversations",
]
