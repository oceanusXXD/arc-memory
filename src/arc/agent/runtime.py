from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from arc.agent.memory import MemoryPacket
from arc.agent.reader import CompletionError, ClaudeCodeClient


@dataclass(frozen=True)
class AgentResult:
    outcome: str
    usage: dict[str, Any]
    raw: Any
    stdout: str
    stderr: str


def build_agent_request(task: Any, memory_text: str, initial_state: Any | None = None) -> str:
    task_text = task if isinstance(task, str) else json.dumps(task, ensure_ascii=False, sort_keys=True)
    lines = ["You are the fixed target agent for an arc run.",
             "Complete the task using only the injected task-scoped memory.",
             "Do not use hidden labels, reference answers, or external memory.",
             "", "Task:", task_text.strip(), "",
             "Injected memory:", memory_text.strip() if memory_text.strip() else "(empty)"]
    if initial_state is not None:
        state = initial_state if isinstance(initial_state, str) else json.dumps(initial_state, ensure_ascii=False, sort_keys=True)
        lines.extend(["", "Initial state:", state.strip()])
    return "\n".join(lines)


def question_answer_task(question: str) -> str:
    return "\n".join(["Answer the question using only the injected memory packet.",
                         "Return only a short answer. If unsupported, return exactly 'No information available'.",
                         f"Question: {question}", "Answer:"])


def _usage(completion: Any, provider: str) -> dict[str, Any]:
    return {"agent": provider, "provider": provider, "model": completion.model,
            "prompt_tokens": completion.prompt_tokens, "completion_tokens": completion.completion_tokens,
            "total_tokens": completion.total_tokens, "latency_ms": completion.latency_ms,
            "request_count": completion.request_count, "usage_complete": completion.usage_complete,
            "finish_reason": ((completion.raw.get("choices") or [{}])[0]).get("finish_reason")}


def run_agent(task: Any, memory_text: str, initial_state: Any | None = None,
              model_id: str | None = None, config: dict[str, Any] | None = None) -> AgentResult:
    settings = config or {}
    agent = settings.get("agent") or {}
    provider = str(agent.get("provider") or "claude_code")
    client = ClaudeCodeClient(settings)
    completion = client.complete("agent", build_agent_request(task, memory_text, initial_state),
                                 max_tokens=int(agent.get("max_tokens") or 2048),
                                 temperature=float(agent.get("temperature", 0.0)), model=model_id)
    usage = _usage(completion, client.provider)
    if ((completion.raw.get("choices") or [{}])[0]).get("finish_reason") == "length":
        raise CompletionError("agent output truncated at token limit", usage)
    if not completion.text:
        raise CompletionError("agent returned empty content", usage)
    return AgentResult(completion.text, usage, completion.raw, completion.text, "")


def agent_usage_row(sample_id: str, qa_id: str, phase: str, usage: dict[str, Any]) -> dict[str, Any]:
    return {"sample_id": str(sample_id), "qa_id": str(qa_id), "phase": phase, "role": "agent",
            "model": usage.get("model"), "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"), "total_tokens": usage.get("total_tokens"),
            "observable_usage": usage, "latency_ms": usage.get("latency_ms"),
            "usage_complete": usage.get("usage_complete", False), "request_count": usage.get("request_count"),
            "status": "ok"}


class Agent:
    def __init__(self, config: dict[str, Any], memory_stage: Any):
        self.config = config
        self.memory_stage = memory_stage

    def prepare_memory(self, request: Any) -> MemoryPacket:
        from arc.agent.memory import prepare_memory
        return prepare_memory(self.config, request, self.memory_stage)

    def run(self, task: Any, packet: MemoryPacket, initial_state: Any = None) -> AgentResult:
        return run_agent(task, packet.text, initial_state=initial_state, config=self.config)
