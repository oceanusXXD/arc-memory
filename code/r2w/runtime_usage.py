"""供应商实际返回的用量记录。

该模块只保存 API 响应显式给出的 token 数；缺失的字段保持为 ``null``，
从不以 ``word_count`` 或估算值替代。它用于运行报告，绝不进入 R2W 的
冻结算法成本函数。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping


@dataclass
class UsageCounter:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reported_calls: int = 0

    def add(self, usage: Mapping | None) -> None:
        self.calls += 1
        if not isinstance(usage, Mapping):
            return
        values = {}
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                values[name] = int(value)
        if not values:
            return
        self.reported_calls += 1
        self.prompt_tokens += values.get("prompt_tokens", 0)
        self.completion_tokens += values.get("completion_tokens", 0)
        self.total_tokens += values.get(
            "total_tokens", values.get("prompt_tokens", 0) + values.get("completion_tokens", 0)
        )

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "reported_calls": self.reported_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class UsageLedger:
    def __init__(self):
        self._counters: dict[str, UsageCounter] = defaultdict(UsageCounter)

    def add(self, purpose: str, usage: Mapping | None) -> None:
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError("usage purpose 必须是非空字符串。")
        self._counters[purpose].add(usage)

    def snapshot(self) -> dict[str, UsageCounter]:
        return {
            name: UsageCounter(**counter.to_dict())
            for name, counter in self._counters.items()
        }

    def delta(self, before: dict[str, UsageCounter]) -> dict:
        result = {}
        for name in sorted(set(before) | set(self._counters)):
            previous = before.get(name, UsageCounter())
            current = self._counters.get(name, UsageCounter())
            result[name] = {
                field: getattr(current, field) - getattr(previous, field)
                for field in (
                    "calls",
                    "reported_calls",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                )
            }
        return result

    def to_dict(self) -> dict:
        return {
            "unit": "provider_reported_tokens",
            "note": "Only fields returned by the provider are counted; missing usage is never estimated.",
            "by_purpose": {
                name: counter.to_dict() for name, counter in sorted(self._counters.items())
            },
        }


def provider_usage_payload(**components) -> dict:
    """序列化多个运行组件的用量账本。"""
    result = {
        "note": "Provider-reported token usage only. R2W algorithm costs remain word_count.",
        "components": {},
    }
    for name, value in components.items():
        ledger = getattr(value, "usage", None)
        if ledger is not None and hasattr(ledger, "to_dict"):
            result["components"][name] = ledger.to_dict()
        else:
            result["components"][name] = {
                "unit": "unavailable",
                "note": f"{name} backend does not expose provider usage.",
            }
    return result
