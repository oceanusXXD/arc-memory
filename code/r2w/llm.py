"""使用 FutureMem 同款阿里 OpenAI-compatible HTTP 接口的真实 LLM 客户端。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .config import R2WConfig, llm_runtime_config
from .runtime_usage import UsageLedger


class LLMProtocolError(RuntimeError):
    """远端 LLM 返回内容无法满足 R2W 的结构化契约。"""


class LLMClient:
    THINKING_ONLY_MODELS = ("kimi-k3", "kimi-k2.7-code")

    def __init__(self, cfg: R2WConfig):
        api_key, base_url, model = llm_runtime_config(cfg)
        if not base_url:
            raise RuntimeError("R2W 的真实 LLM 调用必须配置 llm_base_url。")
        if cfg.llm_api_timeout <= 0:
            raise ValueError("llm_api_timeout 必须为正数。")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        ordered_models = []
        for candidate in (model, *cfg.llm_failover_models):
            value = str(candidate).strip()
            if value and value not in ordered_models:
                ordered_models.append(value)
        self.models = tuple(ordered_models)
        self.model_index = 0
        self.initial_model = self.models[0]
        self.model = self.initial_model
        self.model_switch_trace: list[dict[str, object]] = []
        self.max_tokens = cfg.llm_max_tokens
        self.timeout = cfg.llm_api_timeout
        self.reasoning_effort = cfg.llm_reasoning_effort.strip() or None
        self._cache: dict[str, Any] = {}
        self.usage = UsageLedger()
        self.last_response_metadata: dict[str, Any] = {}

    def _complete(
        self,
        prompt: str,
        response_format: dict | None = None,
        *,
        purpose: str = "unspecified",
    ) -> str:
        while True:
            try:
                result = self._complete_once(prompt, response_format)
                break
            except urllib.error.HTTPError as exc:
                payload = self._error_payload(exc)
                reason = self._failover_reason(payload)
                if reason is None or self.model_index + 1 >= len(self.models):
                    raise RuntimeError(
                        f"LLM API 请求失败: {self.model} HTTP {exc.code}: "
                        f"{self._error_message(payload)}"
                    ) from exc
                previous = self.model
                self.model_index += 1
                self.model = self.models[self.model_index]
                self.model_switch_trace.append(
                    {
                        "from": previous,
                        "to": self.model,
                        "status": exc.code,
                        "reason": reason,
                    }
                )
                continue
        self.usage.add(purpose, result.get("usage"))
        choices = result.get("choices") or []
        if not choices:
            raise LLMProtocolError(f"LLM API 响应缺少 choices: {result.get('error')!r}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") in (None, "text")
            )
        if not isinstance(content, str) or not content.strip():
            raise LLMProtocolError("LLM API 返回空文本。")
        self.last_response_metadata = {
            "model": self.model,
            "finish_reason": choices[0].get("finish_reason"),
            "usage": result.get("usage") or {},
        }
        return content.strip()

    def _complete_once(self, prompt: str, response_format: dict | None) -> dict:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort, "exclude": True}
        elif not any(name in self.model.casefold() for name in self.THINKING_ONLY_MODELS):
            body["enable_thinking"] = False
        if response_format is not None:
            body["response_format"] = response_format
        endpoint = (
            "/chat/completions"
            if self.base_url.endswith("/v1")
            else "/v1/chat/completions"
        )
        request = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not isinstance(result, dict):
            raise LLMProtocolError("LLM API 响应根节点必须是 object。")
        return result

    @staticmethod
    def _error_payload(error: urllib.error.HTTPError) -> dict:
        try:
            raw = error.read().decode("utf-8", errors="replace")
            value = json.loads(raw)
            return value if isinstance(value, dict) else {"message": raw}
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {"message": str(error.reason or error)}

    @staticmethod
    def _error_object(payload: dict) -> dict:
        value = payload.get("error", payload)
        return value if isinstance(value, dict) else {"message": str(value)}

    @classmethod
    def _error_message(cls, payload: dict) -> str:
        value = cls._error_object(payload)
        return str(value.get("message") or value.get("code") or "unknown API error")

    @classmethod
    def _failover_reason(cls, payload: dict) -> str | None:
        value = cls._error_object(payload)
        code = str(value.get("code") or value.get("type") or "").lower()
        message = str(value.get("message") or "").lower()
        combined = f"{code} {message}"
        if any(
            term in combined
            for term in (
                "insufficient_quota",
                "quota exhausted",
                "quota_exhausted",
                "quota exceeded",
                "resource exhausted",
                "free quota",
            )
        ):
            return "quota_exhausted"
        if any(
            term in combined
            for term in (
                "model_not_found",
                "model not found",
                "model does not exist",
                "model unavailable",
                "model_unavailable",
                "not available for",
            )
        ):
            return "model_unavailable"
        if any(term in combined for term in ("rate_limit", "rate limit", "too many requests", "throttled")):
            return "rate_limited"
        return None

    def text(self, prompt: str, *, purpose: str = "unspecified") -> str:
        cache_key = f"text:{purpose}:{prompt}"
        if cache_key not in self._cache:
            self._cache[cache_key] = self._complete(prompt, purpose=purpose)
        return str(self._cache[cache_key])

    def json(self, prompt: str, *, purpose: str = "unspecified") -> dict[str, Any]:
        cache_key = f"json:{purpose}:{prompt}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        content = self._complete(
            prompt, {"type": "json_object"}, purpose=purpose
        )
        try:
            value = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMProtocolError(
                f"LLM 返回的不是 JSON object: {content[:200]!r}"
            ) from exc
        if not isinstance(value, dict):
            raise LLMProtocolError("LLM JSON 根节点必须是 object。")
        self._cache[cache_key] = value
        return value
