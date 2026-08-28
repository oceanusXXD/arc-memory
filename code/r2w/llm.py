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
        self.model = model
        self.max_tokens = cfg.llm_max_tokens
        self.seed = cfg.random_seed
        self.timeout = cfg.llm_api_timeout
        self._cache: dict[str, Any] = {}
        self.usage = UsageLedger()

    def _complete(
        self,
        prompt: str,
        response_format: dict | None = None,
        *,
        purpose: str = "unspecified",
    ) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        }
        if not any(name in self.model.casefold() for name in self.THINKING_ONLY_MODELS):
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
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode("utf-8", errors="replace"))
                error = payload.get("error", payload)
                message = (
                    error.get("message", error) if isinstance(error, dict) else error
                )
            except (OSError, UnicodeError, json.JSONDecodeError):
                message = exc.reason
            raise RuntimeError(f"LLM API 请求失败: HTTP {exc.code}: {message}") from exc
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
        self.usage.add(purpose, result.get("usage"))
        return content.strip()

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
