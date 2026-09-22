"""Anthropic Messages facade backed by a vLLM OpenAI-compatible server.

The Claude Code CLI speaks Anthropic Messages; this adapter uses vLLM's
OpenAI Chat Completions endpoint. It translates messages, tool blocks and
streaming events, but never executes tools. Claude Code remains responsible
for MCP, Bash, Read and Edit.
"""
from __future__ import annotations

import json
import queue
import threading
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterable


MAX_BODY_BYTES = 16 * 1024 * 1024


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _text(value: Any) -> str:
    """Flatten Anthropic text/thinking/tool-result blocks for OpenAI."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: list[str] = []
        for block in value:
            if not isinstance(block, dict):
                chunks.append(str(block))
                continue
            kind = block.get("type")
            if kind == "text":
                chunks.append(str(block.get("text", "")))
            elif kind == "thinking":
                continue
            elif kind == "tool_result":
                chunks.append(_text(block.get("content", "")))
            else:
                raise ValueError(f"unsupported content block in text-only bridge: {kind}")
        return "".join(chunks)
    if value is None:
        return ""
    return _json(value)


def _thinking_from_text(value: str) -> tuple[str, str]:
    """Fallback parser for backends that put thinking in the text field."""
    start, end = value.find("<think>"), value.find("</think>")
    if start >= 0 and end > start:
        thinking = value[start + len("<think>"):end]
        return thinking, value[:start] + value[end + len("</think>"):]
    return "", value


def _anthropic_to_openai(system: Any, items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system_parts = [_text(system)] if system else []
    items = list(items or [])
    for item in items:
        if item.get("role") == "system":
            system_parts.append(_text(item.get("content", "")))
    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

    for item in items or []:
        role = item.get("role", "user")
        if role == "system":
            continue
        content = item.get("content", "")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            messages.append({"role": role, "content": _text(content)})
            continue

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        thinking_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                text_parts.append(str(block))
                continue
            kind = block.get("type")
            if kind == "text":
                text_parts.append(str(block.get("text", "")))
            elif kind == "thinking":
                thinking = str(block.get("thinking", ""))
                if thinking:
                    thinking_parts.append(thinking)
            elif kind == "tool_use":
                tool_calls.append({
                    "id": str(block.get("id", "call-unknown")),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name", "")),
                        "arguments": _json(block.get("input", {})),
                    },
                })
            elif kind == "tool_result":
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": str(block.get("tool_use_id", "call-unknown")),
                    "content": _text(block.get("content", "")),
                })
            else:
                raise ValueError(f"unsupported content block in text-only bridge: {kind}")

        if role == "user" and tool_results:
            messages.extend(tool_results)
            if text_parts:
                messages.append({"role": "user", "content": "".join(text_parts)})
            continue

        message: dict[str, Any] = {"role": role, "content": "".join(text_parts)}
        if not text_parts and tool_calls:
            message["content"] = None
        if tool_calls:
            message["tool_calls"] = tool_calls
        if thinking_parts and role == "assistant":
            # Let the official template decide which reasoning belongs to the
            # active tool turn and which older reasoning should be stripped.
            message["reasoning_content"] = "".join(thinking_parts)
        messages.append(message)
    return messages


def _tools_to_openai(tools: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        # Accept both Anthropic's flat tool schema and an already converted
        # OpenAI schema so the bridge can be used with replay fixtures.
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            result.append(tool)
            continue
        result.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object"}),
            },
        })
    return result


def _tool_choice(value: Any) -> Any:
    if value in (None, "auto", "none", "required"):
        return value
    if value == "any":
        return "required"
    if isinstance(value, dict) and value.get("type") == "tool":
        return {"type": "function", "function": {"name": value.get("name", "")}}
    if isinstance(value, dict) and value.get("type") in {"auto", "any", "none"}:
        return {"auto": "auto", "any": "required", "none": "none"}[value["type"]]
    return value


def _parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError):
        raise RuntimeError("backend returned malformed tool arguments") from None
    if not isinstance(parsed, dict):
        raise RuntimeError("backend tool arguments must be a JSON object")
    return parsed


def _usage(value: Any) -> dict[str, int]:
    value = value if isinstance(value, dict) else {}
    details = value.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens", value.get("cache_read_input_tokens", 0)) or 0)
    total = int(value.get("prompt_tokens", value.get("input_tokens", 0)) or 0)
    return {
        "input_tokens": max(0, total - cached) if "prompt_tokens" in value else total,
        "output_tokens": int(value.get("completion_tokens", value.get("output_tokens", 0)) or 0),
        "cache_creation_input_tokens": int(value.get("cache_creation_input_tokens", 0) or 0),
        "cache_read_input_tokens": cached,
    }


def _finish_reason(value: Any, has_tools: bool = False) -> str:
    if value in ("length", "max_tokens"):
        return "max_tokens"
    if has_tools or value == "tool_calls":
        return "tool_use"
    return "end_turn"


def _completion_message(result: dict[str, Any], requested_model: str) -> dict[str, Any]:
    if result.get("error") or not result.get("choices"):
        raise RuntimeError("backend returned no completion")
    choice = (result.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    content = message.get("content") or ""
    if not reasoning and isinstance(content, str):
        reasoning, content = _thinking_from_text(content)
    blocks: list[dict[str, Any]] = []
    if reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning, "signature": ""})
    if content:
        blocks.append({"type": "text", "text": content})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        blocks.append({
            "type": "tool_use",
            "id": call.get("id") or "call-unknown",
            "name": function.get("name", ""),
            "input": _parse_arguments(function.get("arguments", "{}")),
        })
    finish = _finish_reason(choice.get("finish_reason"), bool(message.get("tool_calls")))
    return {
        "id": result.get("id", "vllm-anthropic"),
        "type": "message",
        "role": "assistant",
        "model": result.get("model", requested_model),
        "content": blocks,
        "stop_reason": finish,
        "stop_sequence": None,
        "usage": _usage(result.get("usage")),
    }


def _sse(event: dict[str, Any]) -> bytes:
    return (f"event: {event['type']}\ndata: " + _json(event) + "\n\n").encode("utf-8")


def _stream_frames(response: Any) -> Iterable[dict[str, Any]]:
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        if not payload:
            continue
        try:
            frame = json.loads(payload)
        except ValueError:
            raise RuntimeError("backend returned malformed SSE JSON") from None
        if isinstance(frame, dict):
            if frame.get("error"):
                raise RuntimeError("backend stream returned an error")
            yield frame


class _Handler(BaseHTTPRequestHandler):
    upstream = ""
    model = ""
    api_key = ""
    max_body_bytes = MAX_BODY_BYTES
    sampling: dict[str, Any] = {}
    thinking = "auto"

    def log_message(self, *_args: Any) -> None:
        # Claude Code output is already captured by the parent trace. Avoid
        # writing request bodies or credentials into a second log stream.
        pass

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0") or 0)
        if length <= 0 or length > self.max_body_bytes:
            raise ValueError("request body must be between 1 byte and 16 MiB")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _write_json(self, status: int, value: dict[str, Any]) -> None:
        data = _json(value).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _upstream_request(self, body: dict[str, Any]) -> urllib.request.Request:
        headers = {
            "content-type": "application/json",
            "accept": "text/event-stream" if body.get("stream") else "application/json",
        }
        if self.api_key:
            headers["authorization"] = "Bearer " + self.api_key
        return urllib.request.Request(
            self.upstream.rstrip("/") + "/chat/completions",
            data=_json(body).encode("utf-8"), headers=headers, method="POST",
        )

    def _openai_body(self, raw: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model or raw.get("model"),
            "messages": _anthropic_to_openai(raw.get("system"), raw.get("messages", [])),
            "max_tokens": int(raw.get("max_tokens", 1024)),
            "stream": bool(raw.get("stream")),
        }
        if raw.get("tools"):
            body["tools"] = _tools_to_openai(raw["tools"])
        if raw.get("tool_choice") is not None:
            body["tool_choice"] = _tool_choice(raw["tool_choice"])
            if isinstance(raw["tool_choice"], dict) and raw["tool_choice"].get("disable_parallel_tool_use"):
                body["parallel_tool_calls"] = False
        for key in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "repetition_penalty"):
            if key in raw:
                body[key] = raw[key]
        if "stop_sequences" in raw:
            body["stop"] = raw["stop_sequences"]
        body.update(self.sampling)
        thinking = raw.get("thinking") or {}
        if self.thinking != "auto":
            body["chat_template_kwargs"] = {"enable_thinking": self.thinking == "on"}
        elif thinking.get("type") in {"enabled", "adaptive", "disabled"}:
            body["chat_template_kwargs"] = {"enable_thinking": thinking["type"] != "disabled"}
        # vLLM returns usage in the final stream frame when this option is
        # accepted. Unknown extra fields are ignored by older versions.
        if body["stream"]:
            body["stream_options"] = {"include_usage": True}
        return body

    def _count_tokens(self, raw: dict[str, Any]) -> int:
        body = self._openai_body(raw)
        payload = {key: body[key] for key in ("model", "messages", "tools", "chat_template_kwargs")
                   if key in body}
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = "Bearer " + self.api_key
        # vLLM's tokenizer endpoint is outside /v1 and applies the same chat template.
        root = self.upstream.removesuffix("/v1")
        request = urllib.request.Request(root + "/tokenize", data=_json(payload).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            return int(json.load(response)["count"])

    def _error(self, status: int, message: str, *, streaming: bool = False) -> None:
        value = {"type": "error", "error": {
            "type": "invalid_request_error" if status == 400 else "api_error", "message": message}}
        if streaming:
            try:
                self.wfile.write(_sse(value))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self._write_json(status, value)

    def _send_stream(self, response: Any, body: dict[str, Any]) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream; charset=utf-8")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(_sse({
            "type": "message_start",
            "message": {
                "id": "msg_" + uuid.uuid4().hex,
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }))
        self.wfile.flush()

        # Reading a socket can block during a long thinking interval. A small
        # queue lets this thread emit Anthropic ping events without buffering
        # the whole response or making Claude Code think the server died.
        frames: queue.Queue[Any] = queue.Queue(maxsize=32)
        cancelled = threading.Event()

        def publish(frame: Any) -> None:
            while not cancelled.is_set():
                try:
                    frames.put(frame, timeout=0.2)
                    return
                except queue.Full:
                    pass

        def reader() -> None:
            try:
                for frame in _stream_frames(response):
                    if cancelled.is_set():
                        break
                    publish(frame)
            except Exception as exc:  # pragma: no cover - network dependent
                publish(exc)
            finally:
                publish(None)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        tools: dict[int, dict[str, Any]] = {}
        next_index = 0
        active_type = None
        finish_reason = None
        usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        try:
            while True:
                try:
                    frame = frames.get(timeout=10)
                except queue.Empty:
                    self.wfile.write(_sse({"type": "ping"}))
                    self.wfile.flush()
                    continue
                if frame is None:
                    break
                if isinstance(frame, Exception):
                    raise frame
                choice = (frame.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if frame.get("usage"):
                    usage = _usage(frame["usage"])
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

                for kind, fragment in (("thinking", delta.get("reasoning_content") or delta.get("reasoning")),
                                       ("text", delta.get("content"))):
                    if not fragment:
                        continue
                    if active_type != kind:
                        if active_type:
                            self.wfile.write(_sse({"type": "content_block_stop", "index": next_index}))
                            next_index += 1
                        block = {"type": kind, kind: ""}
                        if kind == "thinking":
                            block["signature"] = ""
                        self.wfile.write(_sse({"type": "content_block_start", "index": next_index,
                                              "content_block": block}))
                        active_type = kind
                    self.wfile.write(_sse({"type": "content_block_delta", "index": next_index,
                                          "delta": {"type": kind + "_delta", kind: fragment}}))

                # Tool headers and JSON can span several OpenAI frames. Emit
                # tools only after validating the complete call, before CLI execution.
                for call in delta.get("tool_calls") or []:
                    state = tools.setdefault(int(call.get("index", 0)), {"id": "", "name": "", "arguments": ""})
                    function = call.get("function") or {}
                    if call.get("id"):
                        state["id"] = call["id"]
                    state["name"] += function.get("name") or ""
                    state["arguments"] += function.get("arguments") or ""
                self.wfile.flush()
            if finish_reason is None:
                raise RuntimeError("backend stream ended before a finish_reason")
            parsed_tools = []
            for call in tools.values():
                if not call["id"] or not call["name"]:
                    raise RuntimeError("backend returned an incomplete tool header")
                parsed_tools.append((call, _parse_arguments(call["arguments"])))
            if active_type:
                self.wfile.write(_sse({"type": "content_block_stop", "index": next_index}))
                next_index += 1
            for call, arguments in parsed_tools:
                self.wfile.write(_sse({"type": "content_block_start", "index": next_index,
                                      "content_block": {"type": "tool_use", "id": call["id"],
                                                        "name": call["name"], "input": {}}}))
                self.wfile.write(_sse({"type": "content_block_delta", "index": next_index,
                                      "delta": {"type": "input_json_delta", "partial_json": _json(arguments)}}))
                self.wfile.write(_sse({"type": "content_block_stop", "index": next_index}))
                next_index += 1
        finally:
            cancelled.set()
        self.wfile.write(_sse({
            "type": "message_delta",
            "delta": {"stop_reason": _finish_reason(finish_reason, bool(tools)), "stop_sequence": None},
            "usage": usage,
        }))
        self.wfile.write(_sse({"type": "message_stop"}))
        self.wfile.flush()

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in {"/health", "/v1/health"}:
            self._write_json(200, {"ok": True, "backend": self.upstream})
            return
        if path == "/v1/models":
            self._write_json(200, {
                "object": "list",
                "data": [{"id": self.model, "object": "model", "owned_by": "local"}],
            })
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/v1/messages/count_tokens":
            try:
                raw = self._read_json()
                self._write_json(200, {"input_tokens": self._count_tokens(raw)})
            except ValueError as exc:
                self._error(400, str(exc))
            except Exception as exc:
                self._error(502, str(exc))
            return
        if path != "/v1/messages":
            self.send_error(404)
            return
        headers_sent = False
        try:
            raw = self._read_json()
            body = self._openai_body(raw)
        except (ValueError, TypeError, KeyError) as exc:
            self._error(400, str(exc))
            return
        try:
            request = self._upstream_request(body)
            with urllib.request.urlopen(request, timeout=1800) as response:
                if body["stream"]:
                    headers_sent = True
                    self._send_stream(response, body)
                else:
                    result = json.loads(response.read())
                    self._write_json(200, _completion_message(result, body["model"]))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:4000]
            self._error(exc.code if exc.code in {400, 401, 403, 404, 413, 429} else 502,
                        f"vLLM HTTP {exc.code}: {detail}", streaming=headers_sent)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            self._error(502, str(exc), streaming=headers_sent)


class VLLMProxy:
    """Loopback Anthropic endpoint for one vLLM OpenAI server."""

    def __init__(self, upstream: str, model: str = "", api_key: str | None = None,
                 host: str = "127.0.0.1", port: int = 0,
                 sampling: dict[str, Any] | None = None, thinking: str = "auto"):
        if host not in {"127.0.0.1", "localhost"}:
            raise ValueError("the unauthenticated bridge must bind to loopback")
        handler = type("VLLMProxyHandler", (_Handler,), {
            "upstream": upstream.rstrip("/"),
            "model": model,
            "api_key": api_key or "",
            "sampling": dict(sampling or {}),
            "thinking": thinking,
        })
        self.server = ThreadingHTTPServer((host, port), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        # Claude Code appends /v1/messages itself.
        return f"http://127.0.0.1:{self.server.server_port}"

    @property
    def messages_url(self) -> str:
        return self.base_url + "/v1/messages"

    def start(self) -> "VLLMProxy":
        self.thread.start()
        return self

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
