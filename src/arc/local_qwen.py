"""Managed local Qwen3.5-4B serving for Claude Code.

This module starts a vLLM OpenAI server and exposes a loopback Anthropic
Messages bridge. It is intentionally independent from the circuit evaluator:
the bridge handles protocol conversion, while Claude Code still owns all MCP
tool execution and permissions.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from arc.vllm_proxy import VLLMProxy


@dataclass
class Qwen35ServeConfig:
    model: str = "Qwen/Qwen3.5-4B"
    served_model_name: str = "qwen-local"
    host: str = "127.0.0.1"
    port: int = 8000
    dtype: str = "bfloat16"
    max_model_len: int = 131072
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 1
    tensor_parallel_size: int = 1
    language_model_only: bool = True
    enable_prefix_caching: bool = True
    reasoning_parser: str = "qwen3"
    tool_call_parser: str = "qwen3_coder"
    vllm_bin: str = "vllm"
    api_key: str = ""
    startup_timeout_s: float = 600.0
    revision: str | None = None
    sampling_profile: str = "coding"
    thinking: str = "on"
    max_output_tokens: int = 32768
    max_num_batched_tokens: int = 8192
    log_file: str | None = None

    def __post_init__(self) -> None:
        if not 0.1 <= self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be between 0.1 and 1.0")
        if self.max_model_len <= 0 or self.max_num_seqs <= 0:
            raise ValueError("max_model_len and max_num_seqs must be positive")
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if not math.isfinite(self.startup_timeout_s) or self.startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be finite and positive")
        if not 0 < self.max_output_tokens < self.max_model_len:
            raise ValueError("max_output_tokens must be positive and smaller than max_model_len")
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError("max_num_batched_tokens must be at least max_num_seqs")
        if self.sampling_profile not in {"coding", "general", "passthrough"}:
            raise ValueError("unknown sampling_profile")
        if self.thinking not in {"on", "off", "auto"}:
            raise ValueError("thinking must be on, off or auto")
        if not self.model.strip() or not self.served_model_name.strip():
            raise ValueError("model names cannot be empty")

    def sampling_parameters(self) -> dict[str, Any]:
        if self.sampling_profile == "passthrough":
            return {}
        if self.thinking == "off":
            return {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                    "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
        return {"temperature": 0.7,
                "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                "presence_penalty": 0.0 if self.sampling_profile == "coding" else 1.5,
                "repetition_penalty": 1.0}

    @property
    def backend_base_url(self) -> str:
        host = "127.0.0.1" if self.host == "0.0.0.0" else self.host
        return f"http://{host}:{self.port}/v1"

    def command(self) -> list[str]:
        """Build a reproducible vLLM command without shell interpolation."""
        executable = shutil.which(self.vllm_bin) or self.vllm_bin
        command = [executable, "serve", self.model,
                   "--host", self.host,
                   "--port", str(self.port),
                   "--served-model-name", self.served_model_name,
                   "--dtype", self.dtype,
                   "--max-model-len", str(self.max_model_len),
                   "--gpu-memory-utilization", str(self.gpu_memory_utilization),
                   "--max-num-seqs", str(self.max_num_seqs),
                   "--max-num-batched-tokens", str(self.max_num_batched_tokens),
                   "--enable-chunked-prefill",
                   "--enable-prompt-tokens-details",
                   "--kv-cache-dtype", "auto",
                   "--tensor-parallel-size", str(self.tensor_parallel_size),
                   "--reasoning-parser", self.reasoning_parser,
                   "--enable-auto-tool-choice",
                   "--tool-call-parser", self.tool_call_parser]
        if self.language_model_only:
            command.append("--language-model-only")
        command.append("--enable-prefix-caching" if self.enable_prefix_caching
                       else "--no-enable-prefix-caching")
        if self.revision:
            command.extend(["--revision", self.revision, "--tokenizer-revision", self.revision])
        # Credentials belong in the child environment, not process argv.
        return command

    def descriptor(self) -> dict[str, Any]:
        value = asdict(self)
        value["api_key"] = "[configured]" if self.api_key else ""
        value["backend_base_url"] = self.backend_base_url
        value["command"] = self.command()
        value["sampling_parameters"] = self.sampling_parameters()
        return value


def claude_environment(base_url: str, model: str, *, token: str = "local-vllm",
                       context_length: int = 131072, max_output_tokens: int = 32768) -> dict[str, str]:
    # Unknown model IDs use Claude Code's generic context accounting. Compact
    # early enough to reserve output plus a margin for tokenizer differences.
    window = min(200000, max(100000, context_length))
    pct = max(1, min(80, int(100 * (context_length - max_output_tokens - 8192) / window)))
    env = {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_API_KEY": token,
        "ANTHROPIC_AUTH_TOKEN": token,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_SMALL_FAST_MODEL": model,
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(max_output_tokens),
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": str(window),
        "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": str(pct),
        "API_TIMEOUT_MS": "1800000",
        "ENABLE_TOOL_SEARCH": "false",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
    }
    for tier in ("HAIKU", "SONNET", "OPUS", "FABLE"):
        env[f"ANTHROPIC_DEFAULT_{tier}_MODEL"] = model
    return env


class LocalQwenServer:
    """Own a vLLM process and an Anthropic bridge for one Claude session."""

    def __init__(self, config: Qwen35ServeConfig | None = None,
                 *, bridge_host: str = "127.0.0.1", bridge_port: int = 0,
                 bridge_api_key: str = "local-vllm"):
        self.config = config or Qwen35ServeConfig()
        if bridge_host not in {"127.0.0.1", "localhost"}:
            raise ValueError("the unauthenticated bridge must bind to loopback")
        if not 0 <= bridge_port <= 65535:
            raise ValueError("bridge_port must be between 0 and 65535")
        self.bridge_host = bridge_host
        self.bridge_port = bridge_port
        self.bridge_api_key = bridge_api_key
        self.process: subprocess.Popen[str] | None = None
        self.proxy: VLLMProxy | None = None
        self._log_tail: deque[str] = deque(maxlen=80)
        self._log_thread: threading.Thread | None = None
        self._log_file: Any = None

    def _drain_logs(self, process: subprocess.Popen[str]) -> None:
        if process.stdout:
            with process.stdout:
                for line in process.stdout:
                    self._log_tail.append(line[-2000:])
                    if self._log_file:
                        self._log_file.write(line)
                        self._log_file.flush()

    @property
    def base_url(self) -> str:
        if not self.proxy:
            raise RuntimeError("local Qwen server has not been started")
        return self.proxy.base_url

    @property
    def model(self) -> str:
        return self.config.served_model_name

    def _ready(self) -> bool:
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = "Bearer " + self.config.api_key
        try:
            request = urllib.request.Request(self.config.backend_base_url + "/models", headers=headers)
            with urllib.request.urlopen(request, timeout=2) as response:
                data = json.load(response)
                return any(item.get("id") == self.model for item in data.get("data", []))
        except (OSError, ValueError, urllib.error.URLError):
            pass
        return False

    def start(self) -> "LocalQwenServer":
        if self.process and self.process.poll() is None:
            return self
        # Never mistake an unrelated server already using this port for ours.
        with socket.socket() as probe:
            probe.bind((self.config.host, self.config.port))
        command = self.config.command()
        env = os.environ.copy()
        if self.config.api_key:
            env["VLLM_API_KEY"] = self.config.api_key
        if self.config.log_file:
            path = Path(self.config.log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = path.open('a', encoding='utf-8')
        try:
            self.process = subprocess.Popen(
                command, env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                start_new_session=(os.name == "posix"),
            )
        except BaseException:
            if self._log_file:
                self._log_file.close()
                self._log_file = None
            raise
        self._log_tail.clear()
        self._log_thread = threading.Thread(target=self._drain_logs, args=(self.process,), daemon=True)
        self._log_thread.start()
        deadline = time.monotonic() + self.config.startup_timeout_s
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    self._log_thread.join(timeout=2)
                    raise RuntimeError(f"vLLM exited with code {self.process.returncode}: " + "".join(self._log_tail)[-4000:])
                if self._ready() and self.process.poll() is None:
                    self.proxy = VLLMProxy(
                        self.config.backend_base_url, model=self.model,
                        api_key=self.config.api_key, host=self.bridge_host, port=self.bridge_port,
                        sampling=self.config.sampling_parameters(), thinking=self.config.thinking,
                    ).start()
                    return self
                time.sleep(0.5)
            raise TimeoutError(f"vLLM did not become ready within {self.config.startup_timeout_s:.0f}s: "
                               + "".join(self._log_tail)[-4000:])
        except BaseException:
            self.stop()
            raise

    def claude_env(self) -> dict[str, str]:
        if not self.proxy:
            raise RuntimeError("local Qwen server has not been started")
        return claude_environment(self.proxy.base_url, self.model, token=self.bridge_api_key,
                                  context_length=self.config.max_model_len,
                                  max_output_tokens=self.config.max_output_tokens)

    def descriptor(self) -> dict[str, Any]:
        return {
            "backend": self.config.descriptor(),
            "bridge_base_url": self.proxy.base_url if self.proxy else None,
            "bridge_messages_url": self.proxy.messages_url if self.proxy else None,
            "model": self.model,
            "running": bool(self.process and self.process.poll() is None),
        }

    def stop(self) -> None:
        if self.proxy:
            self.proxy.close()
            self.proxy = None
        process = self.process
        self.process = None
        if process:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=5)
        if self._log_thread:
            self._log_thread.join(timeout=2)
            self._log_thread = None
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def __enter__(self) -> "LocalQwenServer":
        return self.start()

    def __exit__(self, _type: Any, _value: Any, _tb: Any) -> None:
        self.stop()


def config_from_args(args: Any) -> Qwen35ServeConfig:
    """Create a config from argparse values while preserving safe defaults."""
    return Qwen35ServeConfig(
        model=args.model,
        served_model_name=args.served_model_name,
        host=args.host,
        port=args.port,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        language_model_only=True,
        enable_prefix_caching=not args.no_prefix_caching,
        reasoning_parser=args.reasoning_parser,
        tool_call_parser=args.tool_call_parser,
        vllm_bin=args.vllm_bin,
        api_key=os.environ.get('VLLM_API_KEY', ''),
        startup_timeout_s=args.startup_timeout,
        revision=args.revision,
        sampling_profile=args.sampling_profile,
        thinking=args.thinking,
        max_output_tokens=args.max_output_tokens,
        max_num_batched_tokens=args.max_num_batched_tokens,
        log_file=args.log_file,
    )


def wait_for_signal(server: LocalQwenServer) -> None:
    """Block until SIGINT/SIGTERM for the foreground ``serve-local`` command."""
    done = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal done
        done = True

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while not done:
            if server.process is None or server.process.poll() is not None:
                raise RuntimeError("vLLM stopped: " + "".join(server._log_tail)[-4000:])
            time.sleep(0.5)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
