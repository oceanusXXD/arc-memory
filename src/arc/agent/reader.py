"""Native Claude Code calls with explicit credentials and per-call audit files."""
from __future__ import annotations

import json
import hashlib
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from .relay import MessagesRelay


class CompletionError(RuntimeError):
    def __init__(self, message: str, usage: Mapping[str, Any] | None = None, raw: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.usage = dict(usage or {})
        self.raw = dict(raw or {})


@dataclass(frozen=True)
class Completion:
    text: str
    role: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: int
    raw: dict[str, Any]
    request_count: int = 1
    usage_complete: bool = True
    audit_path: str | None = None

    def cost_row(self, sample_id: str, qa_id: str, phase: str) -> dict[str, Any]:
        return {'sample_id': str(sample_id), 'qa_id': str(qa_id), 'phase': phase,
                'role': self.role, 'model': self.model, 'prompt_tokens': self.prompt_tokens,
                'completion_tokens': self.completion_tokens, 'total_tokens': self.total_tokens,
                'latency_ms': self.latency_ms, 'status': 'ok', 'request_count': self.request_count,
                'usage_complete': self.usage_complete, 'audit_path': self.audit_path,
                'finish_reason': self.raw.get('stop_reason') or ((self.raw.get('choices') or [{}])[0]).get('finish_reason')}


def _integer(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _redact(value: Any, env: Mapping[str, str]) -> Any:
    secrets = [v for k, v in env.items() if len(v) >= 8 and
               any(part in k.upper() for part in ('KEY', 'TOKEN', 'PASSWORD', 'SECRET'))]
    text = json.dumps(value, ensure_ascii=False, default=str)
    for secret in secrets:
        text = text.replace(secret, '<redacted>')
    return json.loads(text)


class ClaudeCodeClient:
    provider = 'claude_code'

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.spec = dict(config.get('claude_code') or {})
        self.command = str(self.spec.get('command') or os.environ.get('CLAUDE_CODE_COMMAND') or 'claude')
        self.timeout = float(self.spec.get('timeout_seconds') or 900)
        self.model = str(self.spec.get('model') or os.environ.get('ANTHROPIC_MODEL') or 'Qwen/Qwen3.5-4B')
        self.extra_args = [str(value) for value in self.spec.get('extra_args') or []]
        self.thinking = bool(self.spec.get('thinking', True))
        self.thinking_budget_tokens = int(self.spec.get('thinking_budget_tokens') or 1024)

    def environment(self, model: str | None = None) -> dict[str, str]:
        env = os.environ.copy()
        key_env = str(self.spec.get('api_key_env') or 'ANTHROPIC_API_KEY')
        # Only the explicitly configured credential source is mapped.
        if not env.get('ANTHROPIC_API_KEY') and not env.get('ANTHROPIC_AUTH_TOKEN') and env.get(key_env):
            env['ANTHROPIC_API_KEY'] = env[key_env]
        if self.spec.get('base_url') and not env.get('ANTHROPIC_BASE_URL'):
            env['ANTHROPIC_BASE_URL'] = str(self.spec['base_url'])
        if model:
            env['ANTHROPIC_MODEL'] = model
            env['ANTHROPIC_SMALL_FAST_MODEL'] = model
            for tier in ('HAIKU', 'SONNET', 'OPUS', 'FABLE'):
                env[f'ANTHROPIC_DEFAULT_{tier}_MODEL'] = model
        return env

    def credential_status(self) -> dict[str, Any]:
        env = self.environment()
        return {'has_api_key': bool(env.get('ANTHROPIC_API_KEY')),
                'has_auth_token': bool(env.get('ANTHROPIC_AUTH_TOKEN')),
                'base_url': env.get('ANTHROPIC_BASE_URL', ''),
                'credentials_present': bool(env.get('ANTHROPIC_API_KEY') or env.get('ANTHROPIC_AUTH_TOKEN'))}

    def check_credentials(self) -> None:
        status = self.credential_status()
        # With no custom endpoint, native Claude Code owns its login flow.
        if self.spec.get('require_credentials') and status['base_url'] and not status['credentials_present']:
            raise CompletionError('Missing ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN or configured api_key_env. '
                                  'Load .env first. LIGHTNING_API_KEY is not mapped automatically.',
                                  {'provider': self.provider, 'usage_complete': False, 'request_count': 0})

    def model_for(self, role: str) -> str:
        return str((self.config.get('models') or {}).get(role) or self.model)

    def complete(self, role: str, prompt: str, max_tokens: int,
                 temperature: float | None = None, model: str | None = None,
                 *, json_object: bool = False) -> Completion:
        selected_model = str(model or self.model_for(role))
        temperature = float(temperature if temperature is not None else (self.config.get('decoder') or {}).get('temperature', 0.7))
        child_env = self.environment(selected_model)
        execution = self.config.get('execution') or {}
        cache_key = None
        cached_path = None
        if execution.get('resume_calls'):
            identity = {
                'work_key': execution.get('work_key'), 'role': role, 'prompt': prompt,
                'model': selected_model, 'temperature': temperature, 'max_tokens': max_tokens,
                'thinking': self.thinking, 'thinking_budget_tokens': self.thinking_budget_tokens,
                'base_url': child_env.get('ANTHROPIC_BASE_URL'), 'extra_args': self.extra_args,
            }
            cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            folder = Path((self.config.get('paths') or {}).get('runs_dir', 'runs')) / str(self.config.get('run_id', 'locomo_arc')) / 'calls'
            folder.mkdir(parents=True, exist_ok=True)
            cached_path = folder / f'{cache_key}.json'
            if cached_path.exists():
                cached = json.loads(cached_path.read_text(encoding='utf-8'))
                if cached.get('status') in {'in_flight', 'outcome_unknown'}:
                    raise CompletionError('BLOCKED: interrupted request has an unknown server outcome; reconcile its log before retrying.', {'request_count': 0, 'audit_path': str(cached_path)})
                if cached.get('status') == 'ok':
                    usage = cached['usage']
                    body = cached['raw_response']
                    text = str(body.get('result') or body.get('text') or body.get('content') or '').strip()
                    return Completion(text, role, selected_model, usage.get('prompt_tokens'),
                                      usage.get('completion_tokens'), usage.get('total_tokens'),
                                      cached['latency_ms'], body,
                                      usage_complete=bool(usage.get('usage_complete')), audit_path=str(cached_path))
                # Preserve a failed attempt before explicitly retrying it.
                attempt = 1
                while cached_path.with_name(f'{cache_key}.failed-{attempt}.json').exists():
                    attempt += 1
                cached_path.rename(cached_path.with_name(f'{cache_key}.failed-{attempt}.json'))
        child_env['CLAUDE_CODE_MAX_OUTPUT_TOKENS'] = str(max_tokens)
        args = [self.command, '-p', '--output-format', 'json', '--name', f'arc-{role}',
                '--model', selected_model, *self.extra_args]
        started = time.perf_counter()
        record = {'role': role, 'model': selected_model, 'temperature': temperature, 'max_tokens': max_tokens,
                  'prompt': prompt, 'argv': args, 'base_url': child_env.get('ANTHROPIC_BASE_URL'),
                  'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  'status': 'error', 'request_count': 0}
        if cached_path is not None:
            record.update(cache_key=cache_key, work_key=execution.get('work_key'))
            with cached_path.open('x', encoding='utf-8') as handle:
                os.chmod(cached_path, 0o600)
                json.dump(_redact({**record, 'status': 'in_flight'}, child_env), handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
        relay = None
        completion = None
        failure = None
        body = {}
        usage = {'provider': self.provider, 'model': selected_model, 'usage_complete': False, 'request_count': 0}
        try:
            self.check_credentials()
            host = urlparse(child_env.get('ANTHROPIC_BASE_URL', '')).hostname
            siliconflow = host in {'api.siliconflow.cn', 'api.siliconflow.com'}
            local_bridge = host in {'127.0.0.1', 'localhost', '::1'}
            if not (siliconflow or local_bridge):
                raise CompletionError(
                    'Unsupported endpoint: use Claude Code with SiliconFlow or the local Qwen bridge.'
                )
            # SiliconFlow is reached only through the local Anthropic relay;
            # there is no Python/OpenAI-style direct model client path.
            use_relay = siliconflow
            if use_relay:
                relay = MessagesRelay(child_env['ANTHROPIC_BASE_URL'], temperature=temperature,
                                      max_tokens=max_tokens, thinking=self.thinking,
                                      thinking_budget_tokens=self.thinking_budget_tokens,
                                      timeout=min(self.timeout, 120)).start()
                child_env['ANTHROPIC_BASE_URL'] = relay.base_url
            record['request_count'] = usage['request_count'] = 1
            process = subprocess.run(args, input=prompt, text=True, capture_output=True,
                                     timeout=self.timeout, check=False, env=child_env)
            record.update(returncode=process.returncode, stdout=process.stdout, stderr=process.stderr)
            try:
                body = json.loads(process.stdout)
                if not isinstance(body, dict):
                    raise ValueError('CLI response must be an object')
            except (ValueError, TypeError) as exc:
                raise CompletionError(f'Invalid Claude Code JSON: {exc}') from exc
            body_usage = body.get('usage') or {}
            # Anthropic reports newly processed and cache input tokens separately.
            input_tokens = _integer(body_usage.get('input_tokens', body_usage.get('prompt_tokens')))
            if input_tokens is not None:
                input_tokens += sum(int(body_usage.get(k) or 0) for k in ('cache_read_input_tokens', 'cache_creation_input_tokens'))
            output_tokens = _integer(body_usage.get('output_tokens', body_usage.get('completion_tokens')))
            total = input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None
            usage.update(prompt_tokens=input_tokens, completion_tokens=output_tokens, total_tokens=total,
                         usage_complete=total is not None)
            if (process.returncode != 0 or body.get('is_error') or body.get('type') == 'error'
                    or str(body.get('subtype', '')).startswith('error') or body.get('terminal_reason') == 'api_error'):
                raise CompletionError(str(body.get('result') or body.get('error') or process.stderr or 'Claude Code failed'))
            text = str(body.get('result') or body.get('text') or body.get('content') or '').strip()
            if not text:
                raise CompletionError('Claude Code returned empty content')
            record['status'] = 'ok'
            completion = (text, input_tokens, output_tokens, total)
        except Exception as exc:
            if isinstance(exc, subprocess.TimeoutExpired):
                record['status'] = 'outcome_unknown'
                def decoded(value):
                    return value.decode(errors='replace') if isinstance(value, bytes) else (value or '')
                record.update(stdout=decoded(exc.stdout), stderr=decoded(exc.stderr))
            failure = str(exc)
            record['error'] = failure
        finally:
            if relay:
                relay.close()
                record['api_requests'] = relay.requests
            latency = int((time.perf_counter() - started) * 1000)
            usage['latency_ms'] = latency
            record.update(raw_response=body, usage=usage, latency_ms=latency)
            paths = self.config.get('paths') or {}
            folder = Path(paths.get('runs_dir', 'runs')) / str(self.config.get('run_id', 'locomo_arc')) / 'calls'
            folder.mkdir(parents=True, exist_ok=True)
            path = cached_path or folder / f'{uuid.uuid4().hex}.json'
            temporary = path.with_suffix('.tmp')
            with temporary.open('x', encoding='utf-8') as handle:
                os.chmod(temporary, 0o600)
                json.dump(_redact(record, child_env), handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        if failure is not None:
            usage['audit_path'] = str(path)
            raise CompletionError(_redact(failure, child_env), usage, _redact(body, child_env))
        text, input_tokens, output_tokens, total = completion
        return Completion(text, role, str(body.get('model') or selected_model), input_tokens, output_tokens,
                          total, latency, _redact(body, child_env), usage_complete=total is not None, audit_path=str(path))


def normalize_provider(value: str | None) -> str:
    provider = str(value or 'claude_code').strip().lower().replace('-', '_')
    return {'claude': 'claude_code', 'claudecode': 'claude_code'}.get(provider, provider)


def completion_error_row(sample_id: str, qa_id: str, phase: str, role: str, error: Exception) -> dict[str, Any]:
    return {'sample_id': str(sample_id), 'qa_id': str(qa_id), 'phase': phase, 'role': role,
            'prompt_tokens': None, 'completion_tokens': None, 'total_tokens': None, 'usage_complete': False,
            **getattr(error, 'usage', {}), 'status': 'error', 'error_type': type(error).__name__,
            'error': str(error)[:1000], 'raw_response': getattr(error, 'raw', {})}
