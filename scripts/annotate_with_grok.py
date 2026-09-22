#!/usr/bin/env python3
"""Create a small teacher-annotation sample with Grok-4.6."""
from __future__ import annotations

import argparse
import json
import hashlib
import os
import time
import tomllib
import urllib.error
import urllib.request
from http.client import IncompleteRead
from pathlib import Path
from typing import Any


HARNESS = """You are the requirement teacher.

Given one question, a numbered source list, and the Full-memory builder output, identify the minimal information requirements needed to answer the question. For each requirement, provide one to three alternative support packages. A package is a minimal non-empty set of source numbers whose combined text supports that requirement.

Rules:
- Use only the supplied source numbers.
- Do not infer facts not supported by the source text.
- At most 6 requirements.
- Each requirement has 1 to 3 packages.
- Each package has at least one source number and no duplicates.
- Prefer small packages that contain exactly the necessary sources.
- Return one valid JSON object and nothing else. Do not use Markdown or code fences.

Schema:
{"requirements":[{"id":"r1","text":"requirement in one sentence","packages":[[1]]}]}

If the sources are insufficient, return {"requirements":[]}."""


def _text_from_response(body: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") in {"output_text", "text"}:
                chunks.append(str(content.get("text") or ""))
    return "".join(chunks).strip()


def _parse_json(text: str) -> dict[str, Any]:
    value = text.strip()
    if "```" in value:
        lines = value.splitlines()
        start = next((i for i, line in enumerate(lines) if line.strip().lower() in {"```json", "```"}), None)
        end = next((i for i in range(start + 1, len(lines)) if lines[i].strip() == "```"), None) if start is not None else None
        if end is not None:
            value = "\n".join(lines[start + 1:end]).strip()
    try:
        body = json.loads(value)
    except json.JSONDecodeError:
        # A complete requirements array with only its outer object delimiter
        # missing is recoverable without supplying or changing any content.
        # All other malformed/truncated payloads remain errors.
        if not value.startswith('{"requirements":') or not value.endswith("]"):
            raise
        body = json.loads(value + "}")
        if set(body) != {"requirements"}:
            raise ValueError("cannot repair teacher JSON beyond one missing root brace")
        body["_syntax_repair"] = "appended_missing_root_object_closer"
    if not isinstance(body, dict):
        raise ValueError("teacher response must be a JSON object")
    return body


def _credential(config_path: Path) -> tuple[str, str]:
    """Use the URL and credential belonging to the same active Codex provider."""
    settings_path = config_path / "config.toml"
    if not settings_path.exists():
        raise FileNotFoundError(f"missing Codex provider configuration: {settings_path}")
    settings = tomllib.loads(settings_path.read_text(encoding="utf-8"))
    provider_name = settings.get("model_provider")
    provider = (settings.get("model_providers") or {}).get(provider_name) or {}
    base_url = str(provider.get("base_url") or "").rstrip("/")
    if not base_url:
        raise ValueError("active Codex provider has no explicit base_url")
    if provider.get("wire_api", "responses") != "responses":
        raise ValueError("the teacher requires a Codex provider with wire_api=responses")
    env_key = provider.get("env_key")
    if env_key:
        api_key = os.environ.get(str(env_key))
        if not api_key:
            raise ValueError(f"missing credential environment variable configured by Codex: {env_key}")
        return base_url, api_key
    auth_path = config_path / "auth.json"
    if not auth_path.exists():
        raise FileNotFoundError(f"missing Codex credentials: {auth_path}")
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    api_key = auth.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("missing OPENAI_API_KEY in Codex auth.json")
    return base_url, str(api_key)


def _call_teacher(base_url: str, api_key: str, question: str, sources: list[dict[str, Any]], full_output: str,
                  *, audit_dir: Path | None = None, work_key: dict | None = None) -> dict[str, Any]:
    if not full_output.strip():
        raise ValueError("Full(q) raw output is required before a teacher call")
    prompt = json.dumps({
        "task": HARNESS,
        "question": question,
        "sources": [
            {
                "id": index,
                "source_id": source.get("source_id"),
                "time": source.get("session_datetime"),
                "speaker": source.get("speaker"),
                "text": source.get("text"),
            }
            for index, source in enumerate(sources, 1)
        ],
        "full_output": full_output,
    }, ensure_ascii=False)
    payload = {
        "model": "grok-4.6",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "temperature": 0.0,
        "max_output_tokens": 2000,
    }
    request = urllib.request.Request(
        f"{base_url}/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    audit_path = None
    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(json.dumps({"payload": payload, "work_key": work_key, "base_url": base_url}, sort_keys=True).encode()).hexdigest()
        audit_path = audit_dir / f"{key}.json"
        if audit_path.exists():
            prior = json.loads(audit_path.read_text(encoding="utf-8"))
            if prior.get("status") in {"in_flight", "outcome_unknown"}:
                raise RuntimeError("BLOCKED: teacher request outcome unknown after interruption; reconcile saved request")
            if prior.get("status") == "ok":
                return {**prior["parsed"], "_usage": prior.get("usage"), "_audit_path": str(audit_path)}
            attempt = 1
            while audit_path.with_name(f"{key}.failed-{attempt}.json").exists():
                attempt += 1
            audit_path.rename(audit_path.with_name(f"{key}.failed-{attempt}.json"))
    record = {"status": "in_flight", "model": "grok-4.6", "provider": "codex_config", "base_url": base_url,
              "credential_source": "active Codex provider", "work_key": work_key, "payload": payload,
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "request_count": 1}
    def save():
        if audit_path is not None:
            temporary = audit_path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                os.chmod(temporary, 0o600)
                json.dump(record, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(audit_path)
    save()
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.load(response)
        record.update(raw_response=body, usage=body.get("usage"))
        parsed = _parse_json(_text_from_response(body))
        _validate({"sources": sources, "requirements": parsed.get("requirements")})
        record.update(status="ok", parsed=parsed)
        return {**parsed, "_usage": body.get("usage"), "_audit_path": str(audit_path) if audit_path else None}
    except Exception as exc:
        record.update(status="error", error=str(exc).replace(api_key, "<redacted>"))
        if isinstance(exc, (TimeoutError, IncompleteRead)) or (isinstance(exc, urllib.error.URLError) and not isinstance(exc, urllib.error.HTTPError)):
            record["status"] = "outcome_unknown"
        if isinstance(exc, urllib.error.HTTPError):
            record["http_status"] = exc.code
            record["response"] = exc.read().decode("utf-8").replace(api_key, "<redacted>")
        raise
    finally:
        record["latency_ms"] = int((time.monotonic() - started) * 1000)
        save()


def _validate(row: dict[str, Any]) -> None:
    requirements = row.get("requirements")
    if not isinstance(requirements, list) or len(requirements) > 6:
        raise ValueError("requirements must be a list of at most 6 items")
    source_count = len(row.get("sources") or [])
    seen_ids: set[str] = set()
    for requirement in requirements:
        if not isinstance(requirement, dict):
            raise ValueError("requirement must be an object")
        requirement_id = str(requirement.get("id") or "")
        if not requirement_id or requirement_id in seen_ids:
            raise ValueError(f"invalid or duplicate requirement id: {requirement_id!r}")
        seen_ids.add(requirement_id)
        if not str(requirement.get("text") or "").strip():
            raise ValueError(f"empty requirement text: {requirement_id}")
        packages = requirement.get("packages")
        if not isinstance(packages, list) or not 1 <= len(packages) <= 3:
            raise ValueError(f"requirement {requirement_id} must have 1 to 3 packages")
        for package in packages:
            if not isinstance(package, list) or not package:
                raise ValueError(f"requirement {requirement_id} has an empty package")
            if any(type(value) is not int for value in package):
                raise ValueError(f"requirement {requirement_id} has a non-integer source id")
            if any(value < 1 or value > source_count for value in package):
                raise ValueError(f"requirement {requirement_id} has an out-of-range source id")
            if len(package) != len(set(package)):
                raise ValueError(f"requirement {requirement_id} has a duplicate source id")


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate a small teacher sample with Grok-4.6.")
    parser.add_argument("--input", default="runs/verify_small/requirements.jsonl")
    parser.add_argument("--full", default="runs/verify_small/requirements.full.jsonl")
    parser.add_argument("--output", default="runs/verify_small/requirements.grok.jsonl")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--codex-home", default="/teamspace/studios/this_studio/.codex")
    parser.add_argument("--max-source-bytes", type=int, default=12000)
    args = parser.parse_args()

    inputs = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    full_path = Path(args.full)
    full_rows = {str(row.get("qa_id")): row for row in (json.loads(line) for line in full_path.read_text(encoding="utf-8").splitlines() if line.strip())} if full_path.exists() else {}
    selected = inputs[:args.limit]

    missing = [row["qa_id"] for row in selected if str(row["qa_id"]) not in full_rows]
    if missing:
        raise ValueError(f"missing Full outputs for: {missing}")

    base_url, api_key = _credential(Path(args.codex_home))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing = json.loads(line)
                _validate(existing)
                done.add(str(existing["qa_id"]))

    with output_path.open("a", encoding="utf-8") as handle:
        for row in selected:
            if str(row["qa_id"]) in done:
                continue
            full = full_rows.get(str(row["qa_id"])) or {}
            full_output = str(full.get("full_raw") or full.get("raw_output") or "")
            annotation = _call_teacher(base_url, api_key, str(row["question"]), row["sources"], full_output,
                                       audit_dir=output_path.parent / "teacher_calls", work_key={"qa_id": row["qa_id"]})
            result = {**row, "requirements": annotation["requirements"], "teacher_usage": annotation.get("_usage"), "teacher_audit_path": annotation.get("_audit_path")}
            _validate(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            done.add(str(row["qa_id"]))
            print(f"annotated {row['qa_id']}: {len(annotation['requirements'])} requirements", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
