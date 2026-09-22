#!/usr/bin/env python3
"""Start the repository-owned local Qwen vLLM + Anthropic bridge.

Claude Code remains the client. This script owns both the vLLM process and
the loopback Messages bridge; it has no dependency on another checkout.
"""
from __future__ import annotations
import argparse, json, shlex, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from arc.local_qwen import LocalQwenServer, config_from_args, wait_for_signal

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen3.5-4B')
    parser.add_argument('--served-model-name', default='Qwen/Qwen3.5-4B')
    parser.add_argument('--host', default='127.0.0.1'); parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--bridge-port', type=int, default=8080); parser.add_argument('--bridge-host', default='127.0.0.1')
    parser.add_argument('--vllm-bin', default='vllm'); parser.add_argument('--dtype', choices=['auto','bfloat16','float16','float32'], default='bfloat16')
    parser.add_argument('--max-model-len', type=int, default=32768); parser.add_argument('--max-output-tokens', type=int, default=2048)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.90); parser.add_argument('--max-num-seqs', type=int, default=1)
    parser.add_argument('--max-num-batched-tokens', type=int, default=8192); parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--no-prefix-caching', action='store_true'); parser.add_argument('--reasoning-parser', default='qwen3'); parser.add_argument('--tool-call-parser', default='qwen3_coder')
    parser.add_argument('--revision'); parser.add_argument('--sampling-profile', choices=['coding','general','passthrough'], default='coding'); parser.add_argument('--thinking', choices=['on','off','auto'], default='on')
    parser.add_argument('--startup-timeout', type=float, default=600.0); parser.add_argument('--env-file', default='runs/local/claude.env'); parser.add_argument('--log-file', default='runs/local/vllm.log')
    args = parser.parse_args()
    server = LocalQwenServer(config_from_args(args), bridge_host=args.bridge_host, bridge_port=args.bridge_port)
    try:
        with server:
            exports = ''.join(f'export {key}={shlex.quote(value)}\n' for key, value in server.claude_env().items())
            target = Path(args.env_file); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(exports, encoding='utf-8'); target.chmod(0o600)
            print(json.dumps({'status':'ready','descriptor':server.descriptor(),'env_file':str(target)}, ensure_ascii=False, indent=2), flush=True)
            wait_for_signal(server)
    except KeyboardInterrupt:
        return 130
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
