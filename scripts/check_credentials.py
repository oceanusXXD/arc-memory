#!/usr/bin/env python3
"""Check the same loaded configuration and environment used by evaluation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from arc.agent.config import load_config
from arc.agent.reader import ClaudeCodeClient, CompletionError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/locomo.yaml')
    args = parser.parse_args()
    client = ClaudeCodeClient(load_config(args.config))
    try:
        client.check_credentials()
    except CompletionError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    status = client.credential_status()
    print('credentials: present' if status['credentials_present'] else 'credentials: native Claude Code login')
    print('model:', client.model_for('agent'))
    print('API authentication is verified by the first model call.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
