#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${LLM_API_KEY:?请设置真实 LLM_API_KEY}"
DATA=${DATA:-../data/LoCoMo/data/locomo10.json}
OUT=${OUT:-artifacts}
CONFIG=${CONFIG:-}
if [ -n "$CONFIG" ]; then
  python -m r2w.pipeline_train --config "$CONFIG" --data "$DATA" --out "$OUT"
else
  python -m r2w.pipeline_train --data "$DATA" --out "$OUT"
fi
