#!/usr/bin/env bash
set -euo pipefail

SERVER_URL="${1:-http://localhost:8000}"

curl "${SERVER_URL}/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V2-Lite",
    "prompt": "Explain Mixture-of-Experts routing in one sentence.",
    "max_tokens": 64,
    "temperature": 0.2
  }'
echo
