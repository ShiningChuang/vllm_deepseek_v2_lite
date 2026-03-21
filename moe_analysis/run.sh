#!/usr/bin/env bash
# Run expert hotness analysis inside the vllm-dev docker container.
# Must be executed INSIDE the container (not on the host).

set -e

# Make the analysis dir accessible inside the container
# (host path /users/lxzhong/... must be mounted if not already)
# The script writes figures/npy to the same directory as itself.

pip install datasets matplotlib --quiet 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/analyze_expert_hotness.py"
