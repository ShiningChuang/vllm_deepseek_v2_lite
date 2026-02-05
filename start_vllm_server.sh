#!/usr/bin/env bash
set -euo pipefail

# ---- Config (can be overridden by env vars) ----
: "${HOME_LLM:=/data/lxzhong_home}"
: "${HF_TOKEN:?HF_TOKEN is not set. Please: export HF_TOKEN='hf_...'}"

IMAGE="vllm/vllm-openai:v0.10.2"
MODEL_ID="deepseek-ai/DeepSeek-V2-Lite"

HF_CACHE="${HOME_LLM}/hf_cache"

echo "[1/6] Sanity checks..."
if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] docker not found."
  exit 1
fi

if [[ ! -d "${HF_CACHE}" ]]; then
  echo "[ERROR] HF cache dir not found: ${HF_CACHE}"
  echo "Run stage3 to download model first, or create it: mkdir -p ${HF_CACHE}"
  exit 1
fi

echo "[2/6] GPU visibility check..."
sudo docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi >/dev/null
echo "[OK] Docker can see GPUs."

echo "[3/6] Pull image (optional if already pulled): ${IMAGE}"
sudo docker pull "${IMAGE}" >/dev/null

echo "[4/6] Starting vLLM OpenAI server..."
echo "  Model: ${MODEL_ID}"
echo "  Port : 8000"
echo "  Cache: ${HF_CACHE}"
echo "Press Ctrl+C to stop."

sudo docker run --rm --gpus all \
  -p 8000:8000 \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e VLLM_USE_V1=0 \
  -e VLLM_MLA_DISABLE=1 \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  "${IMAGE}" \
  --model "${MODEL_ID}" \
  --trust-remote-code \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --port 8000
