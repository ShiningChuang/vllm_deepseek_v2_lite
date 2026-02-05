#!/usr/bin/env bash
set -euo pipefail

echo "[0/7] Config..."
export HOME_LLM="/data/lxzhong_home"
VLLM_IMAGE="vllm/vllm-openai:v0.10.2"
MODEL_ID="deepseek-ai/DeepSeek-V2-Lite"

echo "[1/7] Sanity checks..."
if [[ ! -d /data ]]; then
  echo "[ERROR] /data does not exist. Did you mount the expanded disk?"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] docker not found. Run stage2 first."
  exit 1
fi

# HF token check (do NOT hardcode in script)
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[ERROR] HF_TOKEN is not set."
  echo "  Please run: export HF_TOKEN='hf_...'"
  exit 1
fi

echo "[2/7] Ensure /data ownership for current user (so cache dirs are writable)..."
sudo chown -R "$(whoami):$(id -gn)" /data

echo "[3/7] Prepare Hugging Face cache dir on host..."
mkdir -p "$HOME_LLM/hf_cache"
echo "Host HF cache: $HOME_LLM/hf_cache"

echo "[4/7] Quick GPU-in-docker sanity test..."
sudo docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi >/dev/null
echo "[OK] Docker can see GPUs."

echo "[5/7] Pull vLLM image (${VLLM_IMAGE})..."
sudo docker pull "${VLLM_IMAGE}"

echo "[6/7] Pre-download model to host HF cache via container..."
# Use the SAME image tag (v0.10.2) to avoid latest incompatibilities.
sudo docker run --rm --gpus all \
  -e HF_TOKEN="${HF_TOKEN}" \
  -v "${HOME_LLM}/hf_cache:/root/.cache/huggingface" \
  --entrypoint python3 \
  "${VLLM_IMAGE}" \
  -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}', local_dir_use_symlinks=False)"

echo "[7/7] Done."
echo "Model cache should now be under: ${HOME_LLM}/hf_cache"
echo "You can reuse it in future containers via:"
echo "  -v ${HOME_LLM}/hf_cache:/root/.cache/huggingface"
