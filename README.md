# Run

```bash
./stage1_disk_nvidia.sh
./stage2_docker_nvidia.sh
newgrp docker
./stage3_load_deepseek_v2_lite.sh
```

# Enter developer mode

```bash
# pull source code of llm
cd $HOME_LLM/vllm_dev &&
git clone https://github.com/vllm-project/vllm.git &&
cd vllm &&
git checkout v0.10.2

# enter develop docker
docker run --rm -it --gpus all \
  -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_USE_V1=0 \
  -e VLLM_MLA_DISABLE=1 \
  -e VLLM_ATTENTION_BACKEND=XFORMERS \
  -v $HOME_LLM/hf_cache:/root/.cache/huggingface \
  -v $HOME_LLM/vllm_dev/vllm:/workspace/vllm \
  --name vllm-dev \
  --entrypoint bash \
  vllm/vllm-openai:v0.10.2

# editable install
pip uninstall -y vllm || true &&
pip install -e /workspace/vllm &&
python3 -c "import vllm; import os; print('vllm from:', vllm.__file__)"

# run server using source version
python3 -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-V2-Lite \
  --trust-remote-code \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --port 8000
```