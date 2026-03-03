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

export HOME_LLM=/data/lxzhong_home
# enter develop docker
docker run -it --gpus all \
  -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_USE_V1=0 \
  -e VLLM_MLA_DISABLE=1 \
  -e VLLM_ATTENTION_BACKEND=XFORMERS \
  -v $HOME_LLM/hf_cache:/root/.cache/huggingface \
  -v $HOME_LLM/vllm_dev/vllm:/workspace/vllm \
  -v $HOME_LLM/result:/tmp \
  --name vllm-dev \
  --entrypoint bash \
  vllm/vllm-openai:v0.10.2

# editable install
pip uninstall -y vllm || true &&
pip install -e /workspace/vllm &&
python3 -c "import vllm; import os; print('vllm from:', vllm.__file__)"

# run server using source version
export VLLM_LOG_MOE_SHAPES=1 &&
export VLLM_LOG_MOE_RUN_ID=1  &&
export VLLM_MOE_SHAPE_AWARE_ROUTING=0 &&
python3 -m vllm.entrypoints.openai.api_server   --model deepseek-ai/DeepSeek-V2-Lite   --trust-remote-code   --dtype float16   --tensor-parallel-size 4   --max-model-len 8192   --host 0.0.0.0   --port 8000   --enforce-eager

# client test in another terminal
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V2-Lite",
    "prompt": "Say hello in one sentence.",
    "max_tokens": 32,
    "temperature": 0.2
  }'

  # ShareGPT requests
  python3 concurrent_client_test.py

  # analyze approach 3
  python analyze_shape_aware.py     \
  /data/lxzhong_home/result/moe_shapes_run10_rank0.jsonl \
  /data/lxzhong_home/result/moe_shapes_run11_rank0.jsonl  \
  --labels baseline shape_aware

```