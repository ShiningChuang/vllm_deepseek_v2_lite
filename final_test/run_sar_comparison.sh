#!/usr/bin/env bash
# ============================================================
# run_sar_comparison.sh
# Compare SAR=0 vs SAR=1 goodput at concurrency=4.
# Both runs use --seed 42 so prompt/max_tokens sequence is
# identical → apples-to-apples comparison.
#
# run_id=21  SAR=0 (baseline)
# run_id=22  SAR=1 (shape-aware)
#
# Usage:
#   cd ~/vllm_deepseek_v2_lite/final_test
#   bash run_sar_comparison.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="/data/lxzhong_home/result"
CONTAINER="vllm-dev"
MODEL="deepseek-ai/DeepSeek-V2-Lite"
TOTAL_REQUESTS=200
CONCURRENCY=4
SERVER_READY_TIMEOUT=300
HEALTH_URL="http://localhost:8000/health"
SEED=42

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

wait_for_server() {
    local elapsed=0
    log "Waiting for vLLM server (up to ${SERVER_READY_TIMEOUT}s)..."
    while true; do
        if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
            log "Server ready (${elapsed}s elapsed)."
            return 0
        fi
        sleep 5; elapsed=$((elapsed + 5))
        [[ $elapsed -ge $SERVER_READY_TIMEOUT ]] && die "Server not ready within ${SERVER_READY_TIMEOUT}s."
    done
}

kill_server() {
    log "Stopping vLLM server..."
    sudo docker exec "$CONTAINER" bash -c \
        "pkill -f 'vllm.entrypoints.openai.api_server' || true"
    sleep 5
    log "Server stopped."
}

start_server() {
    local run_id=$1 sar=$2
    log "Starting server run_id=${run_id} SAR=${sar}..."
    sudo docker exec -d "$CONTAINER" bash -c "
        export VLLM_LOG_MOE_SHAPES=1
        export VLLM_LOG_MOE_RUN_ID=${run_id}
        export VLLM_MOE_SHAPE_AWARE_ROUTING=${sar}
        export VLLM_USE_V1=0
        export VLLM_MLA_DISABLE=1
        export VLLM_ATTENTION_BACKEND=XFORMERS
        python3 -m vllm.entrypoints.openai.api_server \
            --model ${MODEL} \
            --trust-remote-code \
            --dtype float16 \
            --tensor-parallel-size 2 \
            --max-model-len 8192 \
            --host 0.0.0.0 \
            --port 8000 \
            --enforce-eager \
            > /tmp/result/server_run${run_id}.log 2>&1
    "
}

run_one() {
    local run_id=$1 sar=$2 label=$3

    log ""
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "RUN: run_id=${run_id}  SAR=${sar}  label=${label}"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    kill_server || true
    rm -f "$RESULT_DIR"/moe_shapes_run${run_id}_rank*.jsonl

    start_server "$run_id" "$sar"
    wait_for_server

    # Warm-up (small, deterministic)
    log "Warm-up request..."
    curl -sf http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${MODEL}\",\"prompt\":\"Hello\",\"max_tokens\":4,\"temperature\":0}" \
        > /dev/null
    log "Warm-up done."
    sleep 2

    # Clear warm-up logs
    rm -f "$RESULT_DIR"/moe_shapes_run${run_id}_rank*.jsonl
    log "Cleared warm-up logs. Starting real workload..."

    CLIENT_OUT="$RESULT_DIR/client_${label}.json"
    python3 "$SCRIPT_DIR/run_client.py" \
        --prompts "$SCRIPT_DIR/prompts.txt" \
        --concurrency "$CONCURRENCY" \
        --total-requests "$TOTAL_REQUESTS" \
        --seed "$SEED" \
        --output "$CLIENT_OUT"

    log "Workload done. Results: $CLIENT_OUT"

    RANK0="$RESULT_DIR/moe_shapes_run${run_id}_rank0.jsonl"
    if [[ -f "$RANK0" ]]; then
        CNT=$(wc -l < "$RANK0")
        log "MoE log rank0: ${CNT} lines."
    else
        log "WARNING: $RANK0 not found!"
    fi
}

# ── Pre-flight ────────────────────────────────────────────────
if ! sudo docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    die "Container '${CONTAINER}' not running."
fi
mkdir -p "$RESULT_DIR"

# ── Run SAR=0 ─────────────────────────────────────────────────
run_one 21 0 "sar0_conc4"

# ── Run SAR=1 ─────────────────────────────────────────────────
run_one 22 1 "sar1_conc4"

kill_server || true

# ── Analyse ───────────────────────────────────────────────────
log ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "ANALYSIS (Prefill/Mixed only, tokens_in_chunk > ${CONCURRENCY})"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 "$SCRIPT_DIR/analyze_sar_comparison.py" \
    --baseline "$RESULT_DIR/moe_shapes_run21_rank0.jsonl" \
    --sar      "$RESULT_DIR/moe_shapes_run22_rank0.jsonl" \
    --concurrency "$CONCURRENCY" \
    --output   "$RESULT_DIR/sar_comparison_conc4.png"

log "=== run_sar_comparison.sh DONE ==="
