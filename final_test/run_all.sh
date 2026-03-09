#!/usr/bin/env bash
# ============================================================
# run_all.sh  —  Full experiment: start vLLM server for each
# concurrency level, send ShareGPT workload, then analyse.
#
# concurrency=1  → VLLM_LOG_MOE_RUN_ID=16
# concurrency=4  → VLLM_LOG_MOE_RUN_ID=17
# concurrency=8  → VLLM_LOG_MOE_RUN_ID=18
#
# Usage:
#   cd ~/vllm_deepseek_v2_lite/final_test
#   bash run_all.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="/data/lxzhong_home/result"
CONTAINER="vllm-dev"
MODEL="deepseek-ai/DeepSeek-V2-Lite"
TOTAL_REQUESTS=100        # requests per concurrency level
SERVER_READY_TIMEOUT=300  # seconds to wait for server
HEALTH_URL="http://localhost:8000/health"

CONCURRENCY_LEVELS=(1 4 8)
RUN_IDS=(16 17 18)
LABELS=("conc=1" "conc=4" "conc=8")

# ---------- helpers ----------
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

wait_for_server() {
    local timeout=$SERVER_READY_TIMEOUT
    log "Waiting for vLLM server (up to ${timeout}s)..."
    local elapsed=0
    while true; do
        if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
            log "Server is ready (${elapsed}s elapsed)."
            return 0
        fi
        sleep 5; elapsed=$((elapsed + 5))
        if [[ $elapsed -ge $timeout ]]; then
            die "Server did not become ready within ${timeout}s."
        fi
    done
}

kill_server() {
    log "Stopping vLLM server inside container..."
    sudo docker exec "$CONTAINER" bash -c \
        "pkill -f 'vllm.entrypoints.openai.api_server' || true"
    sleep 5
    log "Server stopped."
}

start_server() {
    local run_id=$1
    log "Starting vLLM server with RUN_ID=${run_id} ..."
    sudo docker exec -d "$CONTAINER" bash -c "
        export VLLM_LOG_MOE_SHAPES=1
        export VLLM_LOG_MOE_RUN_ID=${run_id}
        export VLLM_MOE_SHAPE_AWARE_ROUTING=1
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

# ---------- pre-flight ----------
log "=== run_all.sh started ==="
log "RESULT_DIR  : $RESULT_DIR"
log "CONTAINER   : $CONTAINER"
log "TOTAL_REQ   : $TOTAL_REQUESTS"

# Ensure container is running
if ! sudo docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    die "Container '${CONTAINER}' is not running. Start it first."
fi

mkdir -p "$RESULT_DIR"

# Prepare prompts if not already done
PROMPTS_FILE="$SCRIPT_DIR/prompts.txt"
if [[ ! -f "$PROMPTS_FILE" ]]; then
    log "Preparing prompts (downloading ShareGPT)..."
    python3 "$SCRIPT_DIR/prepare_prompts.py" --output "$PROMPTS_FILE"
fi
log "Prompts file: $PROMPTS_FILE ($(wc -l < "$PROMPTS_FILE") lines)"

# ---------- main loop ----------
for i in "${!CONCURRENCY_LEVELS[@]}"; do
    CONC="${CONCURRENCY_LEVELS[$i]}"
    RUN_ID="${RUN_IDS[$i]}"
    LABEL="${LABELS[$i]}"

    log ""
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "TEST: concurrency=${CONC}  run_id=${RUN_ID}  label=${LABEL}"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Kill any existing server
    kill_server || true

    # Remove old result files for this run_id to avoid mixing data
    rm -f "$RESULT_DIR"/moe_shapes_run${RUN_ID}_rank*.jsonl
    log "Cleared old logs for run_id=${RUN_ID}."

    # Start server
    start_server "$RUN_ID"

    # Wait until ready
    wait_for_server

    # Short warm-up request
    log "Sending warm-up request..."
    curl -sf http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${MODEL}\",\"prompt\":\"Hello\",\"max_tokens\":4,\"temperature\":0}" \
        > /dev/null
    log "Warm-up done."
    sleep 2

    # Clear logs again (warm-up polluted them)
    rm -f "$RESULT_DIR"/moe_shapes_run${RUN_ID}_rank*.jsonl
    log "Cleared warm-up logs. Starting real workload..."

    # Run workload
    CLIENT_OUT="$RESULT_DIR/client_run${RUN_ID}_conc${CONC}.json"
    python3 "$SCRIPT_DIR/run_client.py" \
        --prompts "$PROMPTS_FILE" \
        --concurrency "$CONC" \
        --total-requests "$TOTAL_REQUESTS" \
        --output "$CLIENT_OUT"

    log "Workload done. Client results: $CLIENT_OUT"

    # Check that log files were created
    RANK0_LOG="$RESULT_DIR/moe_shapes_run${RUN_ID}_rank0.jsonl"
    if [[ -f "$RANK0_LOG" ]]; then
        CNT=$(wc -l < "$RANK0_LOG")
        log "MoE log rank0: ${CNT} lines."
    else
        log "WARNING: $RANK0_LOG not found!"
    fi
done

# Kill server after all tests
kill_server || true

log ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "All tests complete. Running analysis..."
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 "$SCRIPT_DIR/analyze_phases.py" \
    --result-dir "$RESULT_DIR" \
    --run-ids "16,17,18" \
    --labels "conc=1,conc=4,conc=8" \
    --concurrencies "1,4,8" \
    --ranks "0,1" \
    2>&1 | tee "$RESULT_DIR/phase_analysis.txt"

log "Analysis saved to $RESULT_DIR/phase_analysis.txt"
log "=== run_all.sh DONE ==="
