#!/usr/bin/env bash
# ============================================================
# run_dailymail.sh  —  DailyMail summarization workload
# concurrency=1  → VLLM_LOG_MOE_RUN_ID=23
# concurrency=4  → VLLM_LOG_MOE_RUN_ID=24
# concurrency=8  → VLLM_LOG_MOE_RUN_ID=25
#
# Usage:
#   cd ~/vllm_deepseek_v2_lite/final_test
#   bash run_dailymail.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="/data/lxzhong_home/result"
CONTAINER="vllm-dev"
MODEL="deepseek-ai/DeepSeek-V2-Lite"
TOTAL_REQUESTS=100
SERVER_READY_TIMEOUT=300
HEALTH_URL="http://localhost:8000/health"
SEED=42

CONCURRENCY_LEVELS=(1 4 8)
RUN_IDS=(23 24 25)
LABELS=("conc=1" "conc=4" "conc=8")

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

wait_for_server() {
    log "Waiting for vLLM server (up to ${SERVER_READY_TIMEOUT}s)..."
    local elapsed=0
    while true; do
        if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
            log "Server ready (${elapsed}s elapsed)."
            return 0
        fi
        sleep 5; elapsed=$((elapsed + 5))
        [[ $elapsed -ge $SERVER_READY_TIMEOUT ]] && die "Server not ready."
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
    local run_id=$1
    log "Starting vLLM server RUN_ID=${run_id}..."
    sudo docker exec -d "$CONTAINER" bash -c "
        export VLLM_LOG_MOE_SHAPES=1
        export VLLM_LOG_MOE_RUN_ID=${run_id}
        export VLLM_MOE_SHAPE_AWARE_ROUTING=0
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

# ── Pre-flight ────────────────────────────────────────────────────────────────
log "=== run_dailymail.sh started ==="
if ! sudo docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    die "Container '${CONTAINER}' not running."
fi
mkdir -p "$RESULT_DIR"

# Prepare DailyMail prompts
PROMPTS_FILE="$SCRIPT_DIR/dailymail_prompts.txt"
if [[ ! -f "$PROMPTS_FILE" ]]; then
    log "Preparing DailyMail prompts..."
    python3 "$SCRIPT_DIR/prepare_prompts_dailymail.py" \
        --output "$PROMPTS_FILE" --max-prompts 2000 --max-chars 3000
fi
log "Prompts: $PROMPTS_FILE ($(wc -l < "$PROMPTS_FILE") lines)"

# ── Main loop ─────────────────────────────────────────────────────────────────
for i in "${!CONCURRENCY_LEVELS[@]}"; do
    CONC="${CONCURRENCY_LEVELS[$i]}"
    RUN_ID="${RUN_IDS[$i]}"
    LABEL="${LABELS[$i]}"

    log ""
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "TEST: concurrency=${CONC}  run_id=${RUN_ID}  label=${LABEL}"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    kill_server || true
    rm -f "$RESULT_DIR"/moe_shapes_run${RUN_ID}_rank*.jsonl
    log "Cleared old logs for run_id=${RUN_ID}."

    start_server "$RUN_ID"
    wait_for_server

    log "Warm-up request..."
    curl -sf http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${MODEL}\",\"prompt\":\"Hello\",\"max_tokens\":4,\"temperature\":0}" \
        > /dev/null
    log "Warm-up done."
    sleep 2

    rm -f "$RESULT_DIR"/moe_shapes_run${RUN_ID}_rank*.jsonl
    log "Cleared warm-up logs. Starting workload..."

    CLIENT_OUT="$RESULT_DIR/client_dailymail_run${RUN_ID}_conc${CONC}.json"
    python3 "$SCRIPT_DIR/run_client.py" \
        --prompts "$PROMPTS_FILE" \
        --concurrency "$CONC" \
        --total-requests "$TOTAL_REQUESTS" \
        --seed "$SEED" \
        --output "$CLIENT_OUT"

    log "Workload done. Client results: $CLIENT_OUT"

    RANK0_LOG="$RESULT_DIR/moe_shapes_run${RUN_ID}_rank0.jsonl"
    if [[ -f "$RANK0_LOG" ]]; then
        CNT=$(wc -l < "$RANK0_LOG")
        log "MoE log rank0: ${CNT} lines."
    else
        log "WARNING: $RANK0_LOG not found!"
    fi
done

kill_server || true

# ── Plots ─────────────────────────────────────────────────────────────────────
log ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Generating CDF plots..."
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 "$SCRIPT_DIR/plot_utilization_cdf.py" \
    --result-dir "$RESULT_DIR" \
    --run-ids "23,24,25" \
    --labels "conc=1,conc=4,conc=8" \
    --concurrencies "1,4,8" \
    --output-prefix "dailymail" \
    --title-suffix "DailyMail Workload"

log "=== run_dailymail.sh DONE ==="
