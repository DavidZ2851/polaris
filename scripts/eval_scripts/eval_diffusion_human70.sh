#!/bin/bash
# Sequentially evaluate human70_robot5 and human70_robot40 diffusion policy checkpoints on one port

set -e

CKPT_DIR="/data/vxiao/benchmark_new/polaris/policy_ckpt/diffusion_policy/red_mug/new_camera_calib"
SERVER_SCRIPT="/home/veraxiao/vxiao/human2robot/benchmark_new/polaris/scripts/diffusion_policy_server.py"
EVAL_SCRIPT="/home/veraxiao/vxiao/human2robot/benchmark_new/polaris/scripts/eval_lerobot.py"
RUN_BASE="${RUN_BASE:-/home/veraxiao/vxiao/human2robot/benchmark_new/polaris/runs/new_camera_calib}"
ENVIRONMENT="DROID-PutRedCup-no-curtain"
ROLLOUTS=30
OPEN_LOOP_HORIZON=8
MAX_EPISODE_LENGTH=300
GPU=${GPU:-0}
PORT=${PORT:-5557}

ROBODIFF_PY="/home/veraxiao/miniconda3/envs/robodiff/bin/python"
POLARIS_PY="/home/veraxiao/vxiao/human2robot/benchmark_new/polaris/.venv/bin/python"
DIFFPOL_DIR="/home/veraxiao/vxiao/human2robot/benchmark_new/diffusion_policy"
POLARIS_DIR="/home/veraxiao/vxiao/human2robot/benchmark_new/polaris"

CKPTS=(
    "$CKPT_DIR/human70_robot40.ckpt"
)

wait_for_port() {
    local port=$1
    echo "Waiting for server on port $port..."
    for i in $(seq 1 60); do
        if nc -z localhost "$port" 2>/dev/null; then
            echo "Server is ready on port $port"
            return 0
        fi
        sleep 2
    done
    echo "ERROR: Server did not start on port $port after 120s"
    exit 1
}

wait_for_port_free() {
    local port=$1
    echo "Waiting for port $port to be released..."
    for i in $(seq 1 30); do
        if ! lsof -i ":$port" > /dev/null 2>&1; then
            echo "Port $port is free."
            return 0
        fi
        sleep 2
    done
    echo "ERROR: Port $port still in use after 60s"
    exit 1
}

SERVER_PID=""
EVAL_PID=""
cleanup() {
    if [ -n "$EVAL_PID" ]; then
        echo "Cleanup: killing eval PID $EVAL_PID"
        kill "$EVAL_PID" 2>/dev/null || true
        wait "$EVAL_PID" 2>/dev/null || true
    fi
    if [ -n "$SERVER_PID" ]; then
        echo "Cleanup: killing server PID $SERVER_PID"
        kill -9 "$SERVER_PID" 2>/dev/null || true
        lsof -ti ":$PORT" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

for CKPT in "${CKPTS[@]}"; do
    NAME=$(basename "$CKPT" .ckpt)
    RUN_FOLDER="$RUN_BASE/$NAME"
    mkdir -p "$RUN_FOLDER"

    echo ""
    echo "========================================="
    echo "Evaluating: $NAME"
    echo "  ckpt: $CKPT"
    echo "  run_folder: $RUN_FOLDER"
    echo "========================================="

    # Start server in background
    (cd "$DIFFPOL_DIR" && CUDA_VISIBLE_DEVICES=$GPU "$ROBODIFF_PY" "$SERVER_SCRIPT" \
        --ckpt_path "$CKPT" --port "$PORT") &
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"

    wait_for_port "$PORT"

    # Start eval in background so we can monitor both
    (cd "$POLARIS_DIR" && CUDA_VISIBLE_DEVICES=$GPU "$POLARIS_PY" "$EVAL_SCRIPT" \
        --policy.client DiffusionPolicy \
        --policy.host localhost \
        --policy.port "$PORT" \
        --policy.open_loop_horizon "$OPEN_LOOP_HORIZON" \
        --environment "$ENVIRONMENT" \
        --run_folder "$RUN_FOLDER" \
        --rollouts "$ROLLOUTS" \
        --max-episode-length "$MAX_EPISODE_LENGTH") &
    EVAL_PID=$!
    echo "Eval PID: $EVAL_PID"

    # Monitor: if server dies before eval finishes, abort
    while kill -0 "$EVAL_PID" 2>/dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: Server (PID $SERVER_PID) died unexpectedly for $NAME. Killing eval."
            kill "$EVAL_PID" 2>/dev/null || true
            wait "$EVAL_PID" 2>/dev/null || true
            EVAL_PID=""
            SERVER_PID=""
            exit 1
        fi
        sleep 5
    done

    # Eval finished normally — capture exit code
    wait "$EVAL_PID"
    EVAL_EXIT=$?
    EVAL_PID=""
    echo "Eval ($NAME) done."

    # Kill server and wait for port to be fully released
    echo "Killing server PID $SERVER_PID"
    kill -9 "$SERVER_PID" 2>/dev/null || true
    # Also kill any child process still holding the port
    lsof -ti ":$PORT" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    echo "Server stopped."
    wait_for_port_free "$PORT"

    if [ $EVAL_EXIT -ne 0 ]; then
        echo "ERROR: Eval exited with code $EVAL_EXIT for $NAME. Stopping."
        exit $EVAL_EXIT
    fi

    echo "Done: $NAME"
done

echo ""
echo "All human70 diffusion evaluations complete."
