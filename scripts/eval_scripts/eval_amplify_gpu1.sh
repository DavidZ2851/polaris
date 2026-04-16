#!/bin/bash
# Evaluates amplify_new checkpoints sequentially on GPU 1, port 5558
# Server: conda amplify env | Client: polaris .venv

set -e

POLARIS_DIR=~/vxiao/human2robot/benchmark_new/polaris
AMPLIFY_ROOT=/home/veraxiao/vxiao/human2robot/benchmark_new/policy/amplify_polaris
CKPT_BASE=/data/vxiao/benchmark_new/polaris/policy_ckpt/amplify_new
TEXT_EMB=$CKPT_BASE/pick_mug_text_emb.npy
RUN_BASE=${RUN_BASE:-/home/veraxiao/vxiao/human2robot/benchmark_new/polaris/runs/new_camera_calib/amplify}
CONDA_PYTHON=/home/veraxiao/miniconda3/envs/amplify/bin/python
GPU=1
PORT=5558

CHECKPOINTS=(
    robot5
    robot100
    robot200
    human40_robot5
    human100_robot5
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

# Kill both server and eval on exit/error
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
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

cd "$POLARIS_DIR"
source .venv/bin/activate

for FOLDER in "${CHECKPOINTS[@]}"; do
    CKPT=$CKPT_BASE/$FOLDER/amplify.pt
    RUN_FOLDER=$RUN_BASE/$FOLDER

    echo ""
    echo "========================================="
    echo "Evaluating: $FOLDER"
    echo "  ckpt: $CKPT"
    echo "  run_folder: $RUN_FOLDER"
    echo "========================================="

    mkdir -p "$RUN_FOLDER"

    # Start server in background
    CUDA_VISIBLE_DEVICES=$GPU $CONDA_PYTHON scripts/amplify_server.py \
        --amplify_root "$AMPLIFY_ROOT" \
        --ckpt_path "$CKPT" \
        --text_emb "$TEXT_EMB" \
        --port "$PORT" \
        --vis_tracks \
        > "$RUN_FOLDER/../server_${FOLDER}.log" 2>&1 &
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"

    wait_for_port "$PORT"

    # Start eval in background so we can monitor both processes
    CUDA_VISIBLE_DEVICES=$GPU python scripts/eval_lerobot.py \
        --policy.client AMPLIFY_FULLRES \
        --policy.host localhost \
        --policy.port "$PORT" \
        --policy.open_loop_horizon 8 \
        --environment DROID-PutRedCup-no-curtain \
        --run_folder "$RUN_FOLDER" \
        --rollouts 30 &
    EVAL_PID=$!
    echo "Eval PID: $EVAL_PID"

    # Monitor: if server dies before eval finishes, abort
    while kill -0 "$EVAL_PID" 2>/dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: Server (PID $SERVER_PID) died unexpectedly for $FOLDER. Killing eval."
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

    # Kill server
    echo "Killing server PID $SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    sleep 3

    if [ $EVAL_EXIT -ne 0 ]; then
        echo "ERROR: Eval exited with code $EVAL_EXIT for $FOLDER. Stopping."
        exit $EVAL_EXIT
    fi

    echo "Done: $FOLDER"
done

echo ""
echo "All GPU1 evaluations complete."
