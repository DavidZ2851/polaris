#!/bin/bash
# Sequentially evaluate all diffusion policy checkpoints (except robot100 and robot200)
# in pairs of 2, using port 5556 and 5557 in parallel.

CKPT_DIR="/data/vxiao/benchmark_new/polaris/policy_ckpt/diffusion_policy/red_mug/new_camera_calib"
SERVER_SCRIPT="/home/veraxiao/vxiao/human2robot/benchmark_new/polaris/scripts/diffusion_policy_server.py"
EVAL_SCRIPT="/home/veraxiao/vxiao/human2robot/benchmark_new/polaris/scripts/eval_lerobot.py"
RUN_BASE="/home/veraxiao/vxiao/human2robot/benchmark_new/polaris/runs/new_camera_calib"
ENVIRONMENT="DROID-PutRedCup-no-curtain"
ROLLOUTS=30
OPEN_LOOP_HORIZON=8
MAX_EPISODE_LENGTH=300

ROBODIFF_PY="/home/veraxiao/miniconda3/envs/robodiff/bin/python"
POLARIS_PY="/home/veraxiao/vxiao/human2robot/benchmark_new/polaris/.venv/bin/python"
DIFFPOL_DIR="/home/veraxiao/vxiao/human2robot/benchmark_new/diffusion_policy"
POLARIS_DIR="/home/veraxiao/vxiao/human2robot/benchmark_new/polaris"

# Collect checkpoints (exclude robot100 and robot200)
CKPTS=()
for f in "$CKPT_DIR"/*.ckpt; do
    name=$(basename "$f" .ckpt)
    if [[ "$name" == "robot100" || "$name" == "robot200" ]]; then
        continue
    fi
    CKPTS+=("$f")
done

echo "Found ${#CKPTS[@]} checkpoints to evaluate:"
for c in "${CKPTS[@]}"; do echo "  $(basename $c)"; done
echo ""

run_pair() {
    local ckpt1="$1"
    local ckpt2="$2"
    local name1=$(basename "$ckpt1" .ckpt)
    local name2=$(basename "$ckpt2" .ckpt)

    echo "======================================================"
    echo "Evaluating: $name1 (port 5556) | $name2 (port 5557)"
    echo "======================================================"

    # Server for ckpt1 on GPU 0, eval for ckpt1 on GPU 1
    (cd "$DIFFPOL_DIR" && CUDA_VISIBLE_DEVICES=0 "$ROBODIFF_PY" "$SERVER_SCRIPT" --ckpt_path "$ckpt1" --port 5556) &
    SERVER1_PID=$!

    # Server for ckpt2 on GPU 1, eval for ckpt2 on GPU 0
    (cd "$DIFFPOL_DIR" && CUDA_VISIBLE_DEVICES=1 "$ROBODIFF_PY" "$SERVER_SCRIPT" --ckpt_path "$ckpt2" --port 5557) &
    SERVER2_PID=$!

    echo "Waiting for servers to start..."
    sleep 15

    (cd "$POLARIS_DIR" && CUDA_VISIBLE_DEVICES=1 "$POLARIS_PY" "$EVAL_SCRIPT" \
        --policy.client DiffusionPolicy \
        --policy.host localhost \
        --policy.port 5556 \
        --policy.open_loop_horizon "$OPEN_LOOP_HORIZON" \
        --environment "$ENVIRONMENT" \
        --run_folder "$RUN_BASE/$name1" \
        --rollouts "$ROLLOUTS" \
        --max-episode-length "$MAX_EPISODE_LENGTH") &
    EVAL1_PID=$!

    (cd "$POLARIS_DIR" && CUDA_VISIBLE_DEVICES=0 "$POLARIS_PY" "$EVAL_SCRIPT" \
        --policy.client DiffusionPolicy \
        --policy.host localhost \
        --policy.port 5557 \
        --policy.open_loop_horizon "$OPEN_LOOP_HORIZON" \
        --environment "$ENVIRONMENT" \
        --run_folder "$RUN_BASE/$name2" \
        --rollouts "$ROLLOUTS" \
        --max-episode-length "$MAX_EPISODE_LENGTH") &
    EVAL2_PID=$!

    wait $EVAL1_PID
    echo "Eval 1 ($name1) done."
    wait $EVAL2_PID
    echo "Eval 2 ($name2) done."

    kill $SERVER1_PID 2>/dev/null
    kill $SERVER2_PID 2>/dev/null
    wait $SERVER1_PID 2>/dev/null
    wait $SERVER2_PID 2>/dev/null
    echo "Servers stopped."
    sleep 3
}

run_solo() {
    local ckpt="$1"
    local name=$(basename "$ckpt" .ckpt)

    echo "======================================================"
    echo "Evaluating solo: $name (server GPU1 / eval GPU0, port 5556)"
    echo "======================================================"

    (cd "$DIFFPOL_DIR" && CUDA_VISIBLE_DEVICES=1 "$ROBODIFF_PY" "$SERVER_SCRIPT" --ckpt_path "$ckpt" --port 5556) &
    SERVER_PID=$!

    echo "Waiting for server to start..."
    sleep 15

    (cd "$POLARIS_DIR" && CUDA_VISIBLE_DEVICES=0 "$POLARIS_PY" "$EVAL_SCRIPT" \
        --policy.client DiffusionPolicy \
        --policy.host localhost \
        --policy.port 5556 \
        --policy.open_loop_horizon "$OPEN_LOOP_HORIZON" \
        --environment "$ENVIRONMENT" \
        --run_folder "$RUN_BASE/$name" \
        --rollouts "$ROLLOUTS" \
        --max-episode-length "$MAX_EPISODE_LENGTH") &
    EVAL_PID=$!

    wait $EVAL_PID
    echo "Eval ($name) done."

    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    echo "Server stopped."
    sleep 3
}

# Process checkpoints in pairs
i=0
while [ $i -lt ${#CKPTS[@]} ]; do
    if [ $((i + 1)) -lt ${#CKPTS[@]} ]; then
        run_pair "${CKPTS[$i]}" "${CKPTS[$((i+1))]}"
        i=$((i + 2))
    else
        run_solo "${CKPTS[$i]}"
        i=$((i + 1))
    fi
done

echo ""
echo "All evaluations complete."
