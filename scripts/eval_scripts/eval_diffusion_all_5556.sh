#!/bin/bash
# Evaluate checkpoints sequentially on port 5556.
# Server on GPU 0, Eval on GPU 1.
# Run in one terminal while eval_diffusion_all_5557.sh runs in another.
#
# Checkpoints: human100_robot40, human25_robot5, human40_robot5, robot40

CKPT_DIR="/data/vxiao/benchmark_new/polaris/policy_ckpt/diffusion_policy/red_mug/new_camera_calib"
SERVER_SCRIPT="/home/veraxiao/vxiao/human2robot/benchmark_new/polaris/scripts/diffusion_policy_server.py"
EVAL_SCRIPT="/home/veraxiao/vxiao/human2robot/benchmark_new/polaris/scripts/eval_lerobot.py"
RUN_BASE="/home/veraxiao/vxiao/human2robot/benchmark_new/polaris/runs/new_camera_calib"
ENVIRONMENT="DROID-PutRedCup-no-curtain"
ROLLOUTS=30
OPEN_LOOP_HORIZON=8
MAX_EPISODE_LENGTH=300
PORT=5556
SERVER_GPU=0
EVAL_GPU=1
# Both evals on GPU1 (clean), both servers on GPU0

ROBODIFF_PY="/home/veraxiao/miniconda3/envs/robodiff/bin/python"
POLARIS_PY="/home/veraxiao/vxiao/human2robot/benchmark_new/polaris/.venv/bin/python"
DIFFPOL_DIR="/home/veraxiao/vxiao/human2robot/benchmark_new/diffusion_policy"
POLARIS_DIR="/home/veraxiao/vxiao/human2robot/benchmark_new/polaris"

CKPTS=(
    "$CKPT_DIR/human100_robot40.ckpt"
    "$CKPT_DIR/human25_robot5.ckpt"
    "$CKPT_DIR/human40_robot5.ckpt"
    "$CKPT_DIR/robot40.ckpt"
)

echo "Checkpoints (port $PORT, server GPU$SERVER_GPU, eval GPU$EVAL_GPU):"
for c in "${CKPTS[@]}"; do echo "  $(basename $c)"; done
echo ""

pkill -f "diffusion_policy_server.py --port $PORT" 2>/dev/null
sleep 2

for ckpt in "${CKPTS[@]}"; do
    name=$(basename "$ckpt" .ckpt)
    csv="$RUN_BASE/$name/eval_results.csv"
    if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -ge 31 ]; then
        echo "Skipping $name (already has 30 rollouts)"
        continue
    fi
    echo "======================================================"
    echo "Evaluating: $name"
    echo "======================================================"

    (cd "$DIFFPOL_DIR" && CUDA_VISIBLE_DEVICES=$SERVER_GPU "$ROBODIFF_PY" "$SERVER_SCRIPT" \
        --ckpt_path "$ckpt" --port "$PORT") &
    SERVER_PID=$!

    echo "Waiting for server to start..."
    sleep 15

    (cd "$POLARIS_DIR" && CUDA_VISIBLE_DEVICES=$EVAL_GPU "$POLARIS_PY" "$EVAL_SCRIPT" \
        --policy.client DiffusionPolicy \
        --policy.host localhost \
        --policy.port "$PORT" \
        --policy.open_loop_horizon "$OPEN_LOOP_HORIZON" \
        --environment "$ENVIRONMENT" \
        --run_folder "$RUN_BASE/$name" \
        --rollouts "$ROLLOUTS" \
        --max-episode-length "$MAX_EPISODE_LENGTH")

    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    echo "Done: $name. Server stopped."
    echo ""
    sleep 3
done

echo "All port $PORT evaluations complete."
