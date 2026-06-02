#!/bin/bash

POLICY_PATH1="${POLICY_PATH:-/home/haotian/lerobot/outputs/train/pick_toys_simrobot_100_h300/checkpoints/last/pretrained_model}"
PORT1="${PORT:-8767}"

# POLICY_PATH2="${POLICY_PATH:-/home/haotian/lerobot/outputs/train/pick_toys_simrobot_100_h200/checkpoints/last/pretrained_model}"
# PORT2="${PORT:-8768}"

echo "Starting Ghost server..."
echo "  Policy:            $POLICY_PATH1"
echo "  Port:              $PORT1"
echo "  Open-loop horizon: $OPEN_LOOP_HORIZON"

CUDA_VISIBLE_DEVICES=1 python /home/haotian/polaris/src/polaris/server/ghost_server.py \
    --policy_path "$POLICY_PATH1" \
    --port "$PORT1" &
PID=$!


# echo "Starting Ghost server..."
# echo "  Policy:            $POLICY_PATH2"
# echo "  Port:              $PORT2"
# echo "  Open-loop horizon: $OPEN_LOOP_HORIZON"

# CUDA_VISIBLE_DEVICES=1 python /home/haotian/polaris/src/polaris/server/ghost_server.py \
#     --policy_path "$POLICY_PATH2" \
#     --port "$PORT2" &
# PID=$!

echo "Ghost server running: PID=$PID (port1 $PORT1, port2 $PORT2)"
echo "Press Ctrl+C to stop"

trap "echo 'Shutting down...'; kill $PID 2>/dev/null; exit" SIGINT SIGTERM

wait
echo "Ghost server shut down."
