#!/bin/bash

POLICY_PATH="${POLICY_PATH:-/home/haotian/lerobot/outputs/train/pick_place_red_mug_r40h100/checkpoints/last/pretrained_model}"
PORT="${PORT:-8768}"

echo "Starting Ghost server..."
echo "  Policy:            $POLICY_PATH"
echo "  Port:              $PORT"
echo "  Open-loop horizon: $OPEN_LOOP_HORIZON"

CUDA_VISIBLE_DEVICES=1 python /home/haotian/polaris/src/polaris/server/ghost_server.py \
    --policy_path "$POLICY_PATH" \
    --port "$PORT" &
PID=$!

echo "Ghost server running: PID=$PID (port $PORT)"
echo "Press Ctrl+C to stop"

trap "echo 'Shutting down...'; kill $PID 2>/dev/null; exit" SIGINT SIGTERM

wait
echo "Ghost server shut down."
