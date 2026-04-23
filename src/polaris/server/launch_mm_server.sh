#!/bin/bash


echo "Starting MimicPlay server on port 8767..."
CUDA_VISIBLE_DEVICES=0 python /home/haotian/polaris/src/polaris/server/mimicplay_server.py \
    --ll_policy_path /home/haotian/MimicPlay/trained_models_lowlevel/mimicplay_robot_40_human_0/20260416190231/models/model_epoch_3000.pth \
    --hl_policy_path /home/haotian/MimicPlay/trained_models_highlevel/mimicplay_robot_40_human_0/20260416113750/models/model_epoch_3000.pth \
    --calib_path /home/haotian/polaris/PolaRiS-Hub/put_red_cup_no_curtain/cam_calibration.json \
    --port 8766 &
PID1=$!

echo "Servers running: PID1=$PID1 (port 8765)"
echo "Press Ctrl+C to stop all servers"

trap "echo 'Shutting down...'; kill $PID1 2>/dev/null; exit" SIGINT SIGTERM

wait
echo "All servers shut down."