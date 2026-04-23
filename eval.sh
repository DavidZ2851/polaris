# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python scripts/eval_policy.py \
#     --environment DROID-HangMug \
#     --run-folder runs/point_policy_hang_mug_200 \
#     --policy.client point_policy \
#     --policy.port 8765 \
#     --rollouts 30 \
#     --max-episode-length 300 \
#     --tqdm-position 0 \
#     --device "cuda:0"  

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python scripts/eval_policy.py \
    --environment DROID-PutRedCup-no-curtain \
    --run-folder runs/mimicplay_robot_40_human_0\
    --policy.client mimicplay \
    --policy.port 8766 \
    --rollouts 30 \
    --max-episode-length 300 \
    --tqdm-position 0 \
    --device "cuda:0" &

wait

echo "All evaluations complete."