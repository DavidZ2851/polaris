CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python scripts/eval_policy.py \
    --environment DROID-PickPlaceToys\
    --run-folder runs/point_policy \
    --policy.client point_policy \
    --policy.port 8765 \
    --rollouts 30 \
    --max-episode-length 600 \
    --tqdm-position 0 \
    --device "cuda:0"  &

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python scripts/eval_policy.py \
    --environment DROID-StackBowls\
    --run-folder runs/ghost \
    --policy.client ghost \
    --policy.port 8766 \
    --rollouts 30 \
    --max-episode-length 300 \
    --tqdm-position 0 \
    --device "cuda:0"  &



wait

echo "All evaluations complete."