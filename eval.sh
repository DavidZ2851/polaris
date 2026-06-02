# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python scripts/eval_policy.py \
#     --environment DROID-PutRedCup-no-curtain\
#     --run-folder runs_seed100/point_policy_pick_place_red_mug_r40h100 \
#     --policy.client point_policy \
#     --policy.port 8765 \
#     --rollouts 30 \
#     --max-episode-length 450 \
#     --tqdm-position 0 \
#     --device "cuda:0"  \
#     --seed 100 &

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python scripts/eval_policy.py \
    --environment DROID-PickPlaceToys\
    --run-folder runs_seed0/point_policy_pick_toys_r100h0 \
    --policy.client point_policy \
    --policy.port 8765 \
    --rollouts 39 \
    --max-episode-length 600 \
    --tqdm-position 0 \
    --device "cuda:0"  \
    --seed 0 &
   
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python scripts/eval_policy.py \
    --environment DROID-PickPlaceToys\
    --run-folder runs_seed100/point_policy_pick_toys_r100h0 \
    --policy.client point_policy \
    --policy.port 8766 \
    --rollouts 39 \
    --max-episode-length 600 \
    --tqdm-position 1 \
    --device "cuda:1"  \
    --seed 100 &

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python scripts/eval_policy.py \
#     --environment DROID-InsertDonut\
#     --run-folder runs_seed100/point_policy_insert_donut_r100h0_2 \
#     --policy.client point_policy \
#     --policy.port 8766 \
#     --rollouts 30 \
#     --max-episode-length 300 \
#     --tqdm-position 1 \
#     --device "cuda:1"  \
#     --seed 100 &

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python scripts/eval_policy.py \
#     --environment DROID-PickPlaceToys \
#     --run-folder runs/point_policy_pick_toys_r1003200 \
#     --policy.client point_policy \
#     --policy.port 8766 \
#     --rollouts 30 \
#     --max-episode-length 600 \
#     --tqdm-position 1 \
#     --device "cuda:1" &

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python scripts/eval_policy.py \
#     --environment DROID-StackBowls\
#     --run-folder runs/ghost_stack_bowls_r100h300 \
#     --policy.client ghost \
#     --policy.port 8767 \
#     --rollouts 30 \
#     --max-episode-length 250 \
#     --tqdm-position 0 \
#     --device "cuda:0" &

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python scripts/eval_policy.py \
#     --environment DROID-PickPlaceToys\
#     --run-folder runs/ghost_pick_place_toys_r100h300 \
#     --policy.client ghost \
#     --policy.port 8767 \
#     --rollouts 30 \
#     --max-episode-length 600 \
#     --tqdm-position 0 \
#     --device "cuda:0" &

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python scripts/eval_policy.py \
#     --environment DROID-PickPlaceToys\
#     --run-folder runs/ghost_pick_place_toys_r100h200 \
#     --policy.client ghost \
#     --policy.port 8768 \
#     --rollouts 30 \
#     --max-episode-length 600 \
#     --tqdm-position 0 \
#     --device "cuda:1" &

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python scripts/eval_policy.py \
#     --environment DROID-StackBowls \
#     --run-folder runs/point_policy_stack_bowls_r100_h300_2 \
#     --policy.client point_policy \
#     --policy.port 8766 \
#     --rollouts 30 \
#     --max-episode-length 150 \
#     --tqdm-position 1 \
#     --device "cuda:1" &

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python scripts/eval_policy.py \
#     --environment DROID-PutRedCup-no-curtain \
#     --run-folder runs/ghost_pick_place_red_mug_r5h100 \
#     --policy.client ghost \
#     --policy.port 8767 \
#     --rollouts 30 \
#     --max-episode-length 150 \
#     --tqdm-position 0 \
#     --device "cuda:0" &

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python scripts/eval_policy.py \
#     --environment DROID-PutRedCup-no-curtain \
#     --run-folder runs/ghost_pick_place_red_mug_r5h100 \
#     --policy.client ghost \
#     --policy.port 8768 \
#     --rollouts 30 \
#     --max-episode-length 150 \
#     --tqdm-position 1 \
#     --device "cuda:1" &

wait

echo "All evaluations complete."