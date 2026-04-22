# export PYTHONPATH="/home/haotian/RoboGen-sim2real":$PYTHONPATH
# export PYTHONPATH="/home/haotian/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy":$PYTHONPATH
# export PROJECT_DIR="/home/haotian/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy"
# export HYDRA_FULL_ERROR=1
# export PROJECT_DIR="~/RoboGen-sim2real"

# python scripts/eval_smith.py \
#     --environment DROID-PutRedCup-no-curtain \
#     --run-folder runs/hl_5_ll_5 \
#     --policy.policy_path \
#         /home/haotian/RoboGen-sim2real/exps/2026-03-23lift_red_cup_5/model_47501.pth \
#         /home/haotian/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0324_lift_red_cup_5/2026.03.25/01.32.22_train_dp3_robogen_open_door \
#     --policy.client Smith \
#     --rollouts 20

# python scripts/debug_smith.py \
#     --environment DROID-PutRedCup-no-curtain \
#     --run-folder runs/debug_smith \
#     --policy.policy_path \
#         /home/haotian/RoboGen-sim2real/exps/2026-03-23lift_red_cup_5/model_47501.pth \
#         /home/haotian/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0325_lift_red_cup_debug_smooth/2026.03.25/15.25.08_train_dp3_robogen_open_door \
#     --policy.client Smith \
#     --rollouts 5


# python scripts/eval_smith.py \
#     --environment DROID-PutRedCup-no-curtain \
#     --run-folder runs/hl_5_ll_50 \
#     --policy.policy_path \
#         /home/haotian/RoboGen-sim2real/exps/2026-03-23lift_red_cup_5/model_47501.pth \
#        /home/haotian/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0325_lift_red_cup_50/2026.03.25/18.43.29_train_dp3_robogen_open_door \
#     --policy.client Smith \
#     --rollouts 20

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python scripts/eval_point_policy.py \
    --environment DROID-PutRedCup-no-curtain \
    --run-folder runs/point_policy_human_25_robot_15 \
    --policy.client point_policy \
    --policy.port 8765 \
    --rollouts 30 \
    --max-episode-length 200 \
    --tqdm-position 0 \
    --device "cuda:0" &

CUDA_VISIBLE_DEVICES=1 python scripts/eval_point_policy.py \
    --environment DROID-PutRedCup-no-curtain \
    --run-folder runs/point_policy_human_35_robot_15 \
    --policy.client point_policy \
    --policy.port 8766 \
    --rollouts 30 \
    --max-episode-length 200 \
    --tqdm-position 1 \
    --device "cuda:0"  &

wait

echo "All evaluations complete."