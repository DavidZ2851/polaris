# export PYTHONPATH="/home/haotian/RoboGen-sim2real":$PYTHONPATH
# export PYTHONPATH="/home/haotian/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy":$PYTHONPATH
# export PROJECT_DIR="/home/haotian/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy"
# export HYDRA_FULL_ERROR=1
export PROJECT_DIR="~/RoboGen-sim2real"

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


python scripts/eval_smith.py \
    --environment DROID-PutRedCup-no-curtain \
    --run-folder runs/hl_5_ll_50 \
    --policy.policy_path \
        /home/haotian/RoboGen-sim2real/exps/2026-03-23lift_red_cup_5/model_47501.pth \
       /home/haotian/RoboGen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0325_lift_red_cup_50/2026.03.25/18.43.29_train_dp3_robogen_open_door \
    --policy.client Smith \
    --rollouts 20