import gymnasium as gym
from polaris.environments.manager_based_rl_splat_environment import (
    ManagerBasedRLSplatEnv,
)
from polaris.environments.droid_cfg import EnvCfg as DroidCfg
from isaaclab.envs import ManagerBasedRLEnv

# Import rubric system
from polaris.environments.rubrics import Rubric
from polaris.utils import DATA_PATH
import polaris.environments.rubrics.checkers as checkers


# =============================================================================
# Environment Registration
# =============================================================================

gym.register(
    id='DROID-BlockStackKitchen',
    entry_point=ManagerBasedRLSplatEnv,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "block_stack_kitchen/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("green_cube", threshold=0.2),
                checkers.reach("wood_cube", threshold=0.2),
                (checkers.lift("green_cube", default_height=0.06, threshold=0.03), [0]),
                (checkers.lift("wood_cube", default_height=0.06, threshold=0.03), [1]),
                (checkers.is_within_xy("green_cube", "tray", 0.8), [2]),
                (checkers.is_within_xy("wood_cube", "tray", 0.8), [3]),
                (checkers.is_within_xy("green_cube", "wood_cube", 0.5), [4, 5]),
            ]
        ),
    },
    disable_env_checker=True,
    order_enforce=False,
)


gym.register(
    id="DROID-FoodBussing",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "food_bussing/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("ice_cream_", threshold=0.2),
                checkers.reach("grapes", threshold=0.2),
                (checkers.lift("ice_cream_", threshold=0.06), [0]),
                (checkers.lift("grapes", threshold=0.06), [1]),
                (
                    checkers.is_within_xy("ice_cream_", "bowl", percent_threshold=0.8),
                    [2],
                ),
                (checkers.is_within_xy("grapes", "bowl", percent_threshold=0.8), [3]),
            ]
        ),
    },
)

gym.register(
    id="DROID-FoodBussing-V2",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "food_blussing_v2/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("battery", threshold=0.2),
                # checkers.reach("grapes", threshold=0.2),
                # (checkers.lift("ice_cream_", threshold=0.06), [0]),
                # (checkers.lift("grapes", threshold=0.06), [1]),
                # (
                #     checkers.is_within_xy("ice_cream_", "bowl", percent_threshold=0.8),
                #     [2],
                # ),
                # (checkers.is_within_xy("grapes", "bowl", percent_threshold=0.8), [3]),
            ]
        ),
    },
)

gym.register(
    id="DROID-PanClean",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "pan_clean/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("sponge", threshold=0.2),
                (checkers.lift("sponge", threshold=0.09, default_height=0.0), [0]),
                (checkers.is_within_xy("sponge", "pan", percent_threshold=0.8), [1]),
            ]
        ),
    },
)


gym.register(
    id="DROID-MoveLatteCup",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "move_latte_cup/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("latteartcup_eval", threshold=0.2),
                (checkers.lift("latteartcup_eval", threshold=0.04), [0]),
                (checkers.is_within_xy("latteartcup_eval", "cuttingboard_eval", percent_threshold=0.8), [1]),
            ]
        ),
    },
)

gym.register(
    id="DROID-OrganizeTools",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "organize_tools/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("scissor", threshold=0.2),
                (checkers.lift("scissor", threshold=0.04), [0]),
                (checkers.is_within_xy("scissor", "container_01", percent_threshold=0.8), [1]),
            ]
        ),
    },
)

gym.register(
    id="DROID-TapeIntoContainer",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "tape_into_container/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("tape_00", threshold=0.2),
                (checkers.lift("tape_00", threshold=0.04), [0]),
                (checkers.is_within_xy("tape_00", "container_02", percent_threshold=0.8), [1]),
            ]
        ),
    },
)

gym.register(
    id="DROID-PutRedCup-no-curtain",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "put_red_cup_no_curtain/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("red_cup", threshold=0.2),
                (checkers.lift("red_cup", threshold=0.04), [0]),
                (checkers.is_within_xy("red_cup", "blue_plate", percent_threshold=0.8), [1]),
            ]
        ),
    },
)

gym.register(
    id="DROID-StackBlock",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "stack_block/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("yellow_block", threshold=0.02),
                (checkers.lift("yellow_block", threshold=0.04), [0]),
                checkers.reach("green_block", threshold=0.02),
                (checkers.is_within_xy("yellow_block", "green_block", percent_threshold=0.8), [1]),
            ]
        ),
    },
)

gym.register(
    id="DROID-InsertDonut",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "insert_donut/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("blue_donut", threshold=0.2),
                (checkers.lift("blue_donut", threshold=0.15), [0]),
                (checkers.is_within_xy("blue_donut", "bar", percent_threshold=0.1, open_finger_threshold=2), [1]),
                (checkers.is_inserted("blue_donut", "bar", xy_threshold=0.06, z_threshold=0.1), [2]),
            ]
        ),
    },
)

gym.register(
    id="DROID-HangMug",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "hang_mug/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.reach("red_mug", threshold=0.2),
                (checkers.lift("red_mug", threshold=0.05), [0]),
                (checkers.is_close_xy("red_mug", "mug_tree", xy_dist_threshold=0.25), [1]),
                (checkers.is_hung("red_mug", "mug_tree", xy_threshold=0.15, z_threshold=0.1, stable_steps=3, vel_threshold=0.05), [2]),
            ]
        ),
    },
)

gym.register(
    id="DROID-PickPlaceToys",
    entry_point=ManagerBasedRLSplatEnv,
    disable_env_checker=True,
    order_enforce=False,
    kwargs={
        "env_cfg_entry_point": DroidCfg,
        "usd_file": str(DATA_PATH / "pick_place_toys/scene.usda"),
        "rubric": Rubric(
            criteria=[
                checkers.is_within_xy("blue_toy_large", "pink_box", percent_threshold=1),
                checkers.is_within_xy("orange_toy", "pink_box", percent_threshold=1),
                checkers.is_within_xy("yellow_toy", "pink_box", percent_threshold=1),
            ]
        ),
    },
)
