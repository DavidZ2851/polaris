"""
Server script for MimicPlay — runs in mimicplay env.

Run with:
    python serve_mimicplay.py \
    --policy_path /path/to/checkpoint.pth \
    --port 8766
"""
import sys
import pathlib


ROOT_DIR = str(pathlib.Path(__file__).parent.parent / "policy" / "MimicPlay" / "mimicplay")
sys.path.insert(0, ROOT_DIR)

import asyncio
import dataclasses
import logging
import socket

import msgpack
import msgpack_numpy as m
m.patch()

import numpy as np
import torch
import tyro
import websockets

import json
import mimicplay.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import cv2

def load_calibration(calib_path, camera_name="cam1"):
    """Load camera calibration from JSON file."""
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    
    cam_calib = calib[camera_name]
    intrinsic = np.array(cam_calib['intrinsic'])
    extrinsic = np.array(cam_calib['extrinsic'])
    
    return intrinsic, extrinsic

def _world_to_camera(points_world: np.ndarray, extrinsic_cam_to_world: np.ndarray) -> np.ndarray:
    T = np.linalg.inv(extrinsic_cam_to_world)
    homo = np.hstack([points_world, np.ones((len(points_world), 1))])
    return (T @ homo.T).T[:, :3]


def _project_to_image(points_cam: np.ndarray, intrinsic: np.ndarray):
    valid = points_cam[:, 2] > 0.01
    homo = (intrinsic @ points_cam.T).T
    pts2d = np.zeros((len(points_cam), 2))
    pts2d[valid, 0] = homo[valid, 0] / homo[valid, 2]
    pts2d[valid, 1] = homo[valid, 1] / homo[valid, 2]
    return pts2d, valid


def _draw_trajectory(img: np.ndarray, pts2d: np.ndarray, valid: np.ndarray,
                     color_start, color_end, thickness=2, radius=4) -> np.ndarray:
    out = img.copy()
    idxs = np.where(valid)[0]
    n = len(pts2d)
    for i in range(len(idxs) - 1):
        a, b = idxs[i], idxs[i + 1]
        t = a / max(n - 1, 1)
        color = tuple(int(color_start[j] * (1 - t) + color_end[j] * t) for j in range(3))
        cv2.line(out, tuple(pts2d[a].astype(int)), tuple(pts2d[b].astype(int)), color, thickness)
    for idx in idxs:
        t = idx / max(n - 1, 1)
        color = tuple(int(color_start[j] * (1 - t) + color_end[j] * t) for j in range(3))
        cv2.circle(out, tuple(pts2d[idx].astype(int)), radius, color, -1)
        cv2.circle(out, tuple(pts2d[idx].astype(int)), radius, (255, 255, 255), 1)
    return out


# apply_image_preprocess crops (1280×720) → 720×720 starting at (380, 0), then resizes to img_size
_CROP_X1, _CROP_Y1, _CROP_SIDE = 380, 0, 720
_DISPLAY_SIZE = 224

def _scale_intrinsic(intrinsic: np.ndarray) -> np.ndarray:
    scale = _DISPLAY_SIZE / _CROP_SIDE
    K = intrinsic.copy().astype(np.float64)
    K[0, 0] *= scale
    K[1, 1] *= scale
    K[0, 2] = (intrinsic[0, 2] - _CROP_X1) * scale
    K[1, 2] = (intrinsic[1, 2] - _CROP_Y1) * scale
    return K


@dataclasses.dataclass
class Args:
    hl_policy_path: str
    ll_policy_path: str
    calib_path: str
    port: int = 8766


def _patch_highlevel_path(ckpt_dict: dict, hl_path: str) -> dict:
    """Replace stale trained_highlevel_planner path inside a lowlevel checkpoint config."""
    import json
    cfg = json.loads(ckpt_dict["config"])
    cfg["algo"]["lowlevel"]["trained_highlevel_planner"] = hl_path
    ckpt_dict = dict(ckpt_dict)
    ckpt_dict["config"] = json.dumps(cfg)
    return ckpt_dict


class MimicPlayServer:
    def __init__(self, args: Args):
        self.device = TorchUtils.get_torch_device(try_to_use_cuda=True)

        logging.info("Loading MimicPlay lowlevel policy from: %s", args.ll_policy_path)
        ll_ckpt_dict = FileUtils.load_dict_from_checkpoint(args.ll_policy_path)
        logging.info("Patching highlevel planner path -> %s", args.hl_policy_path)
        ckpt_dict = _patch_highlevel_path(ll_ckpt_dict, args.hl_policy_path)

        # self.ll_policy, _ = FileUtils.policy_from_checkpoint(
        #     ckpt_path=args.ll_policy_path, device=self.device, verbose=True
        # )
        self.policy, _ = FileUtils.policy_from_checkpoint(
            ckpt_dict=ckpt_dict,
            device=self.device,
            verbose=True,
        )
        self.intrinsic, self.extrinsic = load_calibration(args.calib_path)

        self.policy.start_episode()
        self.step = 0
        logging.info("MimicPlay policy loaded.")

    def reset(self):
        self.policy.start_episode()
        self.step = 0
        logging.info("Policy reset.")

    @torch.no_grad()
    def infer(self, obs: dict, return_viz: bool = False) -> np.ndarray:
        action, future_traj = self.policy(ob=obs, return_guidance=True)

        self.step += 1
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        if return_viz:
            # obs["agentview_image"]: (1, C, H, W) float32 [0,1]
            img = obs["agentview_image"][0]
            img = (img.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            img = cv2.resize(img, (_DISPLAY_SIZE, _DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)

            intrinsic = _scale_intrinsic(self.intrinsic)
            extrinsic = self.extrinsic

            if future_traj is not None:
                curr_eef = obs["robot0_eef_pos"][0]
                if isinstance(curr_eef, torch.Tensor):
                    curr_eef = curr_eef.cpu().numpy()

                future_traj = future_traj.reshape(-1, 3)
                pred_pts3d  = np.vstack([curr_eef.reshape(1, 3), future_traj])
                pred_cam    = _world_to_camera(pred_pts3d, extrinsic)
                pred_2d, pred_valid = _project_to_image(pred_cam, intrinsic)
                img = _draw_trajectory(img, pred_2d, pred_valid,
                                       color_start=(0, 255, 255), color_end=(0, 0, 255))

                # current EEF dot (cyan)
                eef_cam = _world_to_camera(curr_eef.reshape(1, 3), extrinsic)
                eef_2d, eef_valid = _project_to_image(eef_cam, intrinsic)
                if eef_valid[0]:
                    pt = tuple(eef_2d[0].astype(int))
                    cv2.circle(img, pt, 8, (255, 255, 0), -1)
                    cv2.circle(img, pt, 8, (0, 0, 0), 2)

            cv2.putText(img, f"step={self.step}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            return action, img

        return action, None


def main(args: Args):
    server = MimicPlayServer(args)

    async def serve():
        shutdown_event = asyncio.Event()

        async def handle(websocket):
            async for message in websocket:
                data = msgpack.unpackb(message, raw=False)
                command = data.get("command", "infer")

                if command == "reset":
                    server.reset()
                    await websocket.send(msgpack.packb({"status": "reset"}, use_bin_type=True))

                elif command == "infer":
                    obs = data["obs"]
                    return_viz =  data["return_viz"]
                    action, viz = server.infer(obs, return_viz=return_viz)
                    await websocket.send(msgpack.packb({"action": action, "viz": viz}, use_bin_type=True))
                    

                elif command == "shutdown":
                    logging.info("Received shutdown signal.")
                    await websocket.send(msgpack.packb({"status": "shutdown"}, use_bin_type=True))
                    shutdown_event.set()
                    return

                else:
                    await websocket.send(msgpack.packb(
                        {"error": f"Unknown command: {command}"},
                        use_bin_type=True,
                    ))

        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logging.info("Serving on host=%s ip=%s port=%d", hostname, local_ip, args.port)
        async with websockets.serve(handle, "0.0.0.0", args.port, max_size=100 * 1024 * 1024):
            await shutdown_event.wait()
        logging.info("Server shut down.")

    asyncio.run(serve())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))