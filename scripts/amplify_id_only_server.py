"""
AMPLIFY inference server using only the InverseDynamics model (no MotionTokenizer / ForwardDynamics).
Use this when the ID model was trained with cond_on_tracks=false.

Run in the 'amplify' conda env BEFORE running eval in the polaris uv env.

Usage:
    conda activate amplify
    cd /home/veraxiao/vxiao/human2robot/benchmark_new/AMPLIFY
    python ../polaris/scripts/amplify_id_only_server.py \\
        --id_ckpt  ../polaris/policy_ckpt/amplify_noimageresize/robot_onlyinverse_notrack/inverse.pt \\
        --mt_ckpt  ../polaris/policy_ckpt/amplify_noimageresize/robot_mt_fd100/motion.pt \\
        --action_stats ../polaris/policy_ckpt/amplify/pick_mug_action_stats_rot6d.npy \\
        --port 5557

The --mt_ckpt is only used to read the MotionTokenizer config (dimensions, horizon, etc.).
The MT and FD models are never instantiated or loaded.

Request format  (pickle):
    {"image": np.ndarray (v, H, W, 3) float32 [0,1],
     "proprio": np.ndarray (10,) float32}
    or {"reset": True}

Response format (pickle):
    {"action_chunk": np.ndarray (action_horizon, 10) float32}
    actions are in denormalized joint-position space.
"""

import argparse
import pickle
import sys
import pathlib

import numpy as np
import torch
import zmq
from einops import rearrange

# Parse --amplify_root early so imports work before argparse runs fully
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--amplify_root", type=str, default=None)
_pre_args, _ = _pre.parse_known_args()

ROOT_DIR = _pre_args.amplify_root or str(pathlib.Path(__file__).parent.parent.parent / "policy/amplify_polaris")
sys.path.insert(0, ROOT_DIR)

from omegaconf import OmegaConf
from amplify.models.encoders.vision_encoders import VisionEncoder
from amplify.models.inverse_dynamics import InverseDynamics


def load_models(args):
    device = torch.device("cuda:0")

    # Load motion tokenizer config (weights not loaded — only needed for dimension info)
    print(f"Loading MotionTokenizer config from {args.mt_ckpt} (config only, no weights)...")
    mt_ckpt = torch.load(args.mt_ckpt, map_location="cpu", weights_only=False)
    motion_tokenizer_cfg = OmegaConf.create(mt_ckpt["config"])

    # Load ID config and weights
    print(f"Loading InverseDynamics from {args.id_ckpt}...")
    id_ckpt = torch.load(args.id_ckpt, map_location="cpu", weights_only=False)
    id_cfg = OmegaConf.create(id_ckpt["config"])

    # Sanity check
    assert not id_cfg.cond_on_tracks, "This server is for cond_on_tracks=false models only"
    assert not id_cfg.cond_on_text,   "This server does not support text conditioning"

    # Vision encoder
    vision_encoder_cfg = OmegaConf.to_container(id_cfg.vision_encoder, resolve=True)
    img_h, img_w = vision_encoder_cfg["img_size"][0], vision_encoder_cfg["img_size"][1]
    print(f"Vision encoder img_size: {img_h}×{img_w}")

    img_encoder = VisionEncoder(**vision_encoder_cfg).eval().to(device)

    # Inverse dynamics
    inverse_dynamics = InverseDynamics(motion_tokenizer_cfg, id_cfg)
    state = id_ckpt["model"]
    # Strip 'inverse_dynamics.' prefix if present (saved from full AMPLIFY)
    if any(k.startswith("inverse_dynamics.") for k in state):
        state = {k[len("inverse_dynamics."):]: v for k, v in state.items() if k.startswith("inverse_dynamics.")}
    load_res = inverse_dynamics.load_state_dict(state, strict=False)
    if load_res.missing_keys or load_res.unexpected_keys:
        print("[Server] ID load_state_dict missing:", load_res.missing_keys)
        print("[Server] ID load_state_dict unexpected:", load_res.unexpected_keys)
    inverse_dynamics.eval().to(device)

    print(f"Models loaded. img_size={img_h}×{img_w}, action_dim={id_cfg.action_dim}, "
          f"action_horizon={motion_tokenizer_cfg.true_horizon}")
    return img_encoder, inverse_dynamics, img_h, img_w, device


def denormalize_actions(actions: np.ndarray, action_min: np.ndarray, action_max: np.ndarray) -> np.ndarray:
    return (actions + 1.0) / 2.0 * (action_max - action_min) + action_min


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amplify_root", type=str, default=None,
                        help="Path to AMPLIFY repo root (default: ../../../policy/amplify_polaris)")
    parser.add_argument("--id_ckpt",      type=str, required=True, help="Path to InverseDynamics checkpoint")
    parser.add_argument("--mt_ckpt",      type=str, required=True, help="Path to MotionTokenizer checkpoint (config only)")
    parser.add_argument("--action_stats", type=str, default=None, help="Path to .npy action stats dict (min, max). Omit if model was trained with normalize_actions: null")
    parser.add_argument("--port",         type=int, default=5557)
    args = parser.parse_args()

    img_encoder, inverse_dynamics, img_h, img_w, device = load_models(args)

    if args.action_stats:
        action_stats = np.load(args.action_stats, allow_pickle=True).item()
        action_min = action_stats["min"].astype(np.float32)
        action_max = action_stats["max"].astype(np.float32)
        print(f"Action stats loaded: min={action_min.round(3)}, max={action_max.round(3)}")
    else:
        action_min = action_max = None
        print("No action stats provided — model output used as-is (no denormalization)")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")
    print(f"AMPLIFY ID-only server listening on port {args.port}")

    while True:
        raw = socket.recv()
        try:
            request = pickle.loads(raw)
        except Exception as e:
            print(f"[Server] Deserialize error: {e}")
            socket.send(pickle.dumps({"status": "error", "message": str(e)}))
            continue

        if request.get("reset"):
            socket.send(pickle.dumps({"status": "ok"}))
            continue

        # image: (v, H, W, 3) float32 [0,1]
        # proprio: (10,) float32
        image_np  = request["image"]    # (v, H, W, 3)
        proprio_np = request["proprio"]  # (10,)

        image_t   = torch.from_numpy(image_np).float().to(device)    # (v, H, W, 3)
        proprio_t = torch.from_numpy(proprio_np).float().to(device)  # (10,)

        b, v = 1, image_t.shape[0]
        # Add batch dim: (1, v, H, W, 3)
        image_t = image_t.unsqueeze(0)

        with torch.no_grad():
            # Encode images: (b*v, H, W, 3) → (b*v, t, d)
            img = rearrange(image_t, 'b v h w c -> (b v) h w c')
            img_tokens = img_encoder(img)                                      # (b*v, t, d)
            img_tokens_id = rearrange(img_tokens, '(b v) t d -> b t (v d)', b=b, v=v)  # (1, t, v*d)

            input_dict = {
                "img_tokens":     img_tokens_id,
                "proprioception": proprio_t.unsqueeze(0).unsqueeze(1),  # (1, 1, 10)
            }

            actions = inverse_dynamics.act(input_dict)  # (1, action_horizon, 10) normalized

        action_chunk_norm = actions.squeeze(0).cpu().numpy()   # (action_horizon, 10)
        if action_min is not None:
            action_chunk = denormalize_actions(action_chunk_norm, action_min, action_max)
        else:
            action_chunk = action_chunk_norm

        socket.send(pickle.dumps({"action_chunk": action_chunk}))


if __name__ == "__main__":
    main()
