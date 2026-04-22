# Human2Robot Benchmark

This repo contains tools for generating simulation datasets, testing environments, and evaluating policies in IsaacLab.

---

## Environment Setup

### 1. Clone the repository (recursively)

```bash
git clone --recursive git@github.com:DavidZ2851/polaris.git
cd polaris
```

If you cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

### 2. Install dependencies with uv

If you don't have uv installed, see [installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

By default CUDA 13 is supported. If you have an older CUDA version, downgrade torch/torchvision in `pyproject.toml` accordingly.

```bash
uv sync
```

### 3. Activate the environment

```bash
source .venv/bin/activate
```

### 4. Download PolaRiS environments

```bash
uvx hf download owhan/PolaRiS-Hub --repo-type=dataset --local-dir ./PolaRiS-Hub
```

### 5. Install ffmpeg (for saving videos)

```bash
sudo apt install ffmpeg
```

---

## 1. Generating Datasets

Use `scripts/generate_dataset.py` to collect demonstration data using a motion planner inside the sim.

```bash
python scripts/generate_dataset.py \
    --save_dir /home/$USER/polaris/debug
```

Key options (from `DataArgs` in `src/polaris/config.py`):

| Flag | Default | Description |
|------|---------|-------------|
| `--environment` | `DROID-PutRedCup-no-curtain` | IsaacLab environment to use |
| `--save_dir` | required | Directory to save the dataset |
| `--num_episodes` | 50 | Number of episodes to collect |
| `--max_attempts` | 100 | Max planning attempts per episode |
| `--headless` | True | Run without GUI |

---

## 2. Testing the Environment

Verify the environment loads and renders correctly:

```bash
python scripts/test_env.py
```

---

## 3. Policy Evaluation

Evaluation follows a **client-server** architecture:

- **Server** (`src/polaris/server/`): loads the policy checkpoint and serves actions over ZMQ
- **Client** (`src/polaris/client/`): connects to the server, sends observations, receives actions
- **Eval script** (`scripts/eval_policy.py`): runs the IsaacLab environment and drives the eval loop

### Available Servers

| Server | Conda Env | Script |
|--------|-----------|--------|
| Diffusion Policy | `robodiff` | `src/polaris/server/diffusion_policy_server.py` |
| AMPLIFY | `amplify` | `src/polaris/server/amplify_server.py` |
| LeRobot Diffusion | `robodiff` | `src/polaris/server/lerobot_diffusion_server.py` |

### Available Clients

Clients are registered by name and selected via `--policy.client`:

| Client Name | File |
|-------------|------|
| `DiffusionPolicy` | `src/polaris/client/diffusion_policy_client.py` |
| `AMPLIFY` | `src/polaris/client/amplify_client.py` |
| `LeRobotDiffusion` | `src/polaris/client/lerobot_diffusion_zmq_client.py` |
| `DroidJointPos` | `src/polaris/client/droid_jointpos_client.py` |

### Camera Usage per Policy

Camera names in `obs["splat"]` come from the USD scene prim names (e.g. `cam0`, `cam1`, `wrist_cam`).

| Policy | Cameras used | Notes |
|--------|-------------|-------|
| `DiffusionPolicy` | `cam1` | Front camera only (1 view, `(H, W, 3)`) |
| `AMPLIFY` | `cam0`, `cam1` | Two views stacked as `(2, H, W, 3)`; `cam0` = front, `cam1` = left |
| `DroidJointPos` | `external_cam`, `wrist_cam` | Uses sim camera names directly |

---

### Example: Diffusion Policy

**Step 1 — Start the server** (in a separate terminal, `robodiff` env):

```bash
cd ~/vxiao/human2robot/benchmark_new/polaris
conda activate robodiff
python src/polaris/server/diffusion_policy_server.py \
    --ckpt_path policy_ckpt/diffusion_policy/red_mug/new_camera_calib/human40_robot40.ckpt \
    --port 5557
```

**Step 2 — Run evaluation** (polaris `.venv`):

```bash
cd ~/vxiao/human2robot/benchmark_new/polaris
source .venv/bin/activate
python scripts/eval_policy.py \
    --policy.client DiffusionPolicy \
    --policy.host localhost \
    --policy.port 5557 \
    --policy.open_loop_horizon 8 \
    --environment DROID-PutRedCup-no-curtain \
    --run_folder runs/new_camera_calib/human40_robot40 \
    --rollouts 30
```

---

### Example: AMPLIFY

**Step 1 — Bundle checkpoint** (if not already bundled):

```bash
python src/polaris/policy/amplify/amplify/bundle_amplify.py \
    --mt_ckpt  /data/vxiao/benchmark_new/polaris/policy_ckpt/amplify_new/<folder>/motion.pt \
    --fd_ckpt  /data/vxiao/benchmark_new/polaris/policy_ckpt/amplify_new/<folder>/forward.pt \
    --id_ckpt  /data/vxiao/benchmark_new/polaris/policy_ckpt/amplify_new/<folder>/inverse.pt \
    --save-to  /data/vxiao/benchmark_new/polaris/policy_ckpt/amplify_new/<folder>/amplify.pt \
    --overwrite
```

**Step 2 — Start the server** (`amplify` conda env):

```bash
cd ~/vxiao/human2robot/benchmark_new/polaris
conda activate amplify
python src/polaris/server/amplify_server.py \
    --ckpt_path /data/vxiao/benchmark_new/polaris/policy_ckpt/amplify_new/<folder>/amplify.pt \
    --text_emb  /data/vxiao/benchmark_new/polaris/policy_ckpt/amplify_new/pick_mug_text_emb.npy \
    --port 5557 \
    --vis_tracks
```

**Step 3 — Run evaluation** (polaris `.venv`):

```bash
cd ~/vxiao/human2robot/benchmark_new/polaris
source .venv/bin/activate
python scripts/eval_policy.py \
    --policy.client AMPLIFY \
    --policy.host localhost \
    --policy.port 5557 \
    --policy.open_loop_horizon 8 \
    --environment DROID-PutRedCup-no-curtain \
    --run_folder runs/new_camera_calib/amplify/<folder> \
    --rollouts 30
```

---

## Key `eval_policy.py` Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--policy.client` | required | Client name (e.g. `DiffusionPolicy`, `AMPLIFY`) |
| `--policy.host` | required | Server host (e.g. `localhost`) |
| `--policy.port` | required | Server port |
| `--policy.open_loop_horizon` | required | How many steps to execute per action chunk |
| `--environment` | required | IsaacLab environment ID |
| `--run_folder` | required | Where to save videos and `eval_results.csv` |
| `--rollouts` | required | Number of episodes to evaluate |
| `--max-episode-length` | 300 | Max steps per episode |
| `--seed` | 42 | Random seed |

Results are saved to `<run_folder>/eval_results.csv` and `<run_folder>/episode_N.mp4`. Evaluation supports **resuming** — if the CSV already exists, completed episodes are skipped automatically.

---

## Large File Setup

Several large files are not tracked in git and must be downloaded from Google Drive:

**[Download from Google Drive](https://drive.google.com/drive/u/0/folders/17L0uBJ1sspTK_x4m-4raml4SKH1TifHc)**

After downloading, unzip each file and place it in the correct location as described below.

---

### `content.zip` → `src/curobo/src/curobo/content/`

Contains cuRobo asset files (robot meshes, scene assets).

```bash
unzip content.zip -d /
# Files extract to: src/curobo/src/curobo/content/
```

> Note: the zip stores absolute paths so unzipping to `/` places files at the correct location automatically.

---

### `put_red_cup_no_curtain.zip` → `PolaRiS-Hub/put_red_cup_no_curtain/`

Contains the put-red-cup task environment assets (meshes, USD files, splat, config).

```bash
unzip put_red_cup_no_curtain.zip -d /tmp/polaris_env
cp -r /tmp/polaris_env/.../put_red_cup_no_curtain PolaRiS-Hub/
```

Expected result:

```
PolaRiS-Hub/put_red_cup_no_curtain/
├── assets/
├── textures/
├── cam_calibration.json
├── initial_conditions.json
├── scene.json
├── scene.usda
└── task_config.yaml
```

---

### `nvidia_droid.zip` → `PolaRiS-Hub/nvidia_droid/`

Contains the NVIDIA DROID robot USD and segmented Gaussian splat files.

```bash
unzip nvidia_droid.zip -d /
# Files extract to: PolaRiS-Hub/nvidia_droid/
```

Expected result:

```
PolaRiS-Hub/nvidia_droid/
├── franka_robotiq_2f_85_flattened.usd
├── splat.ply
└── SEGMENTED/
    └── *.ply  (per-link splat files)
```

---

## Adding a New Policy

1. Create a new client in `src/polaris/client/` inheriting from `InferenceClient` and decorate with `@InferenceClient.register(client_name="YourPolicy")`
2. Create a corresponding server in `src/polaris/server/` that loads your checkpoint and serves actions over ZMQ
3. Import the new client in `src/polaris/client/__init__.py`
4. Run with `--policy.client YourPolicy`
