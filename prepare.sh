#!/bin/bash
# prepare.sh — run after `uv sync` to install packages that conflict with isaaclab
# Usage: bash prepare.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Fixing source files ===="
cat > "$SCRIPT_DIR/src/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/model/common/lr_scheduler.py" << 'PYEOF'
from typing import Union, Optional
from diffusers.optimization import (
    SchedulerType,
    Optimizer, TYPE_TO_SCHEDULER_FUNCTION,
    get_scheduler,
)
PYEOF
echo "  Fixed: lr_scheduler.py"

echo "==== Installing editable packages (no-deps) ===="
uv pip install -e "$SCRIPT_DIR/src/lerobot" --no-deps
uv pip install -e "$SCRIPT_DIR/src/SMITH_on_mimicgen" --no-deps
uv pip install -e "$SCRIPT_DIR/src/RoboGen_sim2real" --no-deps

echo "==== Writing .pth file for RoboGen_sim2real ===="
PTH_FILE="$SCRIPT_DIR/.venv/lib/python3.11/site-packages/robogen.pth"
cat > "$PTH_FILE" << 'PTH'
/home/haotian/polaris/src/RoboGen_sim2real
/home/haotian/polaris/src/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy
PTH
echo "  Written: $PTH_FILE"

echo "==== Verifying imports ===="
python -c "import lerobot; print('lerobot:', lerobot.__file__)"
python -c "import eval_smith_utils; print('eval_smith_utils:', eval_smith_utils.__file__)"
python -c "import test_PointNet2; print('test_PointNet2:', test_PointNet2.__file__)"
python -c "import train_ddp; print('train_ddp:', train_ddp.__file__)"

echo "==== Done ===="