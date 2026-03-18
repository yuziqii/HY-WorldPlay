#!/usr/bin/env bash
set -euo pipefail

# WAN interactive inference launcher.
# This script reuses wan/run_interactive.py and pre-fills local model paths.

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

PROMPT='A paved pathway leads towards a stone arch bridge spanning a calm body of water. Lush green trees and foliage line the path and the far bank of the water.'
IMAGE_PATH=./assets/img/test.png
OUT_PATH=./outputs/interactive.mp4

# Runtime config
FPS=24
MAX_CHUNKS=12          # 12 chunks x 16 frames / 24 fps = about 8.0s video
INITIAL_ACTION=w
N_INFERENCE_GPU=1
export CUDA_VISIBLE_DEVICES=7

# Optional: set to a number like 2.0 to auto-repeat the last action on timeout.
# Leave empty to wait indefinitely for your next command.
ACTION_TIMEOUT=

# WAN model paths
WORLDPLAY_ROOT=${WORLDPLAY_ROOT:-/your/path/to/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8}
WAN_TRANSFORMER_PATH="$WORLDPLAY_ROOT/wan_transformer"
WAN_CKPT_PATH="$WORLDPLAY_ROOT/wan_distilled_model/model.pt"

CMD=(
  torchrun
  --standalone
  --nproc_per_node="$N_INFERENCE_GPU"
  wan/run_interactive.py
  --ar_model_path "$WAN_TRANSFORMER_PATH"
  --ckpt_path "$WAN_CKPT_PATH"
  --prompt "$PROMPT"
  --image_path "$IMAGE_PATH"
  --fps "$FPS"
  --max_chunks "$MAX_CHUNKS"
  --initial_action "$INITIAL_ACTION"
  --out "$OUT_PATH"
)

if [[ -n "$ACTION_TIMEOUT" ]]; then
  CMD+=(--action_timeout "$ACTION_TIMEOUT")
fi

echo "Starting WAN interactive inference"
echo "  GPU:              $CUDA_VISIBLE_DEVICES"
echo "  Output:           $OUT_PATH"
echo "  Max chunks:       $MAX_CHUNKS"
echo "  Approx duration:  $(python - <<'PY'
max_chunks = 12
fps = 24
print(f"{max_chunks * 16 / fps:.1f}s")
PY
)"
echo
echo "Controls during inference: w / s / a / d / left / right / up / down"
echo "Press Enter to repeat the previous action, or q to stop and save."
echo

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${CMD[@]}"
