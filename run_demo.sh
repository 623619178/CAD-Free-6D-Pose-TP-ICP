#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAM6D_ROOT="$ROOT_DIR/SAM-6D"
TPICP_ROOT="$ROOT_DIR/tpicp"
EXAMPLE_DIR="$ROOT_DIR/examples/can"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/can_demo}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

required_files=(
  "$SAM6D_ROOT/Instance_Segmentation_Model/checkpoints/segment-anything/sam_vit_h_4b8939.pth"
  "$SAM6D_ROOT/Instance_Segmentation_Model/checkpoints/dinov2/dinov2_vitl14_pretrain.pth"
  "$SAM6D_ROOT/Pose_Estimation_Model/checkpoints/sam-6d-pem-base.pth"
  "$EXAMPLE_DIR/proxy_mesh.ply"
  "$SAM6D_ROOT/Data/Example/rgb.png"
  "$SAM6D_ROOT/Data/Example/depth.png"
  "$SAM6D_ROOT/Data/Example/camera.json"
)

missing=()
for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    missing+=("$path")
  fi
done

if (( ${#missing[@]} )); then
  printf 'Missing required files:\n' >&2
  printf '  %s\n' "${missing[@]}" >&2
  printf '\nDownload the official SAM-6D checkpoints as documented in README.md.\n' >&2
  exit 2
fi

if ! command -v blenderproc >/dev/null 2>&1; then
  echo "blenderproc is not available in PATH; activate the SAM-6D environment first." >&2
  exit 2
fi

if [[ -e "$OUTPUT_DIR" && "${OVERWRITE:-0}" != "1" ]]; then
  echo "Output already exists: $OUTPUT_DIR" >&2
  echo "Use a new OUTPUT_DIR or rerun with OVERWRITE=1." >&2
  exit 2
fi

extra_args=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  extra_args+=(--overwrite)
fi

export PYTHONPATH="$TPICP_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/tpicp-matplotlib-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR"

"$PYTHON_BIN" -m tpicp.pipeline \
  --sam6d-root "$SAM6D_ROOT" \
  --proxy-mesh "$EXAMPLE_DIR/proxy_mesh.ply" \
  --rgb "$SAM6D_ROOT/Data/Example/rgb.png" \
  --depth "$SAM6D_ROOT/Data/Example/depth.png" \
  --camera "$SAM6D_ROOT/Data/Example/camera.json" \
  --output-dir "$OUTPUT_DIR" \
  --cuda-device "$CUDA_DEVICE" \
  "${extra_args[@]}"

printf '\nTP-ICP single-query demo completed.\n'
printf 'Result JSON: %s\n' "$OUTPUT_DIR/final_result.json"
printf 'Pose visualization: %s\n' "$OUTPUT_DIR/05_pem/sam6d_results/vis_pem.png"
