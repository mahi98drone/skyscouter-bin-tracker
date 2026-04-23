#!/usr/bin/env bash
# =============================================================================
# Skyscouter – CV Engineer Assessment
# Entry point.
#
# Usage:
#   bash run.sh --video input.mp4 --calib calib.json
#   bash run.sh --video input.mp4 --calib calib.json --kalman
#   bash run.sh --video input.mp4 --calib calib.json --kalman --gpu
# =============================================================================
set -e

# ── Parse arguments ──────────────────────────────────────────────────────────
VIDEO=""
CALIB=""
WAYPOINTS="waypoints.json"
GPU_FLAG=""
KALMAN_FLAG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --video)     VIDEO="$2";             shift 2 ;;
    --calib)     CALIB="$2";             shift 2 ;;
    --waypoints) WAYPOINTS="$2";         shift 2 ;;
    --gpu)       GPU_FLAG="--gpu";       shift   ;;
    --kalman)    KALMAN_FLAG="--kalman"; shift   ;;
    *)           echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$VIDEO" || -z "$CALIB" ]]; then
  echo "Usage: bash run.sh --video <path> --calib <path> [--waypoints <path>] [--gpu] [--kalman]"
  exit 1
fi

# ── Dependencies ──────────────────────────────────────────────────────────────
echo "[run.sh] Installing Python dependencies..."
pip install -q -r requirements.txt

# Pre-download YOLOv8n weights (avoids a surprise mid-run download)
echo "[run.sh] Checking YOLOv8n weights..."
python - <<'EOF'
from ultralytics import YOLO
YOLO("yolov8n.pt")
print("[run.sh] Weights ready.")
EOF

# GPU sanity-check
if [[ -n "$GPU_FLAG" ]]; then
  echo "[run.sh] Verifying CUDA..."
  python -c "
import torch
if not torch.cuda.is_available():
    raise RuntimeError('--gpu passed but CUDA not found. Run without --gpu for CPU mode.')
print('[run.sh] CUDA OK:', torch.cuda.get_device_name(0))
"
fi

mkdir -p results

# ── Run ───────────────────────────────────────────────────────────────────────
echo ""
echo "[run.sh] Starting tracker..."
echo "[run.sh] Video     : $VIDEO"
echo "[run.sh] Calib     : $CALIB"
echo "[run.sh] Waypoints : $WAYPOINTS"
[[ -n "$GPU_FLAG"    ]] && echo "[run.sh] Inference : GPU" || echo "[run.sh] Inference : CPU"
[[ -n "$KALMAN_FLAG" ]] && echo "[run.sh] Kalman    : enabled" || echo "[run.sh] Kalman    : disabled"
echo ""

python track_bin.py \
  --video     "$VIDEO"           \
  --calib     "$CALIB"           \
  --waypoints "$WAYPOINTS"       \
  --output    results/output.csv \
  $GPU_FLAG                      \
  $KALMAN_FLAG
