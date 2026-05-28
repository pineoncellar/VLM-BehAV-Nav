#!/bin/bash

set -e

cd "$(dirname "$0")"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX=2

GEOM_CPUS="${GEOM_CPUS:-0-3}"

DEPTH_TOPIC="${DEPTH_TOPIC:-/gemini330/depth/image_raw}"
DEPTH_INFO_TOPIC="${DEPTH_INFO_TOPIC:-/gemini330/depth/camera_info}"

TARGET_FRAME="${TARGET_FRAME:-tita4264886/base_link}"

# Important:
# Gemini already publishes its internal camera TF chain:
#   camera_link -> camera_depth_frame -> camera_color_frame -> camera_color_optical_frame
#
# Therefore we only publish:
#   tita4264886/base_link -> camera_link
#
# Do NOT publish directly to camera_color_optical_frame.
CAMERA_FRAME="${CAMERA_FRAME:-camera_link}"

CAMERA_X="${CAMERA_X:-0.25}"
CAMERA_Y="${CAMERA_Y:-0.00}"
CAMERA_Z="${CAMERA_Z:-0.25}"

# This transform is base_link -> camera_link.
# Since Gemini handles camera_link -> optical frames internally,
# use identity rotation here first.
# If RViz direction is wrong later, tune this transform, not the optical frame.
CAMERA_QX="${CAMERA_QX:-0.0}"
CAMERA_QY="${CAMERA_QY:-0.0}"
CAMERA_QZ="${CAMERA_QZ:-0.0}"
CAMERA_QW="${CAMERA_QW:-1.0}"

TF_PID=""
GEOM_PID=""

run_on_cpu() {
  CPUS="$1"
  shift
  if command -v taskset >/dev/null 2>&1; then
    taskset -c "$CPUS" "$@"
  else
    "$@"
  fi
}

cleanup() {
  echo
  echo "Stopping geometry-only mapping..."
  kill "${TF_PID:-}" "${GEOM_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Clean our old mapping-related processes.
# These two static TF kill lines intentionally remove both the new correct TF
# and the old incorrect direct-to-optical TF if it is still running.
pkill -f "static_transform_publisher.*camera_link" 2>/dev/null || true
pkill -f "static_transform_publisher.*camera_color_optical_frame" 2>/dev/null || true
pkill -f "depth_grid_geometry_node.py" 2>/dev/null || true
pkill -f "depth_grid_semantic_node.py" 2>/dev/null || true
pkill -f "clipseg_debug_node.py" 2>/dev/null || true

start_tf() {
  ros2 run tf2_ros static_transform_publisher \
    --x "$CAMERA_X" \
    --y "$CAMERA_Y" \
    --z "$CAMERA_Z" \
    --qx "$CAMERA_QX" \
    --qy "$CAMERA_QY" \
    --qz "$CAMERA_QZ" \
    --qw "$CAMERA_QW" \
    --frame-id "$TARGET_FRAME" \
    --child-frame-id "$CAMERA_FRAME" &

  TF_PID=$!
  echo "TF started: PID=$TF_PID, parent=$TARGET_FRAME, child=$CAMERA_FRAME, xyz=($CAMERA_X,$CAMERA_Y,$CAMERA_Z), q=($CAMERA_QX,$CAMERA_QY,$CAMERA_QZ,$CAMERA_QW)"
}

start_grid() {
  run_on_cpu "$GEOM_CPUS" python3 -u depth_grid_semantic_node.py \
    --no-sim-time \
    --depth-topic "$DEPTH_TOPIC" \
    --camera-info-topic "$DEPTH_INFO_TOPIC" \
    --target-frame "$TARGET_FRAME" \
    --grid-topic /local_traversability_grid \
    --x-min 0.0 \
    --x-max 5.0 \
    --y-min -2.5 \
    --y-max 2.5 \
    --resolution 0.2 \
    --min-depth 0.2 \
    --max-depth 6.0 \
    --ground-z 0.0 \
    --obstacle-height 0.25 \
    --max-obstacle-height 1.5 \
    --downsample 2 \
    --auto-downsample \
    --min-downsample 2 \
    --max-downsample 5 \
    --grid-target-load 0.65 \
    --min-points-per-cell 3 \
    --min-obstacle-points-per-cell 3 \
    --min-valid-points 1000 \
    --min-in-area-points 200 \
    --disable-semantic-fusion \
    --image-qos-depth 1 \
    --pub-qos-depth 1 \
    --executor-threads 3 \
    --heartbeat-sec 2.0 \
    --print-every 30 &

  GEOM_PID=$!
  echo "Geometry grid started: PID=$GEOM_PID, CPUs=$GEOM_CPUS, depth_topic=$DEPTH_TOPIC, camera_info=$DEPTH_INFO_TOPIC"
}

start_tf
sleep 1
start_grid

echo
echo "Geometry-only mode started."
echo "CLIPSeg is NOT started."
echo "Semantic fusion is disabled."
echo
echo "Topics:"
echo "  /local_traversability_grid"
echo
echo "Input topics:"
echo "  Depth : $DEPTH_TOPIC"
echo "  Info  : $DEPTH_INFO_TOPIC"
echo
echo "TF:"
echo "  Parent frame: $TARGET_FRAME"
echo "  Child frame : $CAMERA_FRAME"
echo "  Expected full chain:"
echo "    $TARGET_FRAME -> camera_link -> camera_depth_frame -> camera_color_frame -> camera_color_optical_frame"
echo

while true; do
  if ! kill -0 "$TF_PID" 2>/dev/null; then
    echo "WARN: TF process exited. Restarting..."
    wait "$TF_PID" 2>/dev/null || true
    start_tf
  fi

  if ! kill -0 "$GEOM_PID" 2>/dev/null; then
    echo "WARN: geometry grid process exited. Restarting..."
    wait "$GEOM_PID" 2>/dev/null || true
    start_grid
  fi

  sleep 1
done
