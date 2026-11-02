#!/bin/bash
set -euo pipefail

OBJ_NAME=${1:-}
if [ -z "$OBJ_NAME" ]; then
    echo "用法: bash ./src/scripts/render_pipeline.sh <物体名称>" >&2
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}/cnosv2"   # 后面的 python -m src.*、./src/poses/... 都以 cnosv2 为工作目录

export CAD_PATH="${REPO_ROOT}/offline/${OBJ_NAME}/mesh/mesh.obj"
export OUTPUT_DIR="./tmp/${OBJ_NAME}"
export LIGHTING_ITENSITY=1.0
export RADIUS=0.4

echo "正在处理物体: ${OBJ_NAME}"
echo "模型路径: ${CAD_PATH}"
echo "输出目录: ${OUTPUT_DIR}"

python -m src.poses.pyrender \
    "${CAD_PATH}" \
    ./src/poses/predefined_poses/obj_poses_level0.npy \
    "${OUTPUT_DIR}" \
    0 False $LIGHTING_ITENSITY $RADIUS
