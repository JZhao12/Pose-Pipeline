#!/bin/bash
set -euo pipefail

OBJ_NAME=${1:-}
IMG_NAME=${2:-"000000"}
if [ -z "$OBJ_NAME" ]; then
    echo "用法: bash ./src/scripts/inference_pipeline.sh <物体名称> [图片名称, 默认000000]" >&2
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}/cnosv2"   # 后面的 python -m src.*、./src/poses/... 都以 cnosv2 为工作目录

export RGB_PATH="${REPO_ROOT}/offline/${OBJ_NAME}/rgb/${IMG_NAME}.png"
export MASK_PATH="${REPO_ROOT}/offline/${OBJ_NAME}/masks/${IMG_NAME}.png"
export OUTPUT_DIR="./tmp/${OBJ_NAME}"

echo "正在推理物体: ${OBJ_NAME}"
echo "使用图片: ${IMG_NAME}.png"

python -m src.scripts.inference_custom \
    --template_dir "$OUTPUT_DIR" \
    --rgb_path "$RGB_PATH" \
    --mask_path "$MASK_PATH" \
    --stability_score_thresh 0.8
