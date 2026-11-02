#!/usr/bin/env bash
# TORCH_CUDA_ARCH_LIST 控制编译目标架构；未设置时按当前 GPU 的 compute capability 自动探测，
# 交叉编译时显式指定，例如 `TORCH_CUDA_ARCH_LIST=8.9 bash build_all_conda.sh`
PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Install mycpp
cd ${PROJ_ROOT}/mycpp/ && \
rm -rf build && mkdir -p build && cd build && \
cmake .. && \
make -j$(nproc)

# Install mycuda
cd ${PROJ_ROOT}/bundlesdf/mycuda && \
rm -rf build *egg* *.so && \
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  TORCH_CUDA_ARCH_LIST=$(python -c "import torch; print('.'.join(map(str, torch.cuda.get_device_capability())))")
  echo "TORCH_CUDA_ARCH_LIST 未设置，按当前 GPU 自动取 ${TORCH_CUDA_ARCH_LIST}"
fi
export TORCH_CUDA_ARCH_LIST
python -m pip install -e . --no-build-isolation

cd ${PROJ_ROOT}
