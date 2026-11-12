#!/usr/bin/env bash
# Activate a fresh Python 3.10 environment before invoking this script.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CUDA_HOME:?Set CUDA_HOME to the CUDA 11.8 toolkit directory}"
: "${SAN_STORAGE:?Set SAN_STORAGE to the model storage directory}"
export PATH="$CUDA_HOME/bin:$PATH"
unset PYTHONPATH
python -m pip install 'pip<25' 'setuptools<70' wheel ninja -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install "$SAN_STORAGE/wheels/torch-2.1.0+cu118-cp310-cp310-linux_x86_64.whl" "$SAN_STORAGE/wheels/torchvision-0.16.0+cu118-cp310-cp310-linux_x86_64.whl" -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -r deployment/requirements.lock.txt -c deployment/constraints.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# Build on the destination GPU; never copy the source machine's compiled extension.
MAX_JOBS=2 python -m pip install --no-build-isolation --no-deps 'git+https://github.com/facebookresearch/detectron2.git@d1e04565d3bec8719335b88be9e9b961bf3ec464' -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip check
python deployment/probe_gpu.py
