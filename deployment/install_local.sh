#!/usr/bin/env bash
# Local installation coordinator; run with Bash, not inside base Python.
set -euo pipefail
cd /home/pupu/insPart/SAN
source deployment/env.sh
SAN_PYTHON=/home/pupu/miniconda3/envs/instructpart-san/bin/python
export PATH="/home/pupu/miniconda3/envs/instructpart-san/bin:$PATH"
stage() { printf '%s %s\n' "$(date -Iseconds)" "$1" >> logs/progress.log; }
WHEEL="$SAN_STORAGE/wheels/torch-2.1.0+cu118-cp310-cp310-linux_x86_64.whl"
stage '等待 PyTorch wheel 分段合并完成'
while [ "$(stat -c %s "$WHEEL" 2>/dev/null || echo 0)" != 2325888121 ] || [ -d "$WHEEL.parts" ]; do sleep 10; done
stage '验证 PyTorch wheel SHA256'
printf '%s  %s\n' a81b554184492005543ddc32e96469f9369d778dedd195d73bda9bed407d6589 "$WHEEL" | sha256sum -c -
stage '安装 PyTorch 与 torchvision（读取移动硬盘本地 wheel）'
"$SAN_PYTHON" -m pip install "$WHEEL" "$SAN_STORAGE/wheels/torchvision-0.16.0+cu118-cp310-cp310-linux_x86_64.whl" -c deployment/constraints.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
stage 'GPU 基础计算验证'
"$SAN_PYTHON" deployment/probe_gpu.py | tee logs/gpu-probe.json
stage '编译 Detectron2 v0.6 CUDA 扩展，MAX_JOBS=2'
MAX_JOBS=2 "$SAN_PYTHON" -m pip install --no-build-isolation /tmp/detectron2-0.6 -c deployment/constraints.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
"$SAN_PYTHON" -m pip check
"$SAN_PYTHON" -c 'import torch, detectron2, open_clip, timm; from predict import Predictor; print("SAN imports OK")'
stage '框架、Detectron2 编译和依赖导入检查通过；等待模型权重校验与真实前向'
