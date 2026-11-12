#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source deployment/env.sh
SAN_PYTHON="${SAN_PYTHON:-/home/pupu/miniconda3/envs/instructpart-san/bin/python}"
SAN_IMAGE="${SAN_IMAGE:-/home/pupu/insPart/external/X-Decoder/inference/images/fruit.jpg}"
mkdir -p logs outputs
"$SAN_PYTHON" deployment/smoke_forward.py \
 --method san --repo "$SAN_ROOT" \
 --weights "$SAN_STORAGE/weights/san_vit_large_14.pth" \
 --image "$SAN_IMAGE" --text watermelon --short-side 512 \
 --output "$SAN_ROOT/outputs/san_forward" 2>&1 | tee logs/san-forward.log
