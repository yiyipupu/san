# Source this file before running SAN.
export SAN_ROOT="${SAN_ROOT:-/home/pupu/insPart/SAN}"
export SAN_STORAGE="${SAN_STORAGE:-/media/pupu/PUPU/insPart/SAN}"
export SAN_MOUNT="${SAN_MOUNT:-/media/pupu/PUPU}"
if ! mountpoint -q "$SAN_MOUNT"; then
  echo "SAN disk is not mounted" >&2
  return 1
fi
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export HF_HOME="$SAN_STORAGE/cache/huggingface"
export TORCH_HOME="$SAN_STORAGE/cache/torch"
export XDG_CACHE_HOME="$SAN_STORAGE/cache/xdg"
export SAN_CLIP_CACHE="$SAN_STORAGE/cache/clip"
unset PYTHONPATH
