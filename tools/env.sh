#!/usr/bin/env bash
# Usage: source tools/env.sh

export ML22_ROOT="/mnt/d/ML22"
export ML22_CACHE="$ML22_ROOT/.cache"

mkdir -p "$ML22_CACHE"/{pip,torch,matplotlib,huggingface,kaggle}

# Common caches -> D drive
export PIP_CACHE_DIR="$ML22_CACHE/pip"
export TORCH_HOME="$ML22_CACHE/torch"
export MPLCONFIGDIR="$ML22_CACHE/matplotlib"
export HF_HOME="$ML22_CACHE/huggingface"
export TRANSFORMERS_CACHE="$ML22_CACHE/huggingface"
export KAGGLE_CONFIG_DIR="$ML22_CACHE/kaggle"

# (Optional) follow XDG cache convention
export XDG_CACHE_HOME="$ML22_CACHE"

echo "[ml2022] cache redirected to: $ML22_CACHE"
