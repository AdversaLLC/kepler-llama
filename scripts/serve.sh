#!/bin/bash
# Serve a model sharded across GPUs. Usage: serve.sh <llama.cpp dir> <cudaroot> <model.gguf> <split e.g. 15,15,12> [port]
set -euo pipefail
src="$(realpath "$1")"; root="$(realpath "$2")"
export LD_LIBRARY_PATH="$src/build/bin:$root/lib64" CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2}"
exec "$src/build/bin/llama-server" -m "$3" -ngl 99 -ts "$4" --host 127.0.0.1 --port "${5:-8080}" -c 4096
