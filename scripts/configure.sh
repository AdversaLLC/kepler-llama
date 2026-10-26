#!/bin/bash
# Configure a patched llama.cpp checkout for CUDA with clang. Usage: configure.sh <llama.cpp dir> <cudaroot> [archs]
set -euo pipefail
src="$(realpath "${1:?llama.cpp dir}")"; root="$(realpath "${2:?cudaroot}")"; archs="${3:-35;86}"
export CUDA_PATH="$root"
cmake -S "$src" -B "$src/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/bin/clang++ \
  -DCMAKE_CUDA_FLAGS="--cuda-path=$root -DCUB_DISABLE_CDP -DTHRUST_DISABLE_CDP -I$root/shim" \
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared -DCMAKE_CUDA_ARCHITECTURES="$archs" -DGGML_CUDA_FA=OFF \
  -DCUDAToolkit_ROOT="$root" -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
echo "now: ninja -C $src/build llama-server llama-bench llama-completion"
