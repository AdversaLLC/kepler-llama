#!/bin/bash
# Build a CUDA "root" that clang can use to compile for Kepler (sm_35) without nvcc:
#   - ptxas 11.8, libdevice, CUDA 11 runtime/cublas/curand/CCCL headers+libs from NVIDIA's PyPI wheels
#   - fatbinary from the CUDA 13 wheel (it still accepts sm_35 images), behind a small argument-translating shim
# Usage: setup-toolchain.sh <dir>     (needs: uv, python3, clang >= 17 with the NVPTX backend)
set -euo pipefail
dir="$(realpath -m "${1:?usage: setup-toolchain.sh <dir>}")"
mkdir -p "$dir"
cd "$dir"
uv venv -q --python 3.12 .venv
uv pip install -q -p .venv nvidia-cuda-nvcc-cu11 nvidia-cuda-runtime-cu11 nvidia-curand-cu11 \
  nvidia-cublas-cu11 nvidia-cuda-cccl-cu11
mkdir -p cu13
uv pip install -q --target cu13 nvidia-cuda-nvcc

nv="$dir/.venv/lib/python3.12/site-packages/nvidia"
fatbinary_real="$(find "$dir/cu13" -type f -name fatbinary | head -1)"
[ -n "$fatbinary_real" ] || { echo "fatbinary not found in the CUDA 13 wheel" >&2; exit 1; }

root="$dir/cudaroot"
rm -rf "$root"
mkdir -p "$root/bin" "$root/lib64" "$root/nvvm" "$root/include" "$root/shim/thrust/system/cuda"
ln -s "$nv/cuda_nvcc/bin/ptxas" "$root/bin/ptxas"
ln -s "$nv/cuda_nvcc/nvvm/libdevice" "$root/nvvm/libdevice"
echo "CUDA Version 11.8.0" > "$root/version.txt"

for d in cuda_runtime cuda_nvcc curand cublas cuda_cccl; do
  for f in "$nv/$d/include"/*; do ln -sf "$f" "$root/include/$(basename "$f")"; done
done
ln -sf "$nv/cuda_runtime/lib/libcudart.so.11.0" "$root/lib64/libcudart.so"
ln -sf "$nv/cuda_runtime/lib/libcudart.so.11.0" "$root/lib64/libcudart.so.11.0"
for l in libcublas libcublasLt; do
  ln -sf "$nv/cublas/lib/$l.so.11" "$root/lib64/$l.so"
  ln -sf "$nv/cublas/lib/$l.so.11" "$root/lib64/$l.so.11"
done
ar rc "$root/lib64/libcudadevrt.a"   # empty: no dynamic parallelism is used

# CMake only needs *an* nvcc to locate the toolkit root; real compiling is done by clang.
printf '#!/bin/sh\necho "Cuda compilation tools, release 11.8, V11.8.89"\n' > "$root/bin/nvcc"
chmod +x "$root/bin/nvcc"

# clang calls `fatbinary --create X --image=profile=sm_35,file=F`; CUDA 13 wants --image3=kind=elf,sm=35,file=F
cat > "$root/bin/fatbinary" <<SHIM
#!/usr/bin/env python3
import subprocess, sys
REAL = "$fatbinary_real"
args, out, i = sys.argv[1:], [], 0
while i < len(args):
    a = args[i]
    if a in ("--create", "-create"):
        out.append("--create=" + args[i + 1]); i += 2; continue
    if a.startswith(("--image=", "-image=")):
        spec = dict(kv.split("=", 1) for kv in a.split("=", 1)[1].split(","))
        arch = spec["profile"].split("_")[1]
        kind = "ptx" if spec["profile"].startswith("compute") else "elf"
        out.append("--image3=kind=%s,sm=%s,file=%s" % (kind, arch, spec["file"]))
    else:
        out.append(a)
    i += 1
sys.exit(subprocess.call([REAL] + out))
SHIM
chmod +x "$root/bin/fatbinary"

# Thrust/CUB use device-side kernel launches (dynamic parallelism) that clang only allows with -fgpu-rdc.
# Shadow thrust's config.h so __THRUST_HAS_CUDART__ is 0 unless RDC is on.
python3 - "$root" <<'PY'
import sys
root = sys.argv[1]
src = open(root + "/include/thrust/system/cuda/config.h").read()
old = "#  if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__>= 350 && defined(__CUDACC_RDC__))"
assert old in src, "thrust config.h changed; update this shim"
open(root + "/shim/thrust/system/cuda/config.h", "w").write(
    src.replace(old, "#  if defined(__CUDACC_RDC__)  /* shim: no device-side launches under clang */"))
PY
echo "cudaroot ready: $root"
