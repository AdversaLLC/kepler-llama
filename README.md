# kepler-llama: run llama.cpp (CUDA) on Kepler GPUs, and on Ampere with the NVIDIA 470 driver

Old NVIDIA cards (Quadro K6000, Tesla K40/K80, GTX 7xx/Titan) were dropped from current CUDA toolkits and
current llama.cpp. They still work on the 470 legacy driver. This repo is the small set of changes plus a
toolchain recipe that gets a **modern llama.cpp CUDA backend running on a Quadro K6000 (sm_35)**, and fixes the
470 driver so **Ampere cards (A2) also create CUDA contexts on Linux 6.15+/7.x**, so one process can shard a
model across old and new cards together.

Tested on: Quadro K6000 (compute 3.5, 12 GB) + 2x NVIDIA A2 (compute 8.6, 15 GB), driver 470.256.02
(AUR `nvidia-470xx-dkms`, kernel 7.2.11), clang 23, llama.cpp `6b9ba4f`. Other Kepler parts (sm_30/sm_37) are untested.

## Results (Q4_K_M, all layers on GPU)

| Model | Cards | Prompt tok/s | Generate tok/s |
|---|---|---|---|
| Mamba-130M | K6000 | 2033 | 131 |
| AI21-Jamba-Reasoning-3B | K6000 | 163 | 16.0 |
| Falcon-Mamba-7B-Instruct | K6000 | 86.8 | 7.5 |
| Falcon-Mamba-7B-Instruct | A2 | 494.9 | 23.8 |
| **AI21-Jamba-Mini-1.7 (51.6B, 29 GiB)** | **2x A2 + K6000** (`-ts 15/15/12`) | 51.5 | 9.9 |

Output on the 3-card run was coherent and correct. On Jamba-3B the greedy text is identical on the K6000 and the CPU.
On Mamba-130M the first generated word differs from the CPU, while perplexity agrees (1.094 vs 1.101); I believe that is a
near-tied first token, but I have not proven it. Generation on the K6000 is far below its memory-bandwidth limit, so the SSM
kernels have room for tuning. Flash attention is off (`GGML_CUDA_FA=OFF`); no tests of other model families were run.

## What is in here

1. `scripts/setup-toolchain.sh`: builds a CUDA root for **clang** (no nvcc, no old gcc): `ptxas` 11.8, libdevice, CUDA 11
   runtime/cuBLAS/cuRAND/CCCL from NVIDIA's PyPI wheels; `fatbinary` from the CUDA 13 wheel behind a small shim (it still
   accepts sm_35 images); a Thrust/CUB shim so clang does not need `-fgpu-rdc`.
2. `patches/llama.cpp/`: three patches on top of llama.cpp `6b9ba4f`:
   - use clang as the CUDA compiler (drop nvcc-only flags `-extended-lambda`, `-use_fast_math`, `-Xcompiler`);
   - guard a bf16 tensor-core path in `gated-delta-net-mma.cu` that does not exist below sm_80;
   - on cc < 5.0 use `cublasSgemmBatched`/`cublasSgemmStridedBatched`: the `*GemmEx` batched calls return
     `CUBLAS_STATUS_ARCH_MISMATCH` on Kepler (this was the only runtime crash).
3. `patches/nvidia-470xx/uvm-mmap-kernel-6.15.patch`: the AUR 470 patch set stubs `uvm_mmap` to `return -ENOTSUPP` on kernels
   >= 6.15 (logging "Unable to call vm_flags_(set|clear) on uvm_mmap"). Kepler does not need that mapping; **Ampere does**, so
   `cuCtxCreate` fails with error 999 on A2/A30/3090-class cards. The driver already carries a safe `nv_vm_flags_set` for
   >= 6.15 (used by `nv-mmap.c`), so the patch simply removes the stub. It relies on the VMA being write-locked inside
   `f_op->mmap`, as the driver's own comment asserts; it worked in testing but is not formally proven. Apply it to
   `/usr/src/nvidia-470.*/nvidia-uvm/uvm.c`, then `dkms build/install` and reload `nvidia_uvm`. This is a kernel module
   change: back up the file and know how to roll back.
4. `scripts/configure.sh`, `scripts/serve.sh`, `scripts/sandbox-run.sh` (a bubblewrap wrapper used for the build).

## Quick start

```
scripts/setup-toolchain.sh ~/kepler-tc                 # needs uv, python3, clang with NVPTX
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout 6b9ba4f
git am /path/to/patches/llama.cpp/*.patch
/path/to/scripts/configure.sh . ~/kepler-tc/cudaroot "35;86"
ninja -C build llama-server llama-bench llama-completion
/path/to/scripts/serve.sh . ~/kepler-tc/cudaroot model.gguf 15,15,12     # GPUS=0,1,2 by default
```

Use `35` alone for a Kepler-only build (faster). Check `nvidia-smi -L` for the device order before choosing `-ts`.

## Limits and honesty

- Credit: the idea of patching llama.cpp for Kepler comes from `fcheung2888/llama.cpp-kepler` (CUDA 10.2 + gcc 8, sm_30); most
  of its C++14 rewrites are not needed with clang.
- This is a personal experiment on one machine. No upstream PR has been made; the patches will need rebasing as llama.cpp moves.
- The 470 branch is end-of-life and has no security updates; this does not change that.

MIT licensed (llama.cpp patches follow llama.cpp's MIT license).
