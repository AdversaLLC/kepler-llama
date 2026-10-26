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
| **AI21-Jamba-Mini-1.7 (51.6B, 29 GiB)** | **2x A2 + K6000** (`-ts 15/15/12`, memory-proportional) | 51.5 | 9.9 |
| **AI21-Jamba-Mini-1.7, tuned split** | **2x A2 + K6000** (`-ts 20/20/8`) | **93.3** | **12.8** |
| AI21-Jamba-Mini-1.7, tuned split + Q4_K kernel (patch 4) | 2x A2 + K6000 (`-ts 20/20/8`; 21/21/6 gives 14.5) | 93.7 | **14.1** |
| Mamba-Codestral-7B (abliterated), Q4_K_M | K6000, before -> after patch 4 | 73 | 7.6 -> **12.9** |
| Falcon-Mamba-7B-Instruct, Q4_K_M | K6000, before -> after patch 4 | 87 | 7.5 -> **12.8** |
| AI21-Jamba-Reasoning-3B, Q4_K_M | K6000, before -> after patch 4 | 163 -> 203 | 16.0 -> **22.1** |

Output on the 3-card run was coherent and correct. On Jamba-3B the greedy text is identical on the K6000 and the CPU.
On Mamba-130M the first generated word differs from the CPU, while perplexity agrees (1.094 vs 1.101); I believe that is a
near-tied first token, but I have not proven it. Generation on the K6000 is far below its memory-bandwidth limit, so the SSM
kernels have room for tuning. Flash attention is off (`GGML_CUDA_FA=OFF`); no tests of other model families were run.

## Why the K6000 was slow, and the Q4_K kernel (patch 4)
`test-backend-ops perf` on the K6000 showed a Q4_K matrix-vector product (token generation) moving only **35 GB/s** against 217 GB/s for the same
shape in f32 and the card's 288 GB/s peak; the Mamba scan/convolution kernels were negligible (7-87 us). Kepler has no `dp4a` and runs 32-bit
integer shifts/multiplies at about 1/6 rate, so llama.cpp's "quantize activations to int8, then dp4a" path is compute-bound there.
Patch 4 adds `mmvq-kepler.cu`: for single-token decode (dense `MUL_MAT` and `MUL_MAT_ID` with one token) on compute capability < 6.1 it keeps the
activations in float and converts each 4-bit weight with the magic-number trick (OR the nibble into the mantissa of 2^23, subtract 2^23: full-rate
logic and FP32 only). A lane owns a 16-byte chunk (32 weights) so the 6-bit scale decode is amortised; two rows per warp share each activation
load. Micro-benchmark (`MUL_MAT q4_K m=4096 k=14336 n=1`, K6000): 946 us -> 489 us. Three layouts were tried: the first two reached only
851-868 us (instruction-issue bound), the final one 489 us; one row per warp gave 675 us and four gave 525 us.

How it was checked: `test-backend-ops test -o MUL_MAT -p "type_a=q4_K,type_b=f32,m=[0-9]*,n=1,"` and `-o MUL_MAT_ID -p "type_a=q4_K,type_b=f32"` (86 cases)
pass against the CPU reference; perplexity on wikitext-2 (Jamba-3B, 8 chunks) is 9.863 on the K6000 with the kernel, 9.864 on an A2 and 9.853
on the CPU (all within noise). Greedy text can differ in wording from older runs because the activations are no longer rounded to int8.

Limits: only **Q4_K**, only **decode** (one token), unfused ops (fusion is already off for cc <= Pascal); Q6_K and the other quantizations, and
prompt processing, still use the slow path (prompt speed is unchanged). Maxwell (cc 5.x) should benefit but is untested. Running the full
`test-backend-ops` suite on Kepler aborts on a bf16 case (cuBLAS cannot do bf16 there), so filter by type as above.

## Tuning the split across unequal cards
Splitting layers in proportion to VRAM (`-ts 15/15/12`) is not the fastest: the slow card then sets the pace. Same model, `llama-bench -p 128 -n 32` (measured before patch 4; with it every row gains 5-12%, e.g. 15/15/12 -> 12.2, 20/20/8 -> 14.1, 21/21/6 -> 14.5):

| `-ts` (A2 / A2 / K6000) | prompt tok/s | generate tok/s |
|---|---|---|
| 15/15/12 (by memory) | 71.0 | 10.0 |
| 18/18/12 | 72.5 | 10.7 |
| 20/20/10 | 82.3 | 11.8 |
| **20/20/8** | **93.3** | **12.8** |
| 21/21/6 | 95.0 | 12.9 (22/22/6 does not fit) |
| 22/22/8 | 85.0 | 11.3 (noisy) |

Less on the K6000 is better until the A2s run out of memory. Greedy output was identical to the 15/15/12 run. Failure mode to know: when a card's
VRAM is exhausted the error is `CUBLAS_STATUS_NOT_INITIALIZED`, not "out of memory". Note `llama-bench` takes `-ts 20/20/8` (slashes) while
`llama-cli`/`llama-server` take `-ts 20,20,8` (commas); a comma in `llama-bench` is silently misread and puts everything on GPU 0.

## What is in here

1. `scripts/setup-toolchain.sh`: builds a CUDA root for **clang** (no nvcc, no old gcc): `ptxas` 11.8, libdevice, CUDA 11
   runtime/cuBLAS/cuRAND/CCCL from NVIDIA's PyPI wheels; `fatbinary` from the CUDA 13 wheel behind a small shim (it still
   accepts sm_35 images); a Thrust/CUB shim so clang does not need `-fgpu-rdc`.
2. `patches/llama.cpp/`: four patches on top of llama.cpp `6b9ba4f`:
   - use clang as the CUDA compiler (drop nvcc-only flags `-extended-lambda`, `-use_fast_math`, `-Xcompiler`);
   - guard a bf16 tensor-core path in `gated-delta-net-mma.cu` that does not exist below sm_80;
   - on cc < 5.0 use `cublasSgemmBatched`/`cublasSgemmStridedBatched`: the `*GemmEx` batched calls return
     `CUBLAS_STATUS_ARCH_MISMATCH` on Kepler (this was the only runtime crash);
   - **a Q4_K matrix-vector kernel for GPUs without `dp4a`** (see the next section): +70% token generation on the K6000.
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
/path/to/scripts/serve.sh . ~/kepler-tc/cudaroot model.gguf 20,20,8     # GPUS=0,1,2 by default
```

Use `35` alone for a Kepler-only build (faster). Check `nvidia-smi -L` for the device order before choosing `-ts`.

## Limits and honesty

- Credit: the idea of patching llama.cpp for Kepler comes from `fcheung2888/llama.cpp-kepler` (CUDA 10.2 + gcc 8, sm_30); most
  of its C++14 rewrites are not needed with clang.
- This is a personal experiment on one machine. No upstream PR has been made; the patches will need rebasing as llama.cpp moves.
- The 470 branch is end-of-life and has no security updates; this does not change that.

MIT licensed (llama.cpp patches follow llama.cpp's MIT license).
