# CUDA SGEMM on an RTX 3060

Beating `cublasSgemm` at 4096³ FP32 on an RTX 3060 (Ampere, sm_86, 28 SMs), with the GPU clock locked at 2002 MHz. Results are from `nvbench`; correctness is checked against cuBLAS (`rtol=1e-2`, `atol=1e-3`).

**Writeup:** https://easonhuang.dev/blog/gpu_matmul

**Benchmarks:** `benches.cu` · **Tests:** `tests.cu`

## Results

M = N = K = 4096, FP32.

| #      | step                | runtime   | TFLOP/s  | vs cuBLAS  |
| ------ | ------------------- | --------- | -------- | ---------- |
| —      | **cuBLAS**          | **15.81 ms** | **8.69** | **100%** |
| 01     | naive               | 1277 ms   | 0.108    | 1.2%       |
| 02     | gmem coalesced      | 206.35 ms | 0.666    | 7.7%       |
| 03     | smem tiling         | 129.89 ms | 1.06     | 12.2%      |
| 04     | 1d blocktiling      | 45.89 ms  | 3.00     | 34.5%      |
| 05     | 2d blocktiling      | 21.69 ms  | 6.34     | 72.9%      |
| 06     | 2d transposed       | 20.88 ms  | 6.58     | 75.8%      |
| 07     | vectorized float4   | 16.96 ms  | 8.10     | 93.2%      |
| 08     | warptiling          | 15.84 ms  | 8.68     | 99.9%      |
| **09** | **staged**          | **14.23 ms** | **9.66** | **110.8%** |

The 09 row is the tuned staged config: 128×128×16 block, 8×8 thread tile, 64×64 warp tile, `n_warp_iter=1`.

## What each step bought

**01 naive** — one thread per output, no data reuse: 1% of cuBLAS.

**02 gmem coalesced** — warps read consecutive addresses so global loads can be coalesced and grouped into one big transaction.

**03 smem tiling** — a block tile of A and B is staged through shared memory
once and reused by every thread in the block.

**04 1d blocktiling** — each thread computes `TM` outputs down a column,
reusing one A value across `TM` FMAs in registers.

**05 2d blocktiling** — extending it to an extra dimension, with `TM×TN` outputs per thread, so both operands are reused from registers. This allows for greater compute:memory ratios.

**06 2d transposed** — A is transposed into SMEM so the inner-loop reads are
contiguous instead of strided.

**07 vectorized float4** — 128-bit (`float4`) global loads vs 32-bit loads, reducing the
load-instruction count.

**08 warptiling** — warps partition the block tile into warp tiles, and changed indexing math to be more warp-aware.

**09 staged — the win**: instead of prefetching the next SMEM buffer, stage the load. Each thread reads the next k-tile from GMEM into registers, performs the FMAs, then writes that tile into SMEM. Combined with `#pragma unroll`, this lets the compiler interleave loads and FMAs and hides the block-synchronization stalls.

## Misses

The repo keeps the experiments that did not make the final ladder:

- **prefetch smem** (`09_prefetch.cu`) — two SMEM buffers, loading k-tile _i+1_ while computing _i_. Did not improve performance (6.63 TFLOP/s / 71.8% in the earlier run).
- **prefetch to registers** (`10_prefetch_reg.cu`) — another attempt at the same latency problem; also kept as a reference point.
