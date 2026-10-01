# CUDA SGEMM on an RTX 3060

Beating `cublasSgemm` at 4096³ FP32 on a RTX 3060 (Ampere, sm_86, 28 SMs) at a locked clock of
2000 MHz. Checked against cuBLAS.

| #      | step                | TFLOP/s   | vs cuBLAS  |
| ------ | ------------------- | --------- | ---------- |
| —      | **cuBLAS**          | **9.23**  | **100%**   |
| 01     | naive               | 0.11      | 1.2%       |
| 02     | gmem coalesced      | 0.67      | 7.3%       |
| 03     | smem tiling         | 1.07      | 11.6%      |
| 04     | 1d blocktiling      | 3.02      | 32.7%      |
| 05     | 2d blocktiling      | 6.41      | 69.4%      |
| 06     | 2d transposed       | 6.65      | 72.1%      |
| 07     | vectorized float4   | 8.47      | 91.7%      |
| 08     | warptiling          | 9.06      | 98.1%      |
| 09     | prefetch smem       | 6.63      | 71.8%      |
| 11     | staged (default)    | 8.28      | 89.7%      |
| **11** | **staged + unroll** | **10.34** | **112.1%** |

## What each step bought

**01 naive** — one thread per output, no data reuse: 1% of cuBLAS.

**02 gmem coalesced** — warps read consecutive addresses so global loads can be coalesced and grouped into one big transaction

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

**09 prefetch smem** _(miss)_ — two SMEM buffers, loading k-tile _i+1_ while
computing _i_. Intention was software pipelining and latency hiding, did not improve performance

**11 staged + unroll — the win**: Instead of prefetching the next SMEM buffer, stage the load. Each thread reads from GMEM and writes to SMEM. Instead, have each thread read the next k-tile into registers, perform FMAs, and then write it into SMEM. Allows for more latency hiding and less stalls due to block synchronization. Force `#pragma unroll` to allow the compiler to interleave loads and FMAs.
