#pragma once
#include <cuda_runtime.h>

// 08_warptiling.cu, templated on its config so it can be swept.
// Body is a faithful translation -- same indexing, same loop structure, same
// write-only epilogue. Only the constants became template parameters.
//
//   BM,BN,BK   block tile
//   TM,TN      thread tile
//   WM,WN      warp tile
//   WNITER     warp sub-tiles along N (MWITER is derived from it)

template <int BM, int BN, int BK, int TM, int TN, int WM, int WN, int WNITER>
__global__ void WarptilingTpl(int M, int N, int K, const float *A,
                              const float *B, float *C) {
  constexpr int WARPSIZE = 32;
  constexpr int MWITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr int NSUB = WN / WNITER;
  constexpr int MSUB = WM / MWITER;
  constexpr int NUM_THREADS = BM / WM * BN / WN * WARPSIZE;
  constexpr int PACK_A = BM * BK / 4 / NUM_THREADS;
  constexpr int PACK_B = BK * BN / 4 / NUM_THREADS;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  int trow_a = threadIdx.x / (BK / 4);
  int tcol_a = threadIdx.x % (BK / 4);

  int trow_b = threadIdx.x / (BN / 4);
  int tcol_b = threadIdx.x % (BN / 4);

  int warp_idx = threadIdx.x / WARPSIZE;
  int warp_col = warp_idx % (BN / WN);
  int warp_row = warp_idx / (BN / WN);

  int thread_warp_idx = threadIdx.x % WARPSIZE;
  int tcol_warp = thread_warp_idx % (NSUB / TN);
  int trow_warp = thread_warp_idx / (NSUB / TN);

  A += blockIdx.x * BM * K;
  B += blockIdx.y * BN;
  C += (blockIdx.x * BM + warp_row * WM) * N + blockIdx.y * BN + warp_col * WN;

  float acc[MWITER * TM * WNITER * TN] = {0.0f};
  float cacheM[MWITER * TM] = {0.0f};
  float cacheN[WNITER * TN] = {0.0f};

  for (int blk = 0; blk < K; blk += BK) {
    // Transpose A on the way into SMEM so the inner loop reads it contiguously.
    for (int i = 0; i < BM; i += BM / PACK_A) {
      float4 tmp =
          reinterpret_cast<const float4 *>(&A[(trow_a + i) * K + tcol_a * 4])[0];
      As[(tcol_a * 4 + 0) * BM + trow_a + i] = tmp.x;
      As[(tcol_a * 4 + 1) * BM + trow_a + i] = tmp.y;
      As[(tcol_a * 4 + 2) * BM + trow_a + i] = tmp.z;
      As[(tcol_a * 4 + 3) * BM + trow_a + i] = tmp.w;
    }

    for (int i = 0; i < BK; i += BK / PACK_B) {
      reinterpret_cast<float4 *>(&Bs[(trow_b + i) * BN + tcol_b * 4])[0] =
          reinterpret_cast<const float4 *>(&B[(trow_b + i) * N + tcol_b * 4])[0];
    }

    __syncthreads();

    for (int dot_idx = 0; dot_idx < BK; dot_idx++) {
      for (int warp_idx_m = 0; warp_idx_m < MWITER; warp_idx_m++) {
        for (int idx_m = 0; idx_m < TM; idx_m++) {
          cacheM[warp_idx_m * TM + idx_m] =
              As[dot_idx * BM + warp_row * WM + warp_idx_m * MSUB +
                 (trow_warp * TM + idx_m)];
        }
      }

      for (int warp_idx_n = 0; warp_idx_n < WNITER; warp_idx_n++) {
        for (int idx_n = 0; idx_n < TN; idx_n++) {
          cacheN[warp_idx_n * TN + idx_n] =
              Bs[dot_idx * BN + warp_col * WN + warp_idx_n * NSUB +
                 (tcol_warp * TN) + idx_n];
        }
      }

      for (int wsr = 0; wsr < MWITER; wsr++) {
        for (int wsc = 0; wsc < WNITER; wsc++) {
          for (int res_idx_m = 0; res_idx_m < TM; res_idx_m++) {
            for (int res_idx_n = 0; res_idx_n < TN; res_idx_n++) {
              acc[(res_idx_m + TM * wsr) * (TN * WNITER) +
                  (res_idx_n + wsc * TN)] +=
                  cacheM[res_idx_m + TM * wsr] * cacheN[res_idx_n + TN * wsc];
            }
          }
        }
      }
    }

    A += BK;
    B += BK * N;

    __syncthreads();
  }

  // Write-only epilogue -- C is never read.
  for (int wsr = 0; wsr < MWITER; wsr++) {
    for (int wsc = 0; wsc < WNITER; wsc++) {
      float *C_tmp = C + wsr * MSUB * N + wsc * NSUB;
      for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j += 4) {
          const int idx = (wsr * TM + i) * WNITER * TN + wsc * TN + j;
          float4 tmp = {acc[idx + 0], acc[idx + 1], acc[idx + 2], acc[idx + 3]};
          reinterpret_cast<float4 *>(
              &C_tmp[(trow_warp * TM + i) * N + tcol_warp * TN + j])[0] = tmp;
        }
      }
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN, int WM, int WN, int WNITER>
void launchWarptileTpl(int M, int N, int K, const float *A, const float *B,
                       float *C) {
  constexpr int NUM_THREADS = BM / WM * BN / WN * 32;
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
  dim3 block(NUM_THREADS);
  WarptilingTpl<BM, BN, BK, TM, TN, WM, WN, WNITER>
      <<<grid, block>>>(M, N, K, A, B, C);
}
