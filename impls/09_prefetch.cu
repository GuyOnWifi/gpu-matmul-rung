#define LOAD_NEXT_SMEM()                                                       \
  for (int i = 0; i < m_block; i += m_block / pack_per_a) {                    \
    float4 tmp = reinterpret_cast<const float4 *>(                             \
        &A[(trow_a + i) * K + tcol_a * 4])[0];                                 \
    As_next[(tcol_a * 4 + 0) * m_block + trow_a + i] = tmp.x;                  \
    As_next[(tcol_a * 4 + 1) * m_block + trow_a + i] = tmp.y;                  \
    As_next[(tcol_a * 4 + 2) * m_block + trow_a + i] = tmp.z;                  \
    As_next[(tcol_a * 4 + 3) * m_block + trow_a + i] = tmp.w;                  \
  }                                                                            \
                                                                               \
  for (int i = 0; i < k_block; i += k_block / pack_per_b) {                    \
    reinterpret_cast<float4 *>(                                                \
        &Bs_next[(trow_b + i) * n_block + tcol_b * 4])[0] =                    \
        reinterpret_cast<const float4 *>(                                      \
            &B[(trow_b + i) * N + tcol_b * 4])[0];                             \
  }                                                                            \
  A += k_block;                                                                \
  B += k_block * N;

// These are the dimensions of each block.
const int m_block = 64;
const int n_block = 128;
const int k_block = 8;

const int m_thread = 4;
const int n_thread = 4;

const int m_warp = 32;
const int n_warp = 64;

const int WARPSIZE = 32;
const int n_warp_iter = 2;
constexpr int m_warp_iter =
    (m_warp * n_warp) / (WARPSIZE * m_thread * n_thread * n_warp_iter);

const int n_sub_warp = n_warp / n_warp_iter;
const int m_sub_warp = m_warp / m_warp_iter;

const int num_threads = m_block / m_warp * n_block / n_warp * WARPSIZE;
const int pack_per_a = m_block * k_block / 4 / num_threads;
const int pack_per_b = k_block * n_block / 4 / num_threads;

__device__ static void swap(float *&a, float *&b) {
  float *tmp = a;
  a = b;
  b = tmp;
}

__global__ void prefetch(int M, int N, int K, const float *A, const float *B,
                         float *C) {

  __shared__ float As_arr[m_block * k_block];
  __shared__ float Bs_arr[k_block * n_block];

  __shared__ float As_next_arr[m_block * k_block];
  __shared__ float Bs_next_arr[k_block * n_block];

  float *As = As_arr;
  float *Bs = Bs_arr;
  float *As_next = As_next_arr;
  float *Bs_next = Bs_next_arr;

  int trow_a = threadIdx.x / (k_block / 4);
  int tcol_a = threadIdx.x % (k_block / 4);

  int trow_b = threadIdx.x / (n_block / 4);
  int tcol_b = threadIdx.x % (n_block / 4);

  int trow_c = threadIdx.x / (n_block / n_thread);
  int tcol_c = threadIdx.x % (n_block / n_thread);

  int warp_idx = threadIdx.x / WARPSIZE;
  int warp_col = warp_idx % (n_block / n_warp);
  int warp_row = warp_idx / (n_block / n_warp);

  int thread_warp_idx = threadIdx.x % WARPSIZE;
  int tcol_warp = thread_warp_idx % (n_sub_warp / n_thread);
  int trow_warp = thread_warp_idx / (n_sub_warp / n_thread);

  A += blockIdx.x * m_block * K;
  B += blockIdx.y * n_block;
  C += (blockIdx.x * m_block + warp_row * m_warp) * N + blockIdx.y * n_block +
       warp_col * n_warp;

  float acc[m_warp_iter * m_thread * n_warp_iter * n_thread] = {0.0f};
  float cacheM[m_warp_iter * m_thread] = {0.0f};
  float cacheN[n_warp_iter * n_thread] = {0.0f};

  LOAD_NEXT_SMEM()

  swap(As, As_next);
  swap(Bs, Bs_next);

  __syncthreads();

  for (int blk = 0; blk < K; blk += k_block) {
    // Pack into SMEM
    // We'll have to change to account for a derived split work.
    // Transpose the As so reads are contiguous in memory

    if (blk + k_block < K) {
      LOAD_NEXT_SMEM()
    }

    for (int dot_idx = 0; dot_idx < k_block; dot_idx++) {
      for (int warp_idx_m = 0; warp_idx_m < m_warp_iter; warp_idx_m++) {
        for (int idx_m = 0; idx_m < m_thread; idx_m++) {
          cacheM[warp_idx_m * m_thread + idx_m] =
              As[dot_idx * m_block + warp_row * m_warp +
                 warp_idx_m * m_sub_warp + (trow_warp * m_thread + idx_m)];
        }
      }

      for (int warp_idx_n = 0; warp_idx_n < n_warp_iter; warp_idx_n++) {
        for (int idx_n = 0; idx_n < n_thread; idx_n++) {
          cacheN[warp_idx_n * n_thread + idx_n] =
              Bs[dot_idx * n_block + warp_col * n_warp +
                 warp_idx_n * n_sub_warp + (tcol_warp * n_thread) + idx_n];
        }
      }

      // the actual matmul
      for (int warp_sub_row_idx = 0; warp_sub_row_idx < m_warp_iter;
           warp_sub_row_idx++) {
        for (int warp_sub_col_idx = 0; warp_sub_col_idx < n_warp_iter;
             warp_sub_col_idx++) {

          // Our old loop
          for (int res_idx_m = 0; res_idx_m < m_thread; res_idx_m++) {
            for (int res_idx_n = 0; res_idx_n < n_thread; res_idx_n++) {
              acc[(res_idx_m + m_thread * warp_sub_row_idx) *
                      (n_thread * n_warp_iter) +
                  (res_idx_n + warp_sub_col_idx * n_thread)] +=
                  cacheM[res_idx_m + m_thread * warp_sub_row_idx] *
                  cacheN[res_idx_n + n_thread * warp_sub_col_idx];
            }
          }
        }
      }
    }

    swap(As, As_next);
    swap(Bs, Bs_next);

    __syncthreads();
  }

  // Write the accumulator to memory
  for (int warp_sub_row_idx = 0; warp_sub_row_idx < m_warp_iter;
       warp_sub_row_idx++) {
    for (int warp_sub_col_idx = 0; warp_sub_col_idx < n_warp_iter;
         warp_sub_col_idx++) {
      // Wow!
      float *C_tmp =
          C + warp_sub_row_idx * m_sub_warp * N + warp_sub_col_idx * n_sub_warp;
      for (int i = 0; i < m_thread; i++) {
        for (int j = 0; j < n_thread; j += 4) {

          /*
          float4 tmp =
              reinterpret_cast<float4 *>(&C_tmp[(trow_warp * m_thread + i) * N +
                                                tcol_warp * n_thread + j])[0];
          */

          const int idx =
              (warp_sub_row_idx * m_thread + i) * n_warp_iter * n_thread +
              warp_sub_col_idx * n_thread + j;

          float4 tmp = {acc[idx + 0], acc[idx + 1], acc[idx + 2], acc[idx + 3]};

          reinterpret_cast<float4 *>(&C_tmp[(trow_warp * m_thread + i) * N +
                                            tcol_warp * n_thread + j])[0] = tmp;
        }
      }
    }
  }
}

void matmulPrefetch(int M, int N, int K, const float *A, const float *B,
                    float *C) {

  dim3 grid(static_cast<unsigned int>(std::ceil(M / m_block)),
            static_cast<unsigned int>(std::ceil(N / n_block)));
  dim3 block(num_threads);

  prefetch<<<grid, block>>>(M, N, K, A, B, C);
}
