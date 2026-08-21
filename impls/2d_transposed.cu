// These are the dimensions of each block.
const int m_block = 128;
const int n_block = 128;
const int k_block = 8;

const int m_thread = 8;
const int n_thread = 8;

const int num_threads = m_block * n_block / m_thread / n_thread;
const int pack_per_a = m_block * k_block / num_threads;
const int pack_per_b = k_block * n_block / num_threads;

__global__ void BlockTiling2DTransposed(int M, int N, int K, const float *A,
                                        const float *B, float *C) {

  __shared__ float As[m_block * k_block];
  __shared__ float Bs[k_block * n_block];

  int trow_a = threadIdx.x / k_block;
  int tcol_a = threadIdx.x % k_block;

  int trow_b = threadIdx.x / n_block;
  int tcol_b = threadIdx.x % n_block;

  int trow_c = threadIdx.x / (n_block / n_thread);
  int tcol_c = threadIdx.x % (n_block / n_thread);

  A += blockIdx.x * m_block * K;
  B += blockIdx.y * n_block;
  C += blockIdx.x * m_block * N + blockIdx.y * n_block;

  float acc[m_thread * n_thread] = {0.0f};
  float cacheM[m_thread] = {0.0f};
  float cacheN[n_thread] = {0.0f};

  for (int blk = 0; blk < K; blk += k_block) {
    // Pack into SMEM
    // We'll have to change to account for a derived split work.
    // Transpose the As so reads are contiguous in memory
    for (int i = 0; i < m_block; i += m_block / pack_per_a) {
      As[tcol_a * m_block + (trow_a + i)] = A[(trow_a + i) * K + tcol_a];
    }

    for (int i = 0; i < k_block; i += k_block / pack_per_b) {
      Bs[(trow_b + i) * n_block + tcol_b] = B[(trow_b + i) * N + tcol_b];
    }

    __syncthreads();

    A += k_block;
    B += k_block * N;

    /*
    for (int dot_idx = 0; dot_idx < k_block; dot_idx++) {
      for (int res_idx_m = 0; res_idx_m < m_thread; res_idx_m++) {
        for (int res_idx_n = 0; res_idx_n < n_thread; res_idx_n++) {
          acc[res_idx_m * n_thread + res_idx_n] +=
              As[(trow_c * m_thread + res_idx_m) * k_block + dot_idx] *
              Bs[dot_idx * n_block + (tcol_c * n_thread + res_idx_n)];
        }
      }
    }
    */

    for (int dot_idx = 0; dot_idx < k_block; dot_idx++) {
      for (int idx_m = 0; idx_m < m_thread; idx_m++) {
        cacheM[idx_m] = As[dot_idx * m_block + (trow_c * m_thread + idx_m)];
      }

      for (int idx_n = 0; idx_n < n_thread; idx_n++) {
        cacheN[idx_n] = Bs[dot_idx * n_block + (tcol_c * n_thread) + idx_n];
      }

      for (int res_idx_m = 0; res_idx_m < m_thread; res_idx_m++) {
        for (int res_idx_n = 0; res_idx_n < n_thread; res_idx_n++) {
          acc[res_idx_m * n_thread + res_idx_n] +=
              cacheM[res_idx_m] * cacheN[res_idx_n];
        }
      }
    }

    __syncthreads();
  }

  // Write the accumulator to memory
  for (int i = 0; i < m_thread; i++) {
    for (int j = 0; j < n_thread; j++) {
      C[(trow_c * m_thread + i) * N + tcol_c * n_thread + j] =
          acc[i * n_thread + j];
    }
  }
}

void matmulBlockTiling2DTransposed(int M, int N, int K, const float *A,
                                   const float *B, float *C) {

  dim3 grid(static_cast<unsigned int>(std::ceil(M / m_block)),
            static_cast<unsigned int>(std::ceil(N / n_block)));
  dim3 block(num_threads);

  BlockTiling2DTransposed<<<grid, block>>>(M, N, K, A, B, C);
}
