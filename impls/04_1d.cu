// These are the dimensions of each block.
const int m_block = 64;
const int n_block = 64;
const int k_block = 8;

// Each block gets m_block * k_block / m_thread blocks.m_thread is meant to give
// each thread more reuse, by calculating a column of m_thread results
const int m_thread = 8;

__global__ void BlockTiling1D(int M, int N, int K, const float *A,
                              const float *B, float *C) {

  __shared__ float As[m_block * k_block];
  __shared__ float Bs[k_block * n_block];

  // Indices for each block
  int trow_a = threadIdx.x / k_block;
  int tcol_a = threadIdx.x % k_block;

  int trow_b = threadIdx.x / n_block;
  int tcol_b = threadIdx.x % n_block;

  int trow_c = threadIdx.x / n_block;
  int tcol_c = threadIdx.x % n_block;

  A += blockIdx.x * m_block * K;
  B += blockIdx.y * n_block;
  C += blockIdx.x * m_block * N + blockIdx.y * n_block;

  // Accumulate m_thread items
  float acc[m_thread] = {0.0f};
  for (int blk = 0; blk < K; blk += k_block) {
    As[trow_a * k_block + tcol_a] = A[trow_a * K + tcol_a];
    Bs[trow_b * n_block + tcol_b] = B[trow_b * N + tcol_b];

    __syncthreads();

    A += k_block;
    B += k_block * N;

    for (int res_idx = 0; res_idx < m_thread; res_idx++) {
      for (int dot_idx = 0; dot_idx < k_block; dot_idx++) {
        acc[res_idx] += As[(trow_c * m_thread + res_idx) * k_block + dot_idx] *
                        Bs[dot_idx * n_block + tcol_c];
      }
    }

    __syncthreads();
  }

  // Write the accumulator to memory
  for (int i = 0; i < m_thread; i++) {
    C[(trow_c * m_thread + i) * N + tcol_c] = acc[i];
  }
}

void matmulBlockTiling1D(int M, int N, int K, const float *A, const float *B,
                         float *C) {

  dim3 grid(static_cast<unsigned int>(std::ceil(M / m_block)),
            static_cast<unsigned int>(std::ceil(N / n_block)));
  dim3 block(m_block * n_block / m_thread);

  BlockTiling1D<<<grid, block>>>(M, N, K, A, B, C);
}
