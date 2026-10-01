const int BLOCKSIZE = 32;

__global__ void SMEMCoalescedKernel(int M, int N, int K, const float *A,
                                    const float *B, float *C) {
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  int threadRow = threadIdx.x / BLOCKSIZE;
  int threadCol = threadIdx.x % BLOCKSIZE;

  A += blockIdx.x * BLOCKSIZE * K;
  B += blockIdx.y * BLOCKSIZE;
  C += blockIdx.x * BLOCKSIZE * N + blockIdx.y * BLOCKSIZE;

  float acc = 0.0f;
  for (int blk = 0; blk < K; blk += BLOCKSIZE) {
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    __syncthreads();

    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    for (int i = 0; i < BLOCKSIZE; i++) {
      acc += As[threadRow * BLOCKSIZE + i] * Bs[i * BLOCKSIZE + threadCol];
    }

    __syncthreads();
  }
  C[threadRow * N + threadCol] = acc;
}

void matmulSMEMCoalesced(int M, int N, int K, const float *A, const float *B,
                         float *C) {

  dim3 grid(static_cast<unsigned int>(std::ceil(M / 32.0)),
            static_cast<unsigned int>(std::ceil(N / 32.0)));
  dim3 block(32 * 32);

  SMEMCoalescedKernel<<<grid, block>>>(M, N, K, A, B, C);
}
