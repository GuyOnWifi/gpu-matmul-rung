const int BLOCKSIZE = 32;

__global__ void GMEMCoalescedKernel(int M, int N, int K, const float *A,
                                    const float *B, float *C) {
  const uint r = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const uint c = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  float acc = 0.0f;
  for (uint i = 0; i < K; i++) {
    acc += A[r * K + i] * B[i * N + c];
  }

  C[r * N + c] = acc;
}

void matmulGMEMCoalesced(int M, int N, int K, const float *A, const float *B,
                         float *C) {

  dim3 grid(static_cast<unsigned int>(std::ceil(M / 32.0)),
            static_cast<unsigned int>(std::ceil(N / 32.0)));
  dim3 block(32 * 32);

  GMEMCoalescedKernel<<<grid, block>>>(M, N, K, A, B, C);
}
