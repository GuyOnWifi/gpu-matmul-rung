#include "cublas_v2.h"

void matmulCublas(int M, int N, int K, const float *A, const float *B, float *C,
                  cublasHandle_t handle) {
  float alpha = 1.0;
  float beta = 0.0;

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K,
              &beta, C, N);
}
