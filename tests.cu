#include "matmul.cuh"
#include "gtest/gtest.h"
#include <cmath>
#include <cublas_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

const int M = 4096;
const int N = 4096;
const int K = 4096;

static bool is_close(float test, float desired, float rtol, float atol) {
  return fabs(test - desired) <= (atol + rtol * fabs(desired));
}

static void cmp(const float *test, const float *desired) {
  int failures = 0;
  for (int i = 0; i < M * N && failures < 20; i++) {
    int row = i / N;
    int col = i % N;
    if (!is_close(test[i], desired[i], 1e-2, 1e-3)) {
      ADD_FAILURE() << "pos (" << row << ", " << col
                    << ") doesn't match; expected " << desired[i]
                    << " got: " << test[i]
                    << "; diff: " << desired[i] - test[i];
      failures++;
    }
  }
}

class MatmulTest : public testing::Test {
protected:
  MatmulTest() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    auto ptrs = init_matrixes(M, N, K, left, right, result);
    devA = ptrs[0];
    devB = ptrs[1];
    devC = ptrs[2];

    matmulCublas(M, N, K, devA, devB, devC, handle);
    cudaMemcpy(expected, devC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaMemset(devC, 0, sizeof(float) * M * N);
  }

  alignas(64) float left[M * K];
  alignas(64) float right[K * N];
  alignas(64) float expected[M * N];
  alignas(64) float result[M * N];

  float *devA, *devB, *devC;
};

TEST_F(MatmulTest, Naive) {
  matmulNaive(M, N, K, devA, devB, devC);
  cudaDeviceSynchronize();
  cudaMemcpy(result, devC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  cmp(result, expected);
}

TEST_F(MatmulTest, GMEMCoalesced) {
  matmulGMEMCoalesced(M, N, K, devA, devB, devC);
  cudaDeviceSynchronize();
  cudaMemcpy(result, devC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  cmp(result, expected);
}

TEST_F(MatmulTest, SMEMCoalesced) {
  matmulSMEMCoalesced(M, N, K, devA, devB, devC);
  cudaDeviceSynchronize();
  cudaMemcpy(result, devC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  cmp(result, expected);
}

TEST_F(MatmulTest, BlockTiling1D) {
  matmulBlockTiling1D(M, N, K, devA, devB, devC);
  cudaDeviceSynchronize();
  cudaMemcpy(result, devC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  cmp(result, expected);
}

TEST_F(MatmulTest, BlockTiling2D) {
  matmulBlockTiling2D(M, N, K, devA, devB, devC);
  cudaDeviceSynchronize();
  cudaMemcpy(result, devC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  cmp(result, expected);
}

TEST_F(MatmulTest, BlockTiling2DTransposed) {
  matmulBlockTiling2DTransposed(M, N, K, devA, devB, devC);
  cudaDeviceSynchronize();
  cudaMemcpy(result, devC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  cmp(result, expected);
}

TEST_F(MatmulTest, VectorizedGMEM) {
  matmulVectorizedGMEM(M, N, K, devA, devB, devC);
  cudaDeviceSynchronize();
  cudaMemcpy(result, devC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  cmp(result, expected);
}

TEST_F(MatmulTest, Warptiling) {
  matmulWarptiling(M, N, K, devA, devB, devC);
  cudaDeviceSynchronize();
  cudaMemcpy(result, devC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  cmp(result, expected);
}
