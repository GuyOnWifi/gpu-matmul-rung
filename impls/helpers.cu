#include <array>
#include <random>

std::array<float *, 3> init_matrixes(int M, int N, int K, float *A, float *B,
                                     const float *C) {
  std::mt19937 mt(1337);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // arr 1
  for (int r = 0; r < M; r++) {
    for (int i = 0; i < K; i++) {
      A[r * K + i] = dist(mt);
    }
  }

  // arr 2
  for (int i = 0; i < K; i++) {
    for (int c = 0; c < N; c++) {
      B[i * N + c] = dist(mt);
    }
  }

  float *devA, *devB, *devC;

  cudaMalloc(&devA, M * K * sizeof(float));
  cudaMalloc(&devB, K * N * sizeof(float));
  cudaMalloc(&devC, M * N * sizeof(float));

  cudaMemcpy(devA, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(devB, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(devC, C, M * N * sizeof(float), cudaMemcpyHostToDevice);

  return {devA, devB, devC};
}

std::array<float *, 3> init_matrixes_alloc(int M, int N, int K) {
  float *A, *B, *C;
  A = (float *)malloc(M * K * sizeof(float));
  B = (float *)malloc(K * N * sizeof(float));

  std::mt19937 mt(1337);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // arr 1
  for (int r = 0; r < M; r++) {
    for (int i = 0; i < K; i++) {
      A[r * K + i] = dist(mt);
    }
  }

  // arr 2
  for (int i = 0; i < K; i++) {
    for (int c = 0; c < N; c++) {
      B[i * N + c] = dist(mt);
    }
  }

  float *devA, *devB, *devC;

  cudaMalloc(&devA, M * K * sizeof(float));
  cudaMalloc(&devB, K * N * sizeof(float));
  cudaMalloc(&devC, M * N * sizeof(float));

  cudaMemcpy(devA, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(devB, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(devC, 0, M * N * sizeof(float));

  free(A);
  free(B);

  return {devA, devB, devC};
}

std::array<float *, 3> shared_matrixes(int M, int N, int K) {
  static std::array<float *, 3> ptrs = init_matrixes_alloc(M, N, K);
  return ptrs;
}
