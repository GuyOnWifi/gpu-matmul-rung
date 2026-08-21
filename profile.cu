// Minimal driver for ncu. One launch, no warmup, no benchmarking loop.
// Deliberately small (default 1024^3) so profiling doesn't monopolise the GPU
// that is also driving the display.
//
//   ./build/profile               -> BlockTiling1D at 1024^3
//   ./build/profile smem          -> SMEMCoalesced at 1024^3
//   ./build/profile 1d 2048       -> BlockTiling1D at 2048^3
#include "matmul.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char **argv) {
  const char *which = argc > 1 ? argv[1] : "1d";
  const int S = argc > 2 ? std::atoi(argv[2]) : 1024;
  const int M = S, N = S, K = S;

  auto ptrs = init_matrixes_alloc(M, N, K);
  const float *A = ptrs[0];
  const float *B = ptrs[1];
  float *C = ptrs[2];

  if (strcmp(which, "naive") == 0) {
    matmulNaive(M, N, K, A, B, C);
  } else if (strcmp(which, "gmem") == 0) {
    matmulGMEMCoalesced(M, N, K, A, B, C);
  } else if (strcmp(which, "smem") == 0) {
    matmulSMEMCoalesced(M, N, K, A, B, C);
  } else if (strcmp(which, "1d") == 0) {
    matmulBlockTiling1D(M, N, K, A, B, C);
  } else {
    fprintf(stderr, "unknown kernel '%s' (naive|gmem|smem|1d)\n", which);
    return 1;
  }

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "%s\n", cudaGetErrorString(err));
    return 1;
  }
  printf("%s at %d^3 ok\n", which, S);
  return 0;
}
