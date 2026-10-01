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

const int M = 4096;
const int N = 4096;
const int K = 4096;

int main(int argc, char **argv) {

  auto ptrs = init_matrixes_alloc(M, N, K);
  const float *A = ptrs[0];
  const float *B = ptrs[1];
  float *C = ptrs[2];

  matmulWarptiling(M, N, K, A, B, C);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "%s\n", cudaGetErrorString(err));
    return 1;
  }
  return 0;
}
